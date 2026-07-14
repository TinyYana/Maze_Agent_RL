"""凍結對手的中央批次推論 VecEnv 包裝器。

原本每個 SubprocVecEnv worker 各自在單執行緒 CPU 上逐筆跑凍結對手的
ConvGRU (K=8, 128ch ≈ 每次 1G MACs)，是整個對抗訓練的吞吐瓶頸
(3090 pod 實測 fps 88)。這裡把對手推論收回主進程：worker 只跑純 Python
遊戲邏輯，對手觀測按「哪個檢查點」分組、批次過一次 GPU。

避免「GPU 跑完換 CPU 跑」的串行乾等：
  - 玩家側 (kind="master")：master_io 用 env_method 一次廣播全部 worker
    (先全部送出再統一收回，worker 之間並行)。
  - Master 側 (kind="player")：對手必須看編輯後的局面 (apply_edit 兩段式)。
    編輯指令一次扇出給全部 worker 再統一收回，12 個 worker 的遊戲邏輯
    真正並行，而不是 12 次串行 IPC 往返。

注意：不要做「step_wait 先送請求、step_async 再收」的跨步預取——
MaskablePPO 每步會呼叫 get_action_masks -> env_method("action_masks")，
走同一條 pipe，掛著未收回的請求會讓回覆錯位 (實測 shape 錯誤炸掉)。

決策時序與 env 內建模式完全一致，等價性由 test_adversarial.py 驗證。
"""
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper

# SubprocVecEnv 的 worker 協定 (subproc_vec_env.py)：
#   remote.send(("env_method", (方法名, args, kwargs))) -> remote.recv() 拿回傳值
_ENV_METHOD = "env_method"


class BatchedFrozenOpponent(VecEnvWrapper):
    """assignments: 每個 env index 對應的凍結對手檢查點路徑 (不含 .zip)。
    kind: "master" = 對手是 Maze Master (venv 裝 PlayerEnvV2)
          "player" = 對手是玩家 (venv 裝 MasterEnv，opponent=("external", None))
    assignment=None 的 env 沒有對手 (玩家側的靜態迷宮混訓)，直接跳過。
    """

    def __init__(self, venv, assignments, kind, device="auto"):
        super().__init__(venv)
        from sb3_contrib import MaskablePPO

        assert kind in ("master", "player")
        assert len(assignments) == venv.num_envs
        if kind == "player":
            assert all(assignments), "MasterEnv 一定要有對手玩家"
        self.kind = kind
        self.models = {
            path: MaskablePPO.load(path, device=device)
            for path in dict.fromkeys(p for p in assignments if p is not None)
        }
        self.groups = {}  # path -> [env indices]
        for i, path in enumerate(assignments):
            if path is not None:
                self.groups.setdefault(path, []).append(i)
        self.active = sorted(i for idxs in self.groups.values() for i in idxs)

    # ------------------------------------------------------------------
    # worker IPC
    # ------------------------------------------------------------------

    @property
    def _remotes(self):
        return getattr(self.venv, "remotes", None)

    def _apply_edits(self, actions):
        """Master 側：編輯指令一次扇出、統一收回，worker 並行執行"""
        remotes = self._remotes
        if remotes is None:
            return {
                i: self.venv.env_method("apply_edit", int(actions[i]), indices=[i])[0]
                for i in range(self.num_envs)
            }
        for i in range(self.num_envs):
            remotes[i].send((_ENV_METHOD, ("apply_edit", (int(actions[i]),), {})))
        return {i: remotes[i].recv() for i in range(self.num_envs)}

    # ------------------------------------------------------------------
    # 推論與 VecEnv 介面
    # ------------------------------------------------------------------

    def _predict(self, ios):
        """ios = {env_idx: (obs, mask)}；按檢查點分組批次推論，回傳 {env_idx: 動作}"""
        out = {}
        for path, idxs in self.groups.items():
            obs = np.stack([ios[i][0] for i in idxs])
            masks = np.stack([ios[i][1] for i in idxs])
            acts, _ = self.models[path].predict(obs, action_masks=masks, deterministic=False)
            out.update(zip(idxs, acts))
        return out

    def step_async(self, actions):
        if self.kind == "master":
            # env_method 是「全部送出→統一收回」，worker 之間並行
            ios = dict(zip(self.active, self.venv.env_method("master_io", indices=self.active)))
            m_acts = self._predict(ios) if self.active else {}
            merged = [
                np.array([m_acts[i], int(actions[i])], dtype=np.int64) if i in m_acts else int(actions[i])
                for i in range(self.num_envs)
            ]
            self.venv.step_async(merged)
        else:
            opp = self._predict(self._apply_edits(actions))
            self.venv.step_async(np.array([opp[i] for i in range(self.num_envs)], dtype=np.int64))

    def step_wait(self):
        return self.venv.step_wait()

    def reset(self):
        return self.venv.reset()
