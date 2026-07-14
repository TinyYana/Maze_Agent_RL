"""對抗式訓練的神經網路：ConvGRU 規劃核心 + 空間策略頭。

設計 (與使用者討論定案)：
- 共用骨架：Conv3x3 編碼 -> ConvGRU cell 對同一觀測內部迭代 K 次 (DRC 式
  「規劃迭代」，非跨時間步記憶，因此標準 MaskablePPO 可直接訓練)。
  每迭代一次，資訊沿棋盤多傳播一圈，等於讓網路「多想幾步」。
- Master 頭：Conv1x1 -> 5x15x15 動作熱度圖 (每格 x 每種編輯)，攤平成
  Discrete(1125) 的 masked categorical；價值頭走 AlphaZero 式壓縮。
- Player 頭：取玩家所在格特徵 + 全局平均池化 -> FC -> Discrete(4)。

尺寸：128 通道、K=8，約 110 萬參數/agent (3090 版)。
"""
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

CHANNELS = 128
PLAN_ITERS = 8


class ConvGRUCell(nn.Module):
    """卷積版 GRU cell (3x3 門)，hidden 與輸入同為 CxHxW"""

    def __init__(self, ch):
        super().__init__()
        self.gates = nn.Conv2d(2 * ch, 2 * ch, 3, padding=1)  # update z + reset r
        self.cand = nn.Conv2d(2 * ch, ch, 3, padding=1)

    def forward(self, x, h):
        zr = torch.sigmoid(self.gates(torch.cat([x, h], dim=1)))
        z, r = zr.chunk(2, dim=1)
        h_new = torch.tanh(self.cand(torch.cat([x, r * h], dim=1)))
        return (1 - z) * h + z * h_new


class PlannerCore(nn.Module):
    """編碼 + ConvGRU 內部迭代 K 次，輸出 CxHxW 的規劃特徵圖"""

    def __init__(self, in_channels, ch=CHANNELS, iters=PLAN_ITERS):
        super().__init__()
        self.iters = iters
        self.encoder = nn.Sequential(nn.Conv2d(in_channels, ch, 3, padding=1), nn.ReLU())
        self.cell = ConvGRUCell(ch)

    def forward(self, obs_grid):
        x = self.encoder(obs_grid)
        h = torch.zeros_like(x)
        for _ in range(self.iters):
            h = self.cell(x, h)
        return h


class MasterFeaturesExtractor(BaseFeaturesExtractor):
    """輸出攤平的規劃特徵圖 (C*15*15)，由 MasterPolicy 的頭 reshape 回空間形狀"""

    def __init__(self, observation_space: spaces.Box, ch=CHANNELS, iters=PLAN_ITERS):
        n, h, w = observation_space.shape
        super().__init__(observation_space, features_dim=ch * h * w)
        self.core = PlannerCore(n, ch, iters)
        self.spatial_shape = (ch, h, w)

    def forward(self, obs):
        return self.core(obs).flatten(start_dim=1)


class _SpatialActionHead(nn.Module):
    """flat 特徵 -> reshape -> Conv1x1 -> n_edit x H x W logits (攤平)"""

    def __init__(self, spatial_shape, n_edit_types):
        super().__init__()
        self.spatial_shape = spatial_shape
        self.head = nn.Conv2d(spatial_shape[0], n_edit_types, 1)

    def forward(self, latent):
        x = latent.view(-1, *self.spatial_shape)
        return self.head(x).flatten(start_dim=1)


class _SpatialValueHead(nn.Module):
    """flat 特徵 -> reshape -> Conv1x1 壓成 2 通道 -> FC -> V(s)"""

    def __init__(self, spatial_shape):
        super().__init__()
        self.spatial_shape = spatial_shape
        ch, h, w = spatial_shape
        self.squeeze = nn.Sequential(nn.Conv2d(ch, 2, 1), nn.ReLU(), nn.Flatten())
        self.fc = nn.Sequential(nn.Linear(2 * h * w, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, latent):
        x = latent.view(-1, *self.spatial_shape)
        return self.fc(self.squeeze(x))


class MasterPolicy(MaskableActorCriticPolicy):
    """Maze Master 的空間策略：動作 = Discrete(n_edit_types * 15 * 15)。

    net_arch 固定為空 (latent 就是攤平的特徵圖)，
    action_net / value_net 換成空間卷積頭後重建 optimizer。
    """

    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        kwargs["net_arch"] = []
        kwargs.setdefault("features_extractor_class", MasterFeaturesExtractor)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        spatial_shape = self.features_extractor.spatial_shape
        n_cells = spatial_shape[1] * spatial_shape[2]
        assert action_space.n % n_cells == 0, "動作數必須是格子數的整數倍"
        n_edit_types = action_space.n // n_cells

        self.action_net = _SpatialActionHead(spatial_shape, n_edit_types)
        self.value_net = _SpatialValueHead(spatial_shape)
        # 換頭之後 optimizer 還指著舊參數，必須重建
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )


class PlayerFeaturesV2(BaseFeaturesExtractor):
    """玩家特徵：規劃圖上取「玩家所在格」向量 + 全局平均池化，串接後過 FC。

    玩家位置從觀測的 player 通道 (index 1) 讀出，用它當空間權重做加權和，
    等價於 gather 玩家格特徵，且對 batch 友善。
    """

    PLAYER_PLANE = 1

    def __init__(self, observation_space: spaces.Box, ch=CHANNELS, iters=PLAN_ITERS, features_dim=256):
        super().__init__(observation_space, features_dim)
        n = observation_space.shape[0]
        self.core = PlannerCore(n, ch, iters)
        self.fc = nn.Sequential(nn.Linear(2 * ch, features_dim), nn.ReLU())

    def forward(self, obs):
        h = self.core(obs)
        player_plane = obs[:, self.PLAYER_PLANE : self.PLAYER_PLANE + 1]
        at_player = (h * player_plane).sum(dim=(2, 3))
        pooled = h.mean(dim=(2, 3))
        return self.fc(torch.cat([at_player, pooled], dim=1))
