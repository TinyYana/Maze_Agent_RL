"""NN 玩家的神經網路定義。

SB3 預設的 NatureCNN 是為 84x84 Atari 畫面設計 (第一層 kernel 8x8 stride 4)，
在 15x15 的格子盤面上會一次跨過好幾格牆，糊掉「這格是牆、隔壁是路」的細節。
這裡改用 3x3 小 kernel 的三層 CNN，並在攤平後串接狀態向量 (HP/鎚子/剩餘時間)。

架構: 4x15x15 -> Conv3x3x32(s1) -> Conv3x3x64(s2) -> Conv3x3x64(s2)
      -> Flatten 1024 -> concat 狀態向量 -> FC 256 (features_dim)
之後由 SB3 的 ActorCriticPolicy 直接接策略頭 (Discrete 4) 與價值頭。
"""
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PlayerFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        n_channels = observation_space["grid"].shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # 用一筆假資料推得攤平後的維度 (15x15 -> 1024)，避免手算寫死
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space["grid"].shape)
            n_flatten = self.cnn(sample).shape[1]

        state_dim = observation_space["state"].shape[0]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + state_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations) -> torch.Tensor:
        cnn_features = self.cnn(observations["grid"])
        combined = torch.cat([cnn_features, observations["state"]], dim=1)
        return self.linear(combined)
