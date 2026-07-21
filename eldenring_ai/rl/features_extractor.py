"""
features_extractor.py
"""
import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class EldenRingCNN(BaseFeaturesExtractor):
    """CNN for the 256x256x12 frame stack, sized for this resolution
    instead of NatureCNN's 84x84 default. Receives channel-first
    (12,256,256) float tensor, already normalized to [0,1] by SB3."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]  # 12 (FRAME_STACK)

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class EldenRingExtractor(BaseFeaturesExtractor):
    """Combined extractor for the Dict observation space:
    'frame' -> EldenRingCNN, everything else -> flattened (same as
    SB3's default CombinedExtractor behavior for non-image keys)."""

    def __init__(self, observation_space: gym.spaces.Dict, cnn_output_dim: int = 256):
        super().__init__(observation_space, features_dim=1)  # placeholder, fixed below

        extractors = {}
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            if key == "frame":
                extractors[key] = EldenRingCNN(subspace, features_dim=cnn_output_dim)
                total_concat_size += cnn_output_dim
            else:
                extractors[key] = nn.Flatten()
                total_concat_size += int(np.prod(subspace.shape))

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations: dict) -> th.Tensor:
        encoded = [self.extractors[key](observations[key]) for key in self.extractors]
        return th.cat(encoded, dim=1)
