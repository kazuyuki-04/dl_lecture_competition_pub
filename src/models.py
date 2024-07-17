import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 64
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X)

class HybridConvLSTMClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 64,
        lstm_dim: int = 128,
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.lstm = nn.LSTM(hid_dim, lstm_dim, batch_first=True, bidirectional=True)

        self.head = nn.Sequential(
            nn.Linear(2 * lstm_dim, num_classes),  # 2*lstm_dim because of bidirectional
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)
        X = X.transpose(1, 2)  # (batch_size, seq_length, features)
        X, _ = self.lstm(X)
        X = X[:, -1, :]  # Last time step output
        return self.head(X)

class DeeperConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 64,
        num_blocks: int = 4
    ) -> None:
        super().__init__()

        blocks = [ConvBlock(in_channels, hid_dim)]
        for _ in range(1, num_blocks):
            blocks.append(ConvBlock(hid_dim, hid_dim))

        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)
        return self.head(X)

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
        num_groups: int = 8,
        use_groupnorm: bool = True,
        use_maxpool: bool = False,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_groupnorm = use_groupnorm
        self.use_maxpool = use_maxpool

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm_shortcut = nn.BatchNorm1d(num_features=out_dim)

        self.groupnorm0 = nn.GroupNorm(num_groups=num_groups, num_channels=out_dim)
        self.groupnorm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_dim)

        self.dropout = nn.Dropout1d(p_drop)  # Spatial Dropout

        if in_dim != out_dim:
            self.shortcut = nn.Conv1d(in_dim, out_dim, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        if self.use_maxpool:
            self.maxpool = nn.MaxPool1d(kernel_size=2)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(X)
        residual = self.batchnorm_shortcut(residual)

        X = self.conv0(X)
        if self.use_groupnorm:
            X = self.groupnorm0(X)
        else:
            X = self.batchnorm0(X)
        X = F.leaky_relu(X)

        X = self.conv1(X)
        if self.use_groupnorm:
            X = self.groupnorm1(X)
        else:
            X = self.batchnorm1(X)
        X = F.leaky_relu(X)

        X = X + residual  # skip connection
        X = self.dropout(X)

        if self.use_maxpool:
            X = self.maxpool(X)

        return X