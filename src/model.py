# src/model.py
import torch
import torch.nn as nn
import torchvision.models as models


class CNNBaseline(nn.Module):
    """Single-frame CNN classifier using MobileNetV2 backbone."""

    def __init__(self):
        super(CNNBaseline, self).__init__()
        mobilenet = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )
        self.features = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        # Dataset always returns (batch, seq_len, C, H, W)
        # CNN baseline only needs one frame -> extract frame 0
        if x.dim() == 5:
            x = x[:, 0, :, :, :]   # (batch, C, H, W)

        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)


class CNNLSTMModel(nn.Module):
    """
    CNN+LSTM model for sequence-level deepfake detection.

    Args:
        sequence_length : Number of frames per clip (default 10)
        hidden_size     : LSTM hidden state size (default 256)
        num_layers      : Number of stacked LSTM layers (default 1)
        bidirectional   : Use bidirectional LSTM (default True)
    """

    def __init__(
        self,
        sequence_length=10,
        hidden_size=256,
        num_layers=1,
        bidirectional=True
    ):
        super(CNNLSTMModel, self).__init__()

        mobilenet = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )
        self.features = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.bidirectional = bidirectional
        lstm_output_size = hidden_size * (2 if bidirectional else 1)

        self.lstm = nn.LSTM(
            input_size=1280,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.3 if num_layers > 1 else 0.0
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        # x: (batch, seq_len, C, H, W)
        batch_size, seq_len, C, H, W = x.shape

        # CNN feature extraction
        # NOTE: NO torch.no_grad() here - gradient must flow (Week 2 fix)
        x = x.view(batch_size * seq_len, C, H, W)
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)                        # (batch*seq_len, 1280)

        # Reshape for LSTM
        x = x.view(batch_size, seq_len, -1)     # (batch, seq_len, 1280)

        # Temporal modeling
        lstm_out, _ = self.lstm(x)              # (batch, seq_len, lstm_output_size)
        x = lstm_out[:, -1, :]                  # Take last timestep only

        return self.classifier(x)