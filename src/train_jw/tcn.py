"""Causal dilated TCN — IO(채널) → 센서(채널) 시계열 회귀.

causal: 출력 t 는 과거 입력만 사용(미래 누수 없음).
dilated: dilation 1,2,4,8… 로 적은 층으로 긴 과거(=지연 lag) 수용.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    def __init__(self, c_in, c_out, k, dilation):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, k, padding=self.pad, dilation=dilation)

    def forward(self, x):                      # x: (B, C, T)
        y = self.conv(x)
        return y[..., :-self.pad] if self.pad else y


class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(c_in, c_out, k, dilation)
        self.conv2 = CausalConv1d(c_out, c_out, k, dilation)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None

    def forward(self, x):
        y = self.drop(self.act(self.conv1(x)))
        y = self.drop(self.act(self.conv2(y)))
        res = x if self.down is None else self.down(x)
        return self.act(y + res)


class IOTCN(nn.Module):
    """입력 (B, n_io_feat, T) → 출력 (B, n_sensor, T) 예측."""
    def __init__(self, n_in, n_out, channels=64, n_blocks=5, k=3, dropout=0.1):
        super().__init__()
        blocks = []
        c_prev = n_in
        for i in range(n_blocks):
            blocks.append(TCNBlock(c_prev, channels, k, dilation=2 ** i,
                                   dropout=dropout))
            c_prev = channels
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Conv1d(channels, n_out, 1)
        # 수용영역(=잡을 수 있는 최대 지연 lag) 기록
        self.receptive_field = 1 + 2 * (k - 1) * (2 ** n_blocks - 1)

    def forward(self, x):
        return self.head(self.tcn(x))
