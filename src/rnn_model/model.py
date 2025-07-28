import torch
import torch.nn as nn


class FakeNewsDetector(nn.Module):
    def __init__(
        self,
    ):
        super(FakeNewsDetector, self).__init__()
        self.name = "FakeNewsDetector"
        self.rnn = nn.RNN(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            bidirectional=False,
            batch_first=True,
        )
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        initial_state = torch.zeros(2, x.size(0), 128)
        rnn_output, _ = self.rnn(x, initial_state)
        rnn_output = rnn_output[:, -1, :]
        x = self.fc1(rnn_output)
        x = self.fc2(x)
        return self.sigmoid(x).squeeze(1)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
