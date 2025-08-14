import torch
import torch.nn as nn


class FakeNewsDetector(nn.Module):
    def __init__(self):
        super(FakeNewsDetector, self).__init__()
        self.name = "RNN_No_Embedding"

        self.dropout = nn.Dropout(0.2)
        self.rnn = nn.GRU(
            input_size=50,
            hidden_size=32,
            bidirectional=True,
            batch_first=True,
        )
        self.fc1 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        initial_state = torch.zeros(2, x.size(0), 32, device=x.device)
        output, _ = self.rnn(x, initial_state)
        output = torch.cat(
            [torch.max(output, dim=1)[0], torch.mean(output, dim=1)], dim=1
        )
        output = self.dropout(output)
        x = self.fc1(output)

        return self.sigmoid(x).squeeze(1)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
