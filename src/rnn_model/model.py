import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class FakeNewsDetector(nn.Module):
    def __init__(
        self,
    ):
        super(FakeNewsDetector, self).__init__()
        self.name = "FakeNewsDetector"
        # self.embedding = SentenceTransformer("all-MiniLM-L6-v2")
        self.rnn = nn.RNN(
            input_size=384,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
        initial_state = torch.zeros(4, x.size(0), 128)
        rnn_output, _ = self.rnn(x, initial_state)
        rnn_output = rnn_output[:, -1, :]
        x = self.fc1(rnn_output)
        x = self.fc2(x)
        return self.sigmoid(x).squeeze(1)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
