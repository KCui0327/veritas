import torch
import torch.nn as nn
from torchtext.vocab import GloVe


# Model definition
class FakeNewsDetector(nn.Module):
    def __init__(
        self,
    ):
        super(FakeNewsDetector, self).__init__()
        self.name = "FakeNewsDetector"
        glove = GloVe(name="6B", dim=300)
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=glove.vectors,
            freeze=True,
        )
        self.rnn = nn.RNN(
            input_size=300,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        initial_state = torch.zeros(4, x.size(0), 128)
        rnn_output, _ = self.rnn(embedded, initial_state)
        rnn_output = rnn_output[:, -1, :]
        x = self.fc1(rnn_output)
        x = self.fc2(x)
        return self.sigmoid(x).squeeze(1)
