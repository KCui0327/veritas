import torch
import torch.nn as nn
import torchtext
from torchtext.vocab import GloVe


class FakeNewsDetector(nn.Module):
    def __init__(self):
        super(FakeNewsDetector, self).__init__()
        self.name = "FakeNewsDetector"
        glove = torchtext.vocab.GloVe(name="6B", dim=300)
        embedding_dim = glove.vectors.shape[1]
        self.embedding = nn.Embedding.from_pretrained(glove.vectors, freeze=True)
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=300,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.fc1 = nn.Linear(embedding_dim * 2, 300)
        self.fc2 = nn.Linear(300, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)

        initial_state = torch.zeros(4, x.size(0), 300, device=x.device)
        cell_state = torch.zeros(4, x.size(0), 300, device=x.device)
        output, _ = self.rnn(x, (initial_state, cell_state))
        output = torch.max(output, dim=1)[0]

        x = self.fc1(output)
        x = self.fc2(x)
        return self.sigmoid(x).squeeze(1)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
