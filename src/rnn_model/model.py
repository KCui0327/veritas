import torch
import torch.nn as nn
import torchtext
from torchtext.vocab import GloVe


class FakeNewsDetector(nn.Module):
    def __init__(self):
        super(FakeNewsDetector, self).__init__()
        self.name = "FakeNewsDetector"
        glove = torchtext.vocab.GloVe(
            name="6B", dim=300
        )  # can also call this at start of main function
        EmbeddingDim = glove.vectors.shape[1]
        self.embedding = nn.Embedding.from_pretrained(glove.vectors)
        self.rnn = nn.LSTM(
            input_size=EmbeddingDim,
            hidden_size=300,  # keeping it same as glove dim for simpler vanilla RNN
            num_layers=2,  # usually 2 is enough for this vanilla RNN
            bidirectional=True,
            batch_first=True,
        )
        # For bidirectional RNN, output size is hidden_size * 2
        self.fc1 = nn.Linear(600, 300)  # hidden_size * 2 (bidirectional)
        self.fc2 = nn.Linear(300, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)  # here the embedding vector is changed
        # For bidirectional RNN with 2 layers means 4 hidden states
        initial_state = torch.zeros(4, x.size(0), 300, device=x.device)
        cell_state = torch.zeros(4, x.size(0), 300, device=x.device)
        output, _ = self.rnn(x, (initial_state, cell_state))
        output = torch.max(output, dim=1)[0]
        x = self.fc1(output)
        x = self.fc2(x)  # passing hidden state here
        return self.sigmoid(x).squeeze(1)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
