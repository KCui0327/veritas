import torch
import torch.nn as nn


class FakeNewsDetector(nn.Module):
    def __init__(
        self,
        pretrained_embeddings,
        freeze_embeddings,
        hidden_dim,
        num_layers,
        dropout_rate,
        bidirectional,
        output_dim,
    ):
        super(FakeNewsDetector, self).__init__()
        self.name = "FakeNewsDetector"
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=pretrained_embeddings,
            freeze=freeze_embeddings,
        )
        embed_dim = pretrained_embeddings.size(1)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout_rate)
        rnn_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(rnn_output_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)  # [bs, seq_len, embed_dim]
        rnn_out, hn = self.rnn(embedded)
        # Use the last hidden state (concatenate if bidirectional)
        if self.rnn.bidirectional:
            final_state = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        else:
            final_state = hn[-1]
        x = self.dropout(final_state)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x).squeeze(1)
