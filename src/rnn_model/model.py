import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# LSTM will have 2 layers, each with 128 units.

# Embedding maps each of the 20 000 tokens to a 128-dimensional dense vector.


# Attention module
# The attention module will dynamically weigh each hidden state instead of simply taking the final LSTM output.
# ie.  “look back” at earlier states


def attention_net(lstm_output, final_state):
    # lstm_output: [batch_size, seq_len, hidden_dim*directions]
    # final_state: [batch_size, hidden_dim*directions]
    weights = torch.bmm(lstm_output, final_state.unsqueeze(2)).squeeze(2)
    weights = torch.softmax(weights, dim=1)  # [batch_size, seq_len]
    context = torch.bmm(lstm_output.transpose(1, 2), weights.unsqueeze(2)).squeeze(2)
    return context, weights


# Model definition
class FakeNewsDetector(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_layers,
        dropout_rate,
        bidirectional,
        output_dim,
    ):
        super(FakeNewsDetector, self).__init__()
        self.name = "FakeNewsDetector"
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout_rate)
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(lstm_output_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)  # [bs, seq_len, embed_dim]
        lstm_out, (hn, cn) = self.lstm(embedded)
        # Concatenate final forward and backward hidden states
        if self.lstm.bidirectional:
            final_state = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        else:
            final_state = hn[-1]
        # Attention mechanism
        attn_output, attn_weights = attention_net(lstm_out, final_state)
        x = self.dropout(attn_output)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x).squeeze(1)
