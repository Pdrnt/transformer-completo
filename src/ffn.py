import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 1
    seq_len = 3
    d_model = 512

    x = torch.randn(batch_size, seq_len, d_model)

    ffn = FeedForward(d_model=d_model, d_ff=2048)
    output = ffn(x)

    print("Entrada shape:", x.shape)
    print("Saída shape:", output.shape)