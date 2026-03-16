import torch
import torch.nn as nn


class AddNorm(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer_output):
        return self.layer_norm(x + sublayer_output)


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 1
    seq_len = 3
    d_model = 512

    x = torch.randn(batch_size, seq_len, d_model)
    sublayer_output = torch.randn(batch_size, seq_len, d_model)

    add_norm = AddNorm(d_model=d_model)
    output = add_norm(x, sublayer_output)

    print("Entrada shape:", x.shape)
    print("Saída da subcamada shape:", sublayer_output.shape)
    print("Saída final shape:", output.shape)