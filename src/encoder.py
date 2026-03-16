import torch
import torch.nn as nn

from attention import scaled_dot_product_attention
from ffn import FeedForward
from add_norm import AddNorm


class EncoderBlock(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.add_norm1 = AddNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.add_norm2 = AddNorm(d_model)

    def forward(self, x, mask=None):
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        attention_output, _ = scaled_dot_product_attention(Q, K, V, mask)
        x = self.add_norm1(x, attention_output)

        ffn_output = self.ffn(x)
        x = self.add_norm2(x, ffn_output)

        return x


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 1
    seq_len = 4
    d_model = 512

    x = torch.randn(batch_size, seq_len, d_model)

    encoder_block = EncoderBlock(d_model=d_model, d_ff=2048)
    output = encoder_block(x)

    print("Entrada shape:", x.shape)
    print("Saída shape:", output.shape)