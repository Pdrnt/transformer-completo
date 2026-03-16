import torch
import torch.nn as nn

from attention import scaled_dot_product_attention
from ffn import FeedForward
from add_norm import AddNorm


def create_causal_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0)  # (1, seq_len, seq_len)


class DecoderBlock(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, vocab_size=1000):
        super().__init__()

        self.self_q = nn.Linear(d_model, d_model)
        self.self_k = nn.Linear(d_model, d_model)
        self.self_v = nn.Linear(d_model, d_model)

        self.cross_q = nn.Linear(d_model, d_model)
        self.cross_k = nn.Linear(d_model, d_model)
        self.cross_v = nn.Linear(d_model, d_model)

        self.add_norm1 = AddNorm(d_model)
        self.add_norm2 = AddNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.add_norm3 = AddNorm(d_model)

        self.output_linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, y, Z):
        seq_len = y.size(1)
        causal_mask = create_causal_mask(seq_len).to(y.device)

        # 1) Masked Self-Attention
        Q1 = self.self_q(y)
        K1 = self.self_k(y)
        V1 = self.self_v(y)

        self_att_output, _ = scaled_dot_product_attention(Q1, K1, V1, causal_mask)
        y = self.add_norm1(y, self_att_output)

        # 2) Cross-Attention
        Q2 = self.cross_q(y)
        K2 = self.cross_k(Z)
        V2 = self.cross_v(Z)

        cross_att_output, _ = scaled_dot_product_attention(Q2, K2, V2)
        y = self.add_norm2(y, cross_att_output)

        # 3) FFN
        ffn_output = self.ffn(y)
        y = self.add_norm3(y, ffn_output)

        # 4) Projeção para vocabulário + softmax
        logits = self.output_linear(y)
        probs = self.softmax(logits)

        return y, probs


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 1
    target_seq_len = 4
    source_seq_len = 5
    d_model = 512
    vocab_size = 50

    y = torch.randn(batch_size, target_seq_len, d_model)
    Z = torch.randn(batch_size, source_seq_len, d_model)

    decoder_block = DecoderBlock(d_model=d_model, d_ff=2048, vocab_size=vocab_size)
    decoder_output, probs = decoder_block(y, Z)

    print("Entrada do decoder shape:", y.shape)
    print("Memória do encoder Z shape:", Z.shape)
    print("Saída contextualizada do decoder shape:", decoder_output.shape)
    print("Probabilidades no vocabulário shape:", probs.shape)