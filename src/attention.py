import math
import torch


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, seq_len_q, d_k)
    K: (batch, seq_len_k, d_k)
    V: (batch, seq_len_k, d_v)
    mask: tensor compatível com (batch, seq_len_q, seq_len_k)
          1 = pode olhar
          0 = deve mascarar
    """

    d_k = Q.size(-1)

    # scores = QK^T / sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # aplica a máscara, se existir
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # transforma scores em probabilidades
    attention_weights = torch.softmax(scores, dim=-1)

    # combina os valores V usando os pesos de atenção
    output = torch.matmul(attention_weights, V)

    return output, attention_weights


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 1
    seq_len = 3
    d_model = 4

    Q = torch.randn(batch_size, seq_len, d_model)
    K = torch.randn(batch_size, seq_len, d_model)
    V = torch.randn(batch_size, seq_len, d_model)

    output, weights = scaled_dot_product_attention(Q, K, V)

    print("Q shape:", Q.shape)
    print("K shape:", K.shape)
    print("V shape:", V.shape)
    print("Output shape:", output.shape)
    print("Weights shape:", weights.shape)
    print("\nOutput:\n", output)
    print("\nWeights:\n", weights)