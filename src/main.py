import torch
import torch.nn as nn

from encoder import EncoderBlock
from decoder import DecoderBlock


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=32, d_ff=64):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

        self.encoder = EncoderBlock(d_model=d_model, d_ff=d_ff)
        self.decoder = DecoderBlock(d_model=d_model, d_ff=d_ff, vocab_size=vocab_size)

    def encode(self, encoder_input_ids):
        x = self.embedding(encoder_input_ids)
        z = self.encoder(x)
        return z

    def decode(self, decoder_input_ids, z):
        y = self.embedding(decoder_input_ids)
        _, probs = self.decoder(y, z)
        return probs

    def forward(self, encoder_input_ids, decoder_input_ids):
        z = self.encode(encoder_input_ids)
        probs = self.decode(decoder_input_ids, z)
        return probs


if __name__ == "__main__":
    torch.manual_seed(42)

    vocab = {
        "<START>": 0,
        "<EOS>": 1,
        "Thinking": 2,
        "Machines": 3,
        "Pensando": 4,
        "Maquinas": 5
    }

    id_to_token = {idx: token for token, idx in vocab.items()}
    vocab_size = len(vocab)

    model = SimpleTransformer(vocab_size=vocab_size, d_model=32, d_ff=64)

    # frase simulada de entrada: "Thinking Machines"
    encoder_input = torch.tensor([[vocab["Thinking"], vocab["Machines"]]])

    # codifica uma vez
    z = model.encode(encoder_input)

    # decoder inicia com <START>
    decoder_input = torch.tensor([[vocab["<START>"]]])

    max_steps = 6
    generated_tokens = []

    for step in range(max_steps):
        probs = model.decode(decoder_input, z)

        # pega a distribuição do último token gerado
        next_token_probs = probs[:, -1, :]
        next_token_id = torch.argmax(next_token_probs, dim=-1).item()

        generated_tokens.append(next_token_id)

        # para se gerar <EOS>
        if next_token_id == vocab["<EOS>"]:
            break

        next_token_tensor = torch.tensor([[next_token_id]])
        decoder_input = torch.cat([decoder_input, next_token_tensor], dim=1)

    print("Encoder input IDs:", encoder_input.tolist())
    print("Decoder input final IDs:", decoder_input.tolist())
    print("Tokens gerados:", [id_to_token[token_id] for token_id in generated_tokens])