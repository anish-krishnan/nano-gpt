from enum import Enum
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

# for reproducibility
torch.manual_seed(1337)

# hyperparameters
BATCH_SIZE = 32  # number of independent sequences to process in parallel
BLOCK_SIZE = 8  # maximum context length for predictions
LEARNING_RATE = 1e-3
NUM_LEARNING_STEPS = 10_000
TRAINING_RATIO = 0.9  # amount of input data to be used for training split


class Split(Enum):
    TRAIN = 0
    TEST = 1


class DataManager:
    def __init__(self, data, training_ratio=TRAINING_RATIO):
        n = int(training_ratio * len(data))
        self.data = data
        self.train_data = data[:n]
        self.val_data = data[n:]

    def read_file(filename) -> str:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()

    def get_data_for_split(self, split):
        match split:
            case Split.TRAIN:
                return self.train_data
            case Split.TEST:
                return self.val_data


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def generate_from_empty_context(self):
        context = torch.zeros((1, 1), dtype=torch.long)
        return self.generate(context, max_new_tokens=100)[0].tolist()


class Tokenizer:
    def __init__(self, text):
        self.vocab = sorted(list(set(text)))

    def get_vocabulary(self) -> tuple[list, int]:
        return self.vocab, len(self.vocab)

    def get_encoder_and_decoder(self):
        token_to_int = {token: idx for idx, token in enumerate(self.vocab)}
        int_to_token = {idx: token for idx, token in enumerate(self.vocab)}

        encoder = lambda s: [token_to_int[c] for c in s]
        decoder = lambda L: "".join(int_to_token[x] for x in L)

        return encoder, decoder


def read_file(filename) -> str:
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()


def get_batch(data: Tensor, batch_size, block_size) -> tuple[Tensor, Tensor]:
    offsets = torch.randint(len(data) - block_size, (batch_size,))
    contexts = torch.stack([data[offset : offset + block_size] for offset in offsets])
    targets = torch.stack(
        [data[offset + 1 : offset + block_size + 1] for offset in offsets]
    )
    return contexts, targets


def main():
    text = read_file("input.txt")
    tokenizer = Tokenizer(text)
    vocab, vocab_size = tokenizer.get_vocabulary()
    encode, decode = tokenizer.get_encoder_and_decoder()
    data = torch.tensor(encode(text), dtype=torch.long)
    data_manager = DataManager(data)

    model = BigramLanguageModel(vocab_size)
    print(f"pre-training generation: {decode(model.generate_from_empty_context())}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    train_data = data_manager.get_data_for_split(Split.TRAIN)
    for _ in range(NUM_LEARNING_STEPS):
        contexts, targets = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE)

        logits, loss = model(contexts, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"final loss: {loss.item()}")

    print(f"post-training generation: {decode(model.generate_from_empty_context())}")


if __name__ == "__main__":
    main()
