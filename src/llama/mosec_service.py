import torch
from mosec import Server, Worker
from transformers import LlamaForCausalLM, LlamaTokenizer

MODEL = "decapoda-research/llama-7b-hf"
MAX_LENGTH = 50


class TokenEncoder(Worker):
    def __init__(self):
        self.tokenizer = LlamaTokenizer.from_pretrained(MODEL)

    def deserialize(self, data):
        return data.decode()

    def forward(self, data):
        tokens = self.tokenizer(data)
        print(tokens)
        return tokens.input_ids


class Inference(Worker):
    def __init__(self):
        self.model = LlamaForCausalLM.from_pretrained(MODEL)
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(self.device)
        self.model.eval()

    def forward(self, data):
        inputs = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(tokens) for tokens in data], batch_first=True
        ).to(self.device)
        outputs = self.model.generate(inputs, max_length=MAX_LENGTH).tolist()
        return outputs


class TokenDecoder(Worker):
    def __init__(self):
        self.tokenizer = LlamaTokenizer.from_pretrained(MODEL)

    def forward(self, data):
        outputs = self.tokenizer.decode(data, skip_special_tokens=True)
        return outputs


if __name__ == "__main__":
    server = Server()
    server.append_worker(TokenEncoder)
    server.append_worker(Inference, max_batch_size=4, timeout=30)
    server.append_worker(TokenDecoder)
    server.run()
