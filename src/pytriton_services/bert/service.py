import numpy as np
import torch
from pytriton.decorators import batch
from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig
from transformers import (  # type: ignore
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
)

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_INSTANCE = 1
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()


@batch
def infer(messages: np.ndarray):
    msg = [
        np.char.decode(message.astype("bytes"), "utf-8").item() for message in messages
    ]
    inputs = tokenizer(msg, return_tensors="pt", padding=True)
    inputs.to(DEVICE)
    logits = model(**inputs).logits
    label_ids = logits.argmax(dim=1).cpu().tolist()
    return {"labels": np.asarray([model.config.id2label[i] for i in label_ids])}


def main():
    config = TritonConfig(exit_on_error=True)
    with Triton(config=config) as triton:
        triton.bind(
            model_name="distilbert",
            infer_func=[infer] * NUM_INSTANCE,
            inputs=[Tensor(name="messages", dtype=np.bytes_, shape=(1,))],
            outputs=[Tensor(name="labels", dtype=np.bytes_, shape=(1,))],
            config=ModelConfig(
                max_batch_size=32,
                batcher=DynamicBatcher(max_queue_delay_microseconds=10),
            ),
        )
        triton.serve()


if __name__ == "__main__":
    main()
