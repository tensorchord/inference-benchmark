import numpy as np
from pytriton.client import ModelClient

MODEL_NAME = "stable_diffusion"
MODEL_URL = "127.0.0.1"
TIMEOUT_SECOND = 600


def query(url: str = MODEL_URL, name: str = MODEL_NAME):
    with ModelClient(url, name, init_timeout_s=TIMEOUT_SECOND) as client:
        prompt = "A photo of a cat"
        prompt_req = np.char.encode(np.asarray([[prompt]]), "utf-8")
        client.infer_batch(prompt=prompt_req)


if __name__ == "__main__":
    query()
