import numpy as np
from pytriton.client import ModelClient

MODEL_NAME = "distilbert"
MODEL_URL = "127.0.0.1"
TIMEOUT_SECOND = 600


def query(url: str = MODEL_URL, name: str = MODEL_NAME):
    with ModelClient(url, name, init_timeout_s=TIMEOUT_SECOND) as client:
        message = "what a nice day"
        request = np.char.encode(np.asarray([[message]]), "utf-8")
        resp = client.infer_batch(messages=request)
        print(resp)


if __name__ == "__main__":
    query()
