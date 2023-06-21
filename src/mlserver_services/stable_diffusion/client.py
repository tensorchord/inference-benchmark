from threading import Thread

import requests
from mlserver.codecs import StringCodec
from mlserver.types import InferenceRequest

endpoint = "http://127.0.0.1:8080/v2/models/stable-diffusion/infer"

inference_request_dict = InferenceRequest(
    inputs=[
        StringCodec.encode_input(
            name="prompts", payload=["a happy boy with his toys"], use_bytes=False
        )
    ]
).dict()


def post():
    response = requests.post(endpoint, json=inference_request_dict)
    print(response.json())


for _ in range(10):
    # Test adaptive batching.
    Thread(target=post).start()
