from http import HTTPStatus
from threading import Thread

import requests

endpoint = "http://127.0.0.1:3000/batch_generate"


def post():
    response = requests.post(
        endpoint,
        headers={"Content-Type": "application/text"},
        data="a happy boy with his toys",
    )
    assert response.status_code == HTTPStatus.OK


for _ in range(10):
    # Test sequential requests.
    post()

for _ in range(10):
    # Test concurrent requests for adaptive batching.
    Thread(target=post).start()
