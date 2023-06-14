import io

import numpy as np
import soundfile
import torch
from mosec import Server, Worker
from transformers import WhisperForConditionalGeneration, WhisperProcessor

STEREO_CHANNEL_NUM = 2


class Preprocess(Worker):
    def __init__(self):
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-base")

    def deserialize(self, data: bytes) -> any:
        with io.BytesIO(data) as byte_io:
            array, sampling_rate = soundfile.read(byte_io)
        if array.shape[1] == STEREO_CHANNEL_NUM:
            # conbime the channel
            array = np.mean(array, 1)
        return {"array": array, "sampling_rate": sampling_rate}

    def forward(self, data):
        res = self.processor(
            data["array"], sampling_rate=data["sampling_rate"], return_tensors="pt"
        )
        return res.input_features


class Inference(Worker):
    def __init__(self):
        self.model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-base"
        )
        self.model.config.forced_decoder_ids = None
        self.device = (
            torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def forward(self, data):
        ids = self.model.generate(torch.cat(data).to(self.device))
        return ids.cpu().tolist()


class Postprocess(Worker):
    def __init__(self):
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-base")

    def forward(self, data):
        return self.processor.batch_decode(data, skip_special_tokens=True)

    def serialize(self, data: str) -> bytes:
        return data.encode("utf-8")


if __name__ == "__main__":
    server = Server()
    server.append_worker(Preprocess, num=2)
    server.append_worker(Inference, max_batch_size=16, max_wait_time=10)
    server.append_worker(Postprocess, num=2, max_batch_size=8, max_wait_time=5)
    server.run()
