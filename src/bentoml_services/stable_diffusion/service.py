"""BentoML service for stable diffusion.

This implementation is based on the official examples:

- https://github.com/bentoml/BentoML/tree/main/examples/custom_runner
"""
from time import time

import bentoml
import torch
from bentoml.io import Image, Text
from diffusers import StableDiffusionPipeline  # type: ignore

st_time = time()


class StableDiffusionRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        init_st_time = time()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        print(f"model loading time = {time() - init_st_time}")
        print(f"total starting time = {time() - st_time}")

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def inference(self, prompts):
        print(prompts)
        pil_images = self._model(prompts).images
        return pil_images


stable_diffusion_runner = bentoml.Runner(StableDiffusionRunnable, max_batch_size=8)

svc = bentoml.Service("stable_diffusion", runners=[stable_diffusion_runner])


"""
At the time this benchmark was created (27 June 2023), the bentoml batching
seems to have some bugs:
* The async call (`\generate`) failes to batch
* The sync call (`\batch_generate`) allows batch but may lead to the following
error: bentoml.exceptions.ServiceUnavailable: Service Busy
"""


@svc.api(input=Text(), output=Image())
async def generate(input_txt):
    batch_ret = await stable_diffusion_runner.inference.async_run([input_txt])
    return batch_ret[0]


@svc.api(input=Text(), output=Image())
def batch_generate(input_txt):
    batch_ret = stable_diffusion_runner.inference.run([input_txt])
    return batch_ret[0]


# curl -X POST -H "content-type: application/text" --data "a happy boy with his toys" http://127.0.0.1:3000/generate --output result.jpeg
