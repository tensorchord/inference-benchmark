"""MLServer service for stable diffusion.

This implementation is based on the official documentation:

- https://mlserver.readthedocs.io/en/latest/examples/custom/README.html
"""

import base64
from io import BytesIO
from typing import List

import torch  # type: ignore
from diffusers import StableDiffusionPipeline  # type: ignore
from mlserver import MLModel
from mlserver.codecs import decode_args


class StableDiffusion(MLModel):
    async def load(self) -> bool:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        return True

    @decode_args
    async def predict(self, prompts: List[str]) -> List[str]:
        images_b64 = []
        for pil_im in self._model(prompts).images:
            buf = BytesIO()
            pil_im.save(buf, format="JPEG")
            images_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        return images_b64
