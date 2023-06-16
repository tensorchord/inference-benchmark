"""Potassium service for stable diffusion.

This implementation is based on the official README example:

- https://github.com/bananaml/potassium#or-do-it-yourself
"""

import base64
from io import BytesIO

import torch
from diffusers import StableDiffusionPipeline
from potassium import Potassium, Request, Response

app = Potassium(__name__)


@app.init
def init():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.to(device)

    context = {
        "model": model,
    }
    return context


@app.handler()
def handler(context: dict, request: Request) -> Response:
    model = context["model"]
    prompt = request.json.get("prompt")
    image = model(prompt).images[0]
    buf = BytesIO()
    image.save(buf, format="JPEG")
    return Response(
        json={"image_b64": base64.b64encode(buf.getvalue()).decode("utf-8")},
        status=200,
    )


if __name__ == "__main__":
    app.serve()
