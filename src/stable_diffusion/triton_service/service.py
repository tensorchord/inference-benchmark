import torch
import triton_python_backend_utils as triton_utils
from diffusers import StableDiffusionPipeline


class TritonPythonModel:
    def initialize(self, args):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        self.model.to(device)

    def execute(self, requests):
        responses = []
        prompts = []
        for request in requests:
            prompt = triton_utils.get_input_tensor_by_name(request, "PROMPT")
            breakpoint()
            prompts.append(prompt)
    
        images = self.model(prompts).images
        for image in images:
            responses.append(triton_utils.Tensor("IMAGE", image.cpu()))

        return responses
