import torch
import triton_python_backend_utils as triton_utils
from diffusers import StableDiffusionPipeline


class TritonPythonModel:
    def initialize(self, args):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.model.to(self.device)

    def execute(self, requests):
        responses = []
        prompts = []
        for request in requests:
            prompt = (
                triton_utils.get_input_tensor_by_name(request, "PROMPT")
                .as_numpy()
                .tolist()
            )
            prompts.append(prompt[0].decode())

        images = self.model(prompts).images
        for image in images:
            output = triton_utils.Tensor(
                "IMAGE", image.cpu().numpy() if self.device == "cuda" else image.numpy()
            )
            resp = triton_utils.InferenceResponse(output_tensors=[output])
            responses.append(resp)

        return responses
