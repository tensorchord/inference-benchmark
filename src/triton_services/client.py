import numpy as np
import tritonclient.http as httpclient

with httpclient.InferenceServerClient("localhost:8000") as client:
    prompt = httpclient.InferInput("PROMPT", shape=(1,), datatype="BYTES")
    prompt.set_data_from_numpy(np.asarray(["cat"], dtype=object))
    images = httpclient.InferRequestedOutput("IMAGE", binary_data=False)
    response = client.infer(
        model_name="stable_diffusion",
        inputs=[prompt],
        outputs=[images],
    )
    content = response.as_numpy("IMAGE")
