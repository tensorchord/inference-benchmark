import tritonclient.http as httpclient
import numpy as np


with httpclient.InferenceServerClient("localhost:8000") as client:
    prompt = httpclient.InferInput("PROMPT", shape=(1,), datatype="BYTES"),
    prompt.set_data_from_numpy(np.asarray(["cat"]))
    images = httpclient.InferRequestedOutput("IMAGE", binary_data=False)
    client.infer(model_name="stable_diffusion", inputs=[prompt], outputs=[images])
