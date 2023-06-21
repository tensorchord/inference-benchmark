import tritonclient.http as httpclient


with httpclient.InferenceServerClient("localhost:8000") as client:
    inputs = [
        httpclient.InferInput("PROMPT", shape=(1,), datatype="BYTES"),
    ]
    images = httpclient.InferRequestedOutput("IMAGE", binary_data=True)
    client.infer(model_name="stable_diffusion", inputs=inputs, outputs=[images])
