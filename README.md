<div align="center">

# Inference Benchmark

Maximize the potential of your models with the inference benchmark (tool).

</div>

<p align=center>
<a href="https://discord.gg/KqswhpVgdU"><img alt="discord invitation link" src="https://dcbadge.vercel.app/api/server/KqswhpVgdU?style=flat"></a>
<a href="https://twitter.com/TensorChord"><img src="https://img.shields.io/twitter/follow/tensorchord?style=social" alt="trackgit-views" /></a>
</p>

# What is it

Inference benchmark provides a standard way to measure the performance of inference workloads. It is also a tool that allows you to evaluate and optimize the performance of your inference workloads.

# Results

## Bert

We benchmarked [pytriton (triton-inference-server)](https://github.com/triton-inference-server/pytriton) and [mosec](https://github.com/mosecorg/mosec) with bert. We enabled dynamic batching for both frameworks with max batch size 32 and max wait time 10ms. Please checkout the [result](./benchmark/results/bert.md) for more details.

![DistilBert](./benchmark/results/distilbert_serving_benchmark.png)

More [results with different models on different serving frameworks](https://github.com/tensorchord/inference-benchmark/issues/7) are coming soon.
