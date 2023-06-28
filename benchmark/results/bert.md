## Environment

GCP VM with:
- Machine type: n1-standard-8
- CPU: 8 vCPU Intel Haswell
- RAM: 32G
- GPU: 1 x NVIDIA T4
- GPU Driver Version: 510.47.03
- CUDA Version: 11.6

Python:
- MiniConda env with Python 3.8.17 (`pytriton` requires py38)
- torch==2.0.1
- transformers==4.30.2
- nvidia-pytriton==0.2.0
- mosec==0.7.2

## Results

All the results are collected after the **warmup** load test.

Dynamic batching:
- max batch size: 32
- max wait time: 10ms

### pytriton

`hey -m POST -n 50000 -H Inference-Header-Content-Length:175 -D bert.bin http://127.0.0.1:8000/v2/models/distilbert/infer`

Usage:
- GPU Memory: 1117MiB / 15360MiB 
- GPU Util: 39%

```
Summary:
  Total:	55.0552 secs
  Slowest:	0.0911 secs
  Fastest:	0.0110 secs
  Average:	0.0549 secs
  Requests/sec:	908.1787

Response time histogram:
  0.011 [1]	    |
  0.019 [28]	|
  0.027 [18]	|
  0.035 [68]	|
  0.043 [207]	|
  0.051 [31239]	|■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
  0.059 [3274]	|■■■■
  0.067 [6956]	|■■■■■■■■■
  0.075 [2954]	|■■■■
  0.083 [5155]	|■■■■■■■
  0.091 [100]	|

Latency distribution:
  10% in 0.0463 secs
  25% in 0.0476 secs
  50% in 0.0494 secs
  75% in 0.0626 secs
  90% in 0.0752 secs
  95% in 0.0770 secs
  99% in 0.0792 secs

Status code distribution:
  [200]	50000 responses
```

### mosec

```sh
hey -c 50 -m POST -n 50000 -d "The quick brown fox jumps over the lazy dog" http://127.0.0.1:8000/inference
```

Usage:
- GPU Memory: 1043MiB / 15360MiB
- GPU Util: 53%

```
Summary:
  Total:	23.2878 secs
  Slowest:	0.0778 secs
  Fastest:	0.0119 secs
  Average:	0.0230 secs
  Requests/sec:	2147.0514
  
Response time histogram:
  0.012 [1]	|
  0.018 [21773]	|■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
  0.025 [363]	|■
  0.032 [26762]	|■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
  0.038 [902]	|■
  0.045 [125]	|
  0.051 [56]	|
  0.058 [17]	|
  0.065 [0]	|
  0.071 [0]	|
  0.078 [1]	|

Latency distribution:
  10% in 0.0142 secs
  25% in 0.0149 secs
  50% in 0.0278 secs
  75% in 0.0293 secs
  90% in 0.0303 secs
  95% in 0.0310 secs
  99% in 0.0324 secs

Status code distribution:
  [200]	50000 responses
```

The response time is not even due to the request is not enough. We can modify the concurrent workers from `50` to `80`:

```sh
hey -c 80 -m POST -n 50000 -d "The quick brown fox jumps over the lazy dog" http://127.0.0.1:8000/inference
```

Usage:
- GPU Memory: 1043MiB / 15360MiB
- GPU Util: 78%

```
Summary:
  Total:	18.0979 secs
  Slowest:	0.0720 secs
  Fastest:	0.0127 secs
  Average:	0.0286 secs
  Requests/sec:	2762.7568
  
  Total data:	400000 bytes
  Size/request:	8 bytes

Response time histogram:
  0.013 [1]	    |
  0.019 [88]	|
  0.025 [24420]	|■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
  0.030 [419]	|■
  0.036 [24618]	|■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
  0.042 [153]	|
  0.048 [26]	|
  0.054 [222]	|
  0.060 [37]	|
  0.066 [0]	    |
  0.072 [16]	|

Latency distribution:
  10% in 0.0224 secs
  25% in 0.0229 secs
  50% in 0.0315 secs
  75% in 0.0340 secs
  90% in 0.0347 secs
  95% in 0.0351 secs
  99% in 0.0363 secs

Status code distribution:
  [200]	50000 responses
```

If we change the mosec inference number to 2, it will be even faster:

```sh
hey -c 80 -m POST -n 50000 -d "The quick brown fox jumps over the lazy dog" http://127.0.0.1:8000/inference
```

- GPU Memory: 2080MiB / 15360MiB
- GPU Util: 99%

```
Summary:
  Total:	16.5202 secs
  Slowest:	0.1151 secs
  Fastest:	0.0135 secs
  Average:	0.0259 secs
  Requests/sec:	3026.6061

Response time histogram:
  0.013 [1]	    |
  0.024 [24159]	|■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
  0.034 [22667]	|■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
  0.044 [3037]	|■■■■■
  0.054 [59]	|
  0.064 [32]	|
  0.074 [11]	|
  0.085 [20]	|
  0.095 [8]	    |
  0.105 [0]	    |
  0.115 [6]	    |

Latency distribution:
  10% in 0.0201 secs
  25% in 0.0207 secs
  50% in 0.0253 secs
  75% in 0.0305 secs
  90% in 0.0332 secs
  95% in 0.0342 secs
  99% in 0.0382 secs

Status code distribution:
  [200]	50000 responses
```
