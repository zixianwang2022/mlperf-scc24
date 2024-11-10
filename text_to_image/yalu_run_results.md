# Results of running the MLPerf Benchmark

- CLIP higher is better
- FID lower is better

## Accuracy results:

### default safetensor models

```json
"accuracy_results": {
        "CLIP_SCORE": 29.99805213883519,
        "FID_SCORE": 137.89080923108685,
        "scenario": "TestScenario.Offline"
    },

Fastest attempt:
- batch = 1
- total sample = 32

Samples per second: 0.736967
```

## w/ quantized unet and vae

```json
"accuracy_results": {
    "CLIP_SCORE": 30.923738230019808,
    "FID_SCORE": 137.92278513232623,
    "scenario": "TestScenario.Offline"
},

Fastest attempt: Samples per second -> 0.667284
```

### (MIGraphX) quantized unet

1. guidance = 5
2. w/ exhaustive-tune = True

(still terrible, but the images actually look fine)

```json
"accuracy_results": {
    "CLIP_SCORE": 26.513378769159317,
    "FID_SCORE": 354.6436180397586,
    "scenario": "TestScenario.Offline"
},
Fastest Attempt:
- batch = 1

Samples per second: 0.900466
```

### (MIGraphX) default unet.onnx ()

1. **mxr** file 8.97 GB
2. guidance = 5
3. `exhaustive-tune = True`

(The images look pretty good, and speed is not much slower than above)

```json
"accuracy_results": {
    "CLIP_SCORE": 30.85044001042843,
    "FID_SCORE": 145.17722123924995,
    "scenario": "TestScenario.Offline"
},

Fastest Attempt:
- batch = 1
- total sample = 32

Samples per second: 0.893562
```


## Trying to identify issues with MGX

1. The input **tokens** and **input_tokens_2** are the same, same shape and values.

2. Trying to see if the embeddings are different (embeds are of type `<class 'torch.Tensor'>`)