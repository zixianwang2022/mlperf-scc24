# Results of running the MLPerf Benchmark

- CLIP higher is better
- FID lower is better

## Accuracy results: w/ default safetensor models

```json
"accuracy_results": {
        "CLIP_SCORE": 29.99805213883519,
        "FID_SCORE": 137.89080923108685,
        "scenario": "TestScenario.Offline"
    },
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
## w/ migraphx

Accuracy results: (much worse accuracy)

```json
"accuracy_results": {
    "CLIP_SCORE": 17.581279400736094,
    "FID_SCORE": 371.37723025970547,
    "scenario": "TestScenario.Offline"
},

w/ guidance = 5 (still terrible)
"accuracy_results": {
    "CLIP_SCORE": 17.48733188211918,
    "FID_SCORE": 375.5336307825502,
    "scenario": "TestScenario.Offline"
},

w/ exhaustive-tune = True
Fastest Attempt -> Samples per second: 0.900466
```
