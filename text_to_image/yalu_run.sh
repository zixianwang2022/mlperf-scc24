#! /usr/bin/bash

start_time=$(date +%s)
echo "Yalu test script starting to run"

script_cmd="python3 main.py --dataset "coco-1024" --dataset-path coco2014 --profile stable-diffusion-xl-pytorch --model-path /work1/zixian/youyang1/CM/repos/local/cache/e971d8ea733f4a61/stable_diffusion_fp16 --dtype fp16 --device cuda --time 5 --performance-sample-count 10 --scenario Offline --qps 1"
echo "Running the following cmd: $script_cmd"

# Allocation
# salloc -N 1 -n 4 -p mi2104x -t 01:00:00

# Watch ROCM usage
# squeue -u youyang1
# watch -n 1 rocm-smi --showmemuse

python3 main.py --dataset "coco-1024" --dataset-path coco2014 --profile stable-diffusion-xl-pytorch --model-path /work1/zixian/youyang1/CM/repos/local/cache/e971d8ea733f4a61/stable_diffusion_fp16 --dtype fp16 --device cuda --time 5 --performance-sample-count 10 --scenario Offline --qps 1

# huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 --exclude "*.safetensors" --local-dir $WORK/stable-diffusion-xl-base-1_0-onnx 
# python txt2img.py --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" --seed 42 --output jungle_astro.jpg --pipeline-type sdxl --onnx-model-path /work1/zixian/youyang1/models/sdxl-1.0-base

end_time=$(date +%s)
duration=$((end_time - start_time))

echo "Yalu test script completed in $duration seconds."
