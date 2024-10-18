#! /usr/bin/bash

start_time=$(date +%s)
echo "Yalu test script starting to run"

script_cmd="python3 main.py --dataset "coco-1024" --dataset-path coco2014 --profile stable-diffusion-xl-pytorch --model-path /work1/zixian/youyang1/CM/repos/local/cache/2c71269e69d84ba8/stable_diffusion_fp16 --dtype fp16 --device cuda --time 5 --performance-sample-count 10 --scenario Offline --qps 1"
echo "Running the following cmd: $script_cmd"

# Allocation
# salloc -N 1 -n 4 -p mi2104x -t 01:00:00

# Watch ROCM usage
# squeue -u youyang1
# watch -n 1 rocm-smi --showmemuse

python3 main.py --dataset "coco-1024" --dataset-path coco2014 --profile stable-diffusion-xl-pytorch --model-path /work1/zixian/youyang1/CM/repos/local/cache/2c71269e69d84ba8/stable_diffusion_fp16 --dtype fp16 --device cuda --time 5 --performance-sample-count 10 --scenario Offline --qps 1

end_time=$(date +%s)
duration=$((end_time - start_time))

echo "Yalu test script completed in $duration seconds."
