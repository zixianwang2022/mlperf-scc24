#! /usr/bin/bash

start_time=$(date +%s)
echo "Yalu test script starting to run"

# Allocation
# salloc -N 1 -n 4 -p mi2104x -t 01:00:00

# Watch ROCM usage
# squeue -u youyang1
# watch -n 1 rocm-smi --showmemuse

mlperf_pytorch="python3 main.py --dataset "coco-1024" --dataset-path coco2014 --profile stable-diffusion-xl-pytorch --model-path /work1/zixian/youyang1/CM/repos/local/cache/e971d8ea733f4a61/stable_diffusion_fp16 --dtype fp16 --device cuda --time 5 --performance-sample-count 10 --scenario Offline --qps 1"
mlperf_mgx="python3 main.py --dataset "coco-1024" --dataset-path coco2014 --profile stable-diffusion-xl-mgx --model-path /work1/zixian/youyang1/models/sdxl-1.0-base --dtype fp16 --device cuda --time 5 --performance-sample-count 10 --scenario Offline --qps 1"
mgx_cmd="python StableDiffusionMGX.py --seed 42 --pipeline-type sdxl --onnx-model-path /work1/zixian/youyang1/models/sdxl-1.0-base --fp16=all"

if [ "$1" == "pytorch" ]; then
    echo "Running [mlperf_pytorch] cmd: $mlperf_pytorch"
    eval $mlperf_pytorch
elif [ "$1" == "mlperf_mgx" ]; then
    echo "Running [mlperf_mgx] cmd: $mlperf_mgx"
    eval $mlperf_mgx
elif [ "$1" == "mgx" ]; then
    echo "Running [mgx] cmd: $mgx_cmd"
    eval $mgx_cmd
else
    # runs mgx by default
    echo "Running [mlperf_mgx] cmd: $mlperf_mgx"
    eval $mlperf_mgx
fi

# huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 --exclude "*.safetensors" --local-dir $WORK/stable-diffusion-xl-base-1_0-onnx 
# huggingface-cli upload SeaSponge/scc24_mlperf_mgx_exhaustive unet/model_fp16_gpu.mxr unet_nope/model_fp16_gpu.mxr
# export PYTHONPATH=/work1/zixian/youyang1/AMDMIGraphX/build/lib:$PYTHONPATH
# export MGX_SAMPLE_PATH=/work1/zixian/youyang1/AMDMIGraphX/examples/diffusion/python_stable_diffusion_xl
# python txt2img.py --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" --seed 42 --output jungle_astro.jpg --pipeline-type sdxl --onnx-model-path /work1/zixian/youyang1/models/sdxl-1.0-base --fp16=all
# python int8.py --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" --seed 42 --output jungle_astro.jpg --pipeline-type sdxl --onnx-model-path /work1/zixian/youyang1/models/sdxl-1.0-base --fp16=all

end_time=$(date +%s)
duration=$((end_time - start_time))

echo "[$(date)] Yalu test script completed in $duration seconds."
echo "[$(date)] Yalu test script completed in $duration seconds." >> yalu_run_record.txt
