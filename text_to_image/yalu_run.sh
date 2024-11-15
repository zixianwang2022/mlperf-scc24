#! /usr/bin/bash

# set -x  # Enable debugging

start_time=$(date +%s)
echo "Yalu test script starting to run"

# Allocation
# salloc -N 1 -n 4 -p mi2104x -t 01:00:00

# Watch ROCM usage
# squeue -u youyang1
# watch -n 1 rocm-smi --showmemuse

mlperf_pytorch="python3 main.py --dataset "coco-1024" --dataset-path coco2014 --profile stable-diffusion-xl-pytorch --model-path /work1/zixian/youyang1/CM/repos/local/cache/e971d8ea733f4a61/stable_diffusion_fp16 --dtype fp16 --device cuda --time 5 --performance-sample-count 10 --scenario Offline --qps 1"

# Old cmd: python3 main.py --dataset "coco-1024" --dataset-path coco2014 --profile stable-diffusion-xl-mgx --model-path /work1/zixian/youyang1/models/sdxl-1.0-base --dtype fp16 --device cuda --time 5 --performance-sample-count 10 --scenario Offline --qps 1
mlperf_mgx="python3 main.py"

mgx_cmd="python StableDiffusionMGX.py --seed 42 --pipeline-type sdxl --onnx-model-path /work1/zixian/youyang1/models/sdxl-1.0-base --fp16=all"

if [ "$1" == "pytorch" ]; then
    echo "Running [mlperf_pytorch] cmd: $mlperf_pytorch"
    eval $mlperf_pytorch
elif [ "$1" == "mgx_path" ]; then
    export PYTHONPATH=/work1/zixian/youyang1/AMDMIGraphX/build/lib:$PYTHONPATH
elif [ "$1" == "mlperf_mgx" ]; then
    echo "Running [mlperf_mgx] cmd: $mlperf_mgx"
    eval $mlperf_mgx
elif [ "$1" == "mgx" ]; then
    echo "Running [mgx] cmd: $mgx_cmd"
    eval $mgx_cmd
elif [ "$1" == "cm" ]; then
    if [ "$2" == "clean" ]; then
        cm rm cache --tags=inference,src -f
        cm rm cache --tags=inference -f
        cm rm cache --tags=python -f
        cm pull repo
    fi
    # cm run script --tags=run-mlperf,inference,_r4.1-dev,_short,_scc24-base \
    # --model=sdxl \
    # --framework=pytorch \
    # --category=datacenter \
    # --scenario=Offline \
    # --execution_mode=test \
    # --device=rocm \
    # --quiet --precision=float16 \
    # --adr.mlperf-implementation.tags=_branch.yalu,_repo.https://github.com/zixianwang2022/mlperf-scc24 --adr.mlperf-implementation.version=custom  --env.CM_GET_PLATFORM_DETAILS=no
    cm run script --tags=run-mlperf,inference,_r4.1-dev,_scc24-main \
        --model=sdxl \
        --framework=pytorch \
        --category=datacenter \
        --scenario=Offline \
        --execution_mode=test \
        --device=rocm \
        --quiet --precision=float16 \
        --adr.mlperf-implementation.tags=_branch.yalu,_repo.https://github.com/zixianwang2022/mlperf-scc24 --adr.mlperf-implementation.version=custom  --env.CM_GET_PLATFORM_DETAILS=no
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