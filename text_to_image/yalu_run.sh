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

# Needed for mgx
# export PYTHONPATH=/work1/zixian/youyang1/AMDMIGraphX/build/lib:$PYTHONPATH

mgx_multi="True"

if [ "$1" == "pytorch" ]; then
    echo "Running [mlperf_pytorch] cmd: $mlperf_pytorch"
    eval $mlperf_pytorch
elif [ "$1" == "mgx" ]; then
    echo "Running [mlperf_mgx] cmd: $mlperf_mgx"
    eval $mlperf_mgx
elif [ "$1" == "cm" ]; then
    for arg in "$@"; do
        if [ $arg == "clean" ]; then 
            cm rm cache --tags=inference,src -f
            cm rm cache --tags=inference -f
            cm rm cache --tags=python -f
            cm pull repo
        fi

        if [ $arg == "multi_mgx" ]; then
            mgx_multi="True"
            echo "Running mgx multinode"
        fi
    done
    # PyTorch & Multi-node Implementation
    if [ $mgx_multi == "False" ]; then
        cm run script --tags=run-mlperf,inference,_r4.1-dev,_scc24-main \
            --model=sdxl \
            --framework=pytorch \
            --category=datacenter \
            --scenario=Offline \
            --execution_mode=test \
            --device=rocm \
            --quiet --precision=float16 \
            --adr.mlperf-implementation.tags=_branch.multinode,_repo.https://github.com/zixianwang2022/mlperf-scc24 --adr.mlperf-implementation.version=custom  --env.CM_GET_PLATFORM_DETAILS=no
    else
        echo "mgx_multi is True"
    fi
    # cm run script --tags=run-mlperf,inference,_r4.1-dev,_scc24-main \
    #     --model=sdxl \
    #     --framework=pytorch \
    #     --category=datacenter \
    #     --scenario=Offline \
    #     --execution_mode=test \
    #     --device=rocm \
    #     --quiet --precision=float16 \
    #     --adr.mlperf-implementation.tags=_branch.yalu,_repo.https://github.com/zixianwang2022/mlperf-scc24 --adr.mlperf-implementation.version=custom  --env.CM_GET_PLATFORM_DETAILS=no
else
    # runs cm by default
    echo "Running CM multinode..."
    for arg in "$@"; do
        if [ $arg == "clean" ]; then 
            cm rm cache --tags=inference,src -f
            cm rm cache --tags=inference -f
            cm rm cache --tags=python -f
            cm pull repo
        fi

        if [ $arg == "multi_mgx" ]; then
            mgx_multi="True"
            echo "Running mgx multinode"
        fi
    done
    # PyTorch & Multi-node Implementation
    if [ $mgx_multi == "False" ]; then
        cm run script --tags=run-mlperf,inference,_r4.1-dev,_scc24-main \
            --model=sdxl \
            --framework=pytorch \
            --category=datacenter \
            --scenario=Offline \
            --execution_mode=test \
            --device=rocm \
            --quiet --precision=float16 \
            --adr.mlperf-implementation.tags=_branch.multinode,_repo.https://github.com/zixianwang2022/mlperf-scc24 --adr.mlperf-implementation.version=custom  --env.CM_GET_PLATFORM_DETAILS=no
    else
        echo "mgx_multi is True"
    fi
fi


end_time=$(date +%s)
duration=$((end_time - start_time))

echo "[$(date)] Yalu test script completed in $duration seconds."
echo "[$(date)] Yalu test script completed in $duration seconds." >> yalu_run_record.txt


# Multinode from source
# Run these from different windows
# server_sut:
# python sut_over_network_demo.py --dataset "coco-1024" --dataset-path coco2014 --profile stable-diffusion-xl-pytorch --dtype fp16 --device cuda --scenario Offline --max-batchsize 4

# network_server:
# python main.py  --dataset=coco-1024 --dataset-path=/work1/zixian/ziw081/inference/text_to_image/coco2014 --profile=stable-diffusion-xl-pytorch --dtype=fp16 --device=cuda --time=30 --scenario=Offline 