Make sure you have set the Conda environment and have it activated


Install CM:
python3 -m pip install cmind

	
1. Change the environment path (you can customize the path or add this line to .bashrc):
export CM_REPOS=$WORK/CM 


2. Run the command to test if CM has successfully installed and check the repo path:
cm test core


3. Pull the cm script repo:	 
cm pull repo mlcommons@cm4mlops




Build mlperf:
1. Install the environment
cd $WORK
git clone https://github.com/zixianwang2022/mlperf-scc24.git inference 

export ROOT=$WORK/inference
export SD_FOLDER=$WORK/inference/text_to_image
export LOADGEN_FOLDER=$WORK/inference/loadgen
export MODEL_PATH=$WORK/inference/text_to_image/model/

cp $ROOT/mlperf.conf $SD_FOLDER

# install requirements
cd $SD_FOLDER
pip install -r requirements.txt

cd $LOADGEN_FOLDER
CFLAGS="-std=c++14" python setup.py install

# install torch for rocm 
<!-- pip uninstall torch
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/rocm5.2/ -->

<!-- # The above wasn’t working for some people, so this is an alternative -->
# pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.2/

# Download the model
cm run script --tags=get,ml-model,sdxl,_fp16,_rclone -j

# make sure you have installed all the dependencies before running the following commands
# you can find the model downloaded in the CM folder, find it and replace <model path> in below.

cd $SD_FOLDER

python3 main.py --dataset "coco-1024" --dataset-path coco2014 --profile stable-diffusion-xl-pytorch --model-path <model path> --dtype fp16 --device cuda --time 30 --scenario Offline





python3 main.py --dataset "coco-1024" --dataset-path coco2014 --profile stable-diffusion-xl-pytorch --model-path /work1/zixian/ziw081/CM/models/SDXL/official_pytorch/fp16/stable_diffusion_fp16/ --dtype fp16 --device cuda --time 30 --scenario Offline  --max-batchsize 1 

