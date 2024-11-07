from typing import Optional, List, Union
import migraphx as mgx

import os
import torch
import logging
import sys
import backend
import time
import random

from hip import hip
from PIL import Image
from functools import wraps
from collections import namedtuple
from transformers import CLIPTokenizer, CLIPTextModelWithProjection
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from argparse import ArgumentParser
from StableDiffusionMGX import StableDiffusionMGX


HipEventPair = namedtuple('HipEventPair', ['start', 'end'])

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("backend-pytorch")


formatter = logging.Formatter("{levelname} - {message}", style="{")
file_handler = logging.FileHandler("backend.log", mode="a", encoding="utf-8")
file_handler.setLevel("WARNING")
file_handler.setFormatter(formatter)
log.addHandler(file_handler)

class BackendMIGraphX(backend.Backend):
    def __init__(
        self,
        model_path=None,
        model_id="xl",
        guidance=8,
        steps=20,
        batch_size=1,
        device="cuda",
        precision="fp32",
        negative_prompt="normal quality, low quality, worst quality, low res, blurry, nsfw, nude",
    ):
        super(BackendMIGraphX, self).__init__()
        self.model_path = model_path
        self.pipeline_type = None
        if model_id == "xl":
            self.model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            self.pipeline_type = "sdxl"
        else:
            raise ValueError(f"{model_id} is not a valid model id")

        self.device = device if torch.cuda.is_available() else "cpu"
        self.device_num = int(device[-1]) \
            if (device != "cuda" and device != "cpu") else -1
        
        # log.error(f"[mgx backend] self.device -> {self.device} | device_num -> {self.device_num}")        
        
        if precision == "fp16":
            self.dtype = torch.float16
        elif precision == "bf16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        if torch.cuda.is_available():
            self.local_rank = 0
            self.world_size = 1

        self.guidance = guidance
        self.steps = steps
        self.negative_prompt = negative_prompt
        self.max_length_neg_prompt = 77
        self.batch_size = batch_size
        
        self.mgx = None
        tknz_path1 = os.path.join(self.model_path, "tokenizer")
        tknz_path2 = os.path.join(self.model_path, "tokenizer_2")
        self.tokenizer = CLIPTokenizer.from_pretrained(tknz_path1)
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(tknz_path2)
        self.text_encoder = None
        self.text_encoder_2 = None
        self.unet = None
        self.vae = None

    def version(self):
        return torch.__version__

    def name(self):
        return "pytorch-SUT"

    def image_format(self):
        return "NCHW"

    def load(self):
        if self.model_path is None:
            log.warning(
                "Model path not provided, running with default hugging face weights\n"
                "This may not be valid for official submissions"
            )
            
            raise SystemExit("Provide a valid Model Path to correctly run the program, exiting now...")

        else:
            if self.device_num != -1:
                # log.error(f"Hip set device to -> {self.device_num}")
                hip.hipSetDevice(self.device_num)
            
            # raise SystemExit("Stopping to check")
            
            # Parameter explanations here:
            # onnx_model_path = self.model_path
            # path to compiled .mxr can be left as None
            # Don't want to use refiner model
            use_refiner = False
            # Therefore refiner model path also None
            # refiner compiled model path also None
            
            # set fp16 according to initialization input
            fp16 = "all" if self.dtype == torch.float16 else None
            # Don't want to force .onnx to .mxr compile
            force_compile = False
            # Don't use exhaustive tune when compilling .onnx -> .mxr
            exhaustive_tune = False
            
            tokenizers = {"clip": self.tokenizer, "clip2": self.tokenizer_2}
            
            self.mgx = StableDiffusionMGX(self.pipeline_type, onnx_model_path=self.model_path,
                compiled_model_path=None, use_refiner=use_refiner,
                refiner_onnx_model_path=None,
                refiner_compiled_model_path=None, fp16=fp16,
                force_compile=force_compile, exhaustive_tune=exhaustive_tune, tokenizers=tokenizers)
            
        return self
    
    def predict(self, inputs):
        images = []
        
        # Explanation for mgx.run() arguments        
        # negative_prompt = self.negative_prompt
        # steps = self.steps
        # scale refers to guidance scale -> scale = self.guidance
        # the default SDXLPipeline chooses a random seed everytime, we'll do so manually here
        # not using refiner, so refiner_step = 0
        # not using refiner, so aesthetic_score = 0
        # not using refiner, so negative_aesthetic_score = 0
        # defaults to not verbose
        verbose = False
        #! The main pipeline from loadgen doesn't have text prompt, only tokens
        # log.error(f"[mgx.predict()] inputs -> {inputs} | self.batch_size -> {self.batch_size}")
        
        for i in range(0, len(inputs), self.batch_size):
            if self.batch_size == 1:
                prompt_token = inputs[i]["input_tokens"]
                prompt_token2 = inputs[i]["input_tokens_2"]
                seed = random.randint(0, 2**31 - 1)
                result = self.mgx.run(prompt=None, negative_prompt=self.negative_prompt, steps=self.steps, seed=seed,
                    scale=self.guidance, refiner_steps=0,
                    refiner_aesthetic_score=0,
                    refiner_negative_aesthetic_score=0, verbose=verbose,
                    prompt_tokens=(prompt_token, prompt_token2), device=self.device)
                
                # generated = StableDiffusionMGX.convert_to_rgb_image(result)
                #! COCO needs this to be 3-dimensions
                reshaped = result.reshape(3, 1024, 1024)
                # self.mgx.print_summary(self.steps)
                images.append(reshaped)
                
            else:
                prompt_list = []
                for prompt in inputs[i:min(i+self.batch_size, len(inputs))]:
                    assert isinstance(prompt, dict), "prompt (in inputs) isn't a dict"
                    prompt_token = prompt["input_tokens"]
                    prompt_token2 = prompt["input_tokens_2"]
                    prompt_list.append((prompt_token, prompt_token2))
                    
                
                for prompt in prompt_list:
                    seed = random.randint(0, 2**31 - 1)
                    result = self.mgx.run(prompt=None, negative_prompt=self.negative_prompt, steps=self.steps, seed=seed,
                        scale=self.guidance, refiner_steps=0,
                        refiner_aesthetic_score=0,
                        refiner_negative_aesthetic_score=0, verbose=verbose,
                        prompt_tokens=prompt, device=self.device)

                    # generated = StableDiffusionMGX.convert_to_rgb_image(result)
                    reshaped = result.reshape(3, 1024, 1024)
                    self.mgx.print_summary(self.steps)
                    images.append(reshaped)

        return images