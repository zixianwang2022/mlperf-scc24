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
        self.self.pipeline_type = None
        if model_id == "xl":
            self.model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            self.self.pipeline_type = "sdxl"
        else:
            raise ValueError(f"{model_id} is not a valid model id")

        self.device = device if torch.cuda.is_available() else "cpu"
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
        self.tokenizer = None
        self.tokenizer_2 = None
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
            self.mgx = StableDiffusionMGX(self.pipeline_type, onnx_model_path=self.model_path,
                compiled_model_path=None, use_refiner=use_refiner,
                refiner_onnx_model_path=None,
                refiner_compiled_model_path=None, fp16=fp16,
                force_compile=force_compile, exhaustive_tune=exhaustive_tune)
            
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
        for i in range(0, len(inputs), self.batch_size):
            if self.batch_size == 1:
                prompt = inputs[i]["input_tokens"]
                seed = random.randint(0, 2**31 - 1)
                result = self.mgx.run(prompt=prompt, negative_prompt=self.negative_prompt, steps=self.steps, seed=seed,
                    scale=self.guidance, refiner_steps=0,
                    refiner_aesthetic_score=0,
                    refiner_negative_aesthetic_score=0, verbose=verbose)
                
                generated = StableDiffusionMGX.convert_to_rgb_image(result)
                images.extend(generated)
                
            else:
                prompt_list = []
                for prompt in inputs[i:min(i+self.batch_size, len(inputs))]:
                    assert isinstance(prompt, dict), "prompt (in inputs) isn't a dict"
                    text_input = prompt["input_tokens"]
                    prompt_list.append(text_input)
                    
                
                for prompt in prompt_list:
                    seed = random.randint(0, 2**31 - 1)
                    result = self.mgx.run(prompt=prompt, negative_prompt=self.negative_prompt, steps=self.steps, seed=seed,
                        scale=self.guidance, refiner_steps=0,
                        refiner_aesthetic_score=0,
                        refiner_negative_aesthetic_score=0, verbose=verbose)

                    generated = StableDiffusionMGX.convert_to_rgb_image(result)
                    images.extend(generated)

        return images