from typing import Optional, List, Union
import migraphx as mgx

import os
import torch
import logging
import sys
import backend
import time

from hip import hip
from PIL import Image
from functools import wraps
from collections import namedtuple
from transformers import CLIPTokenizer, CLIPTextModelWithProjection
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from argparse import ArgumentParser


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
        if model_id == "xl":
            self.model_id = "stabilityai/stable-diffusion-xl-base-1.0"
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
            
            self.scheduler = EulerDiscreteScheduler.from_pretrained(
                self.model_id, subfolder="scheduler"
            )
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                self.model_id,
                scheduler=self.scheduler,
                safety_checker=None,
                add_watermarker=False,
                variant="fp16" if (self.dtype == torch.float16) else None,
                torch_dtype=self.dtype,
                use_safetensors=False
            )

        else:
            self.scheduler = EulerDiscreteScheduler.from_pretrained(
                os.path.join(self.model_path, "checkpoint_scheduler"),
                subfolder="scheduler",
            )
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                os.path.join(self.model_path, "checkpoint_pipe"),
                scheduler=self.scheduler,
                safety_checker=None,
                add_watermarker=False,
                variant="fp16" if (self.dtype == torch.float16) else None,
                torch_dtype=self.dtype,
                use_safetensors=False
            )
        
        self.pipe.to(self.device)
                
        #self.pipe.set_progress_bar_config(disable=True)

        self.negative_prompt_tokens = self.pipe.tokenizer(
            self.convert_prompt(self.negative_prompt, self.pipe.tokenizer),
            padding="max_length",
            max_length=self.max_length_neg_prompt,
            truncation=True,
            return_tensors="pt",
        )
        self.negative_prompt_tokens_2 = self.pipe.tokenizer_2(
            self.convert_prompt(self.negative_prompt, self.pipe.tokenizer_2),
            padding="max_length",
            max_length=self.max_length_neg_prompt,
            truncation=True,
            return_tensors="pt",
        )
        return self

    def convert_prompt(self, prompt, tokenizer):
        tokens = tokenizer.tokenize(prompt)
        unique_tokens = set(tokens)
        for token in unique_tokens:
            if token in tokenizer.added_tokens_encoder:
                replacement = token
                i = 1
                while f"{token}_{i}" in tokenizer.added_tokens_encoder:
                    replacement += f" {token}_{i}"
                    i += 1

                prompt = prompt.replace(token, replacement)

        return prompt

    def encode_tokens(
        self,
        pipe: StableDiffusionXLPipeline,
        text_input: torch.Tensor,
        text_input_2: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[torch.Tensor] = None,
        negative_prompt_2: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the input tokens into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        device = device or pipe._execution_device
        batch_size = text_input.input_ids.shape[0]

        # Define tokenizers and text encoders
        tokenizers = (
            [pipe.tokenizer, pipe.tokenizer_2]
            if pipe.tokenizer is not None
            else [pipe.tokenizer_2]
        )
        text_encoders = (
            [pipe.text_encoder, pipe.text_encoder_2]
            if pipe.text_encoder is not None
            else [pipe.text_encoder_2]
        )

        if prompt_embeds is None:
            text_input_2 = text_input_2 or text_input

            # textual inversion: procecss multi-vector tokens if necessary
            prompt_embeds_list = []
            text_inputs_list = [text_input, text_input_2]
            for text_inputs, tokenizer, text_encoder in zip(
                text_inputs_list, tokenizers, text_encoders
            ):
                text_input_ids = text_inputs.input_ids
                prompt_embeds = text_encoder(
                    text_input_ids.to(device), output_hidden_states=True
                )

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                if clip_skip is None:
                    prompt_embeds = prompt_embeds.hidden_states[-2]
                else:
                    # "2" because SDXL always indexes from the penultimate layer.
                    prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = (
            negative_prompt is None and pipe.config.force_zeros_for_empty_prompt
        )
        if (
            do_classifier_free_guidance
            and negative_prompt_embeds is None
            and zero_out_negative_prompt
        ):
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            # normalize str to list
            negative_prompt_inputs = (
                negative_prompt.input_ids.repeat(batch_size, 1)
                if (len(negative_prompt.input_ids.shape) == 1)
                else negative_prompt.input_ids
            )
            negative_prompt_2_inputs = (
                negative_prompt_2.input_ids.repeat(batch_size, 1)
                if (len(negative_prompt_2.input_ids.shape) == 1)
                else negative_prompt_2.input_ids
            )

            uncond_inputs = [negative_prompt_inputs, negative_prompt_2_inputs]

            negative_prompt_embeds_list = []
            for uncond_input, tokenizer, text_encoder in zip(
                uncond_inputs, tokenizers, text_encoders
            ):
                negative_prompt_embeds = text_encoder(
                    uncond_input.to(device),
                    output_hidden_states=True,
                )
                # We are only ALWAYS interested in the pooled output of the final text encoder
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        if pipe.text_encoder_2 is not None:
            prompt_embeds = prompt_embeds.to(
                dtype=pipe.text_encoder_2.dtype, device=device
            )
        else:
            prompt_embeds = prompt_embeds.to(dtype=pipe.unet.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            if pipe.text_encoder_2 is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(
                    dtype=pipe.text_encoder_2.dtype, device=device
                )
            else:
                negative_prompt_embeds = negative_prompt_embeds.to(
                    dtype=pipe.unet.dtype, device=device
                )

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(
            1, num_images_per_prompt
        ).view(bs_embed * num_images_per_prompt, -1)
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(
                1, num_images_per_prompt
            ).view(bs_embed * num_images_per_prompt, -1)
        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )
    
    def prepare_inputs(self, inputs, i):
        if self.batch_size == 1:
            return self.encode_tokens(
                self.pipe,
                inputs[i]["input_tokens"],
                inputs[i]["input_tokens_2"],
                negative_prompt=self.negative_prompt_tokens,
                negative_prompt_2=self.negative_prompt_tokens_2,
            )
        else:
            prompt_embeds = []
            negative_prompt_embeds = []
            pooled_prompt_embeds = []
            negative_pooled_prompt_embeds = []
            for prompt in inputs[i:min(i+self.batch_size, len(inputs))]:
                assert isinstance(prompt, dict)
                text_input = prompt["input_tokens"]
                text_input_2 = prompt["input_tokens_2"]
                (
                    p_e,
                    n_p_e,
                    p_p_e,
                    n_p_p_e,
                ) = self.encode_tokens(
                    self.pipe,
                    text_input,
                    text_input_2,
                    negative_prompt=self.negative_prompt_tokens,
                    negative_prompt_2=self.negative_prompt_tokens_2,
                )
                prompt_embeds.append(p_e)
                negative_prompt_embeds.append(n_p_e)
                pooled_prompt_embeds.append(p_p_e)
                negative_pooled_prompt_embeds.append(n_p_p_e)


            prompt_embeds = torch.cat(prompt_embeds)
            negative_prompt_embeds = torch.cat(negative_prompt_embeds)
            pooled_prompt_embeds = torch.cat(pooled_prompt_embeds)
            negative_pooled_prompt_embeds = torch.cat(negative_pooled_prompt_embeds)
            return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def predict(self, inputs):
        images = []
        with torch.no_grad():
            for i in range(0, len(inputs), self.batch_size):
                latents_input = [inputs[idx]["latents"] for idx in range(i, min(i+self.batch_size, len(inputs)))]
                latents_input = torch.cat(latents_input).to(self.device)
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = self.prepare_inputs(inputs, i)
                generated = self.pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    guidance_scale=self.guidance,
                    num_inference_steps=self.steps,
                    output_type="pt",
                    latents=latents_input,
                ).images
                images.extend(generated)
        return images

        # @measure
    @staticmethod
    def load_mgx_model(name,
                       shapes,
                       onnx_model_path,
                       compiled_model_path=None,
                       use_fp16=False,
                       force_compile=False,
                       exhaustive_tune=False,
                       offload_copy=True):
        print(f"Loading {name} model...")
        if compiled_model_path is None:
            compiled_model_path = onnx_model_path
        onnx_file = f"{onnx_model_path}/{name}/model.onnx"
        mxr_file = f"{compiled_model_path}/{name}/model_{'fp16' if use_fp16 else 'fp32'}_{'gpu' if not offload_copy else 'oc'}.mxr"
        if not force_compile and os.path.isfile(mxr_file):
            print(f"Found mxr, loading it from {mxr_file}")
            model = mgx.load(mxr_file, format="msgpack")
        elif os.path.isfile(onnx_file):
            print(f"No mxr found at {mxr_file}")
            print(f"[IMPORTANT] Parsing from {onnx_file}")
            model = mgx.parse_onnx(onnx_file, map_input_dims=shapes)
            if use_fp16:
                print(f"[IMPORTANT] Unet quantizing to FP16...")
                mgx.quantize_fp16(model)
            model.compile(mgx.get_target("gpu"),
                          exhaustive_tune=exhaustive_tune,
                          offload_copy=offload_copy)
            print(f"Saving {name} model to {mxr_file}")
            os.makedirs(os.path.dirname(mxr_file), exist_ok=True)
            mgx.save(model, mxr_file, format="msgpack")
        else:
            print(
                f"No {name} model found at {onnx_file} or {mxr_file}. Please download it and re-try."
            )
            sys.exit(1)
        return model

    # @measure
    def tokenize(self, prompt, negative_prompt):
        def _tokenize(tokenizer):
            return self.tokenizers[tokenizer](
                [prompt, negative_prompt],
                padding="max_length",
                max_length=self.tokenizers[tokenizer].model_max_length,
                truncation=True,
                return_tensors="pt")

        tokens = _tokenize("clip")
        tokens2 = _tokenize("clip2")
        return (tokens, tokens2)

    # @measure
    def get_embeddings(self, prompt_tokens, is_refiner=False):
        def _create_embedding(model, input):
            copy_tensor(self.tensors[model]["input_ids"], input.input_ids)
            run_model_async(self.models[model], self.model_args[model],
                            self.stream)

        clip_input, clip2_input = prompt_tokens
        clip, clip2 = "clip", ("refiner_" if is_refiner else "") + "clip2"
        if not is_refiner:
            _create_embedding(clip, clip_input)
        _create_embedding(clip2, clip2_input)

        hidden_states = torch.concatenate(
            (self.tensors[clip][get_output_name(0)],
             self.tensors[clip2][get_output_name(1)]),
            axis=2) if not is_refiner else self.tensors[clip2][get_output_name(
                1)]
        text_embeds = self.tensors[clip2][get_output_name(0)]
        return (hidden_states, text_embeds)

    @staticmethod
    def convert_to_rgb_image(image):
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        return Image.fromarray(images[0])

    @staticmethod
    def save_image(pil_image, filename="output.png"):
        pil_image.save(filename)

    # @measure
    def denoise_step(self, text_embeddings, hidden_states, latents, t, scale,
                     time_ids, model):
        latents_model_input = torch.cat([latents] * 2)
        latents_model_input = self.scheduler.scale_model_input(
            latents_model_input, t).to(device="cuda")
        timestep = torch.atleast_1d(t.to(device="cuda"))  # convert 0D -> 1D

        copy_tensor(self.tensors[model]["sample"], latents_model_input)
        copy_tensor(self.tensors[model]["encoder_hidden_states"],
                    hidden_states)
        copy_tensor(self.tensors[model]["text_embeds"], text_embeddings)
        copy_tensor(self.tensors[model]["timestep"], timestep)
        copy_tensor(self.tensors[model]["time_ids"], time_ids)
        run_model_async(self.models[model], self.model_args[model],
                        self.stream)

        noise_pred_text, noise_pred_uncond = torch.tensor_split(
            self.tensors[model][get_output_name(0)], 2)

        # perform guidance
        noise_pred = noise_pred_uncond + scale * (noise_pred_text -
                                                  noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        return self.scheduler.step(noise_pred, t, latents).prev_sample

    # @measure
    def decode(self, latents):
        copy_tensor(self.tensors["vae"]["latent_sample"], latents)
        run_model_async(self.models["vae"], self.model_args["vae"],
                        self.stream)
        return self.tensors["vae"][get_output_name(0)]

    # @measure
    def warmup(self, num_runs):
        self.profile_start("warmup")
        init_fn = lambda x: torch.ones if "clip" in x else torch.randn
        for model in self.models.keys():
            for tensor in self.tensors[model].values():
                copy_tensor(tensor, init_fn(model)(tensor.size()))

        for _ in range(num_runs):
            for model in self.models.keys():
                run_model_async(self.models[model], self.model_args[model],
                                self.stream)
        self.profile_end("warmup")