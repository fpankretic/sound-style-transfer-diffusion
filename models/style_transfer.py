import inspect
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from PIL import Image
from diffusers import PNDMScheduler, UNet2DConditionModel, AutoencoderKL
from huggingface_hub import hf_hub_download
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from models.text_transform import TextTransform


def load_traced_unet(
        checkpoint: str,
        subfolder: str,
        filename: str,
        dtype: torch.dtype,
        device: str = "cuda",
        local_files_only=False,
        cache_dir: Optional[str] = None,
) -> Optional[nn.Module]:
    unet_file = hf_hub_download(
        checkpoint,
        subfolder=subfolder,
        filename=filename,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
    )
    unet_traced = torch.jit.load(unet_file)

    class TracedUNet(nn.Module):
        @dataclass
        class UNet2DConditionOutput:
            sample: torch.Tensor

        def __init__(self):
            super(TracedUNet, self).__init__()
            self.in_channels = device
            self.device = device
            self.dtype = dtype

        def forward(self, latent_model_input, t, encoder_hidden_states):
            sample = unet_traced(latent_model_input, t, encoder_hidden_states)[0]
            return self.UNet2DConditionOutput(sample=sample)

    return TracedUNet()


class SoundStyleTransferModel(nn.Module):
    def __init__(self, MODEL="riffusion/riffusion-model-v1", load_trace_unet=False):
        super(SoundStyleTransferModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.text_transform = TextTransform()

        self.vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae")
        self.vae = self.vae.to(self.device)

        self.unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet")
        self.unet = self.unet.to(self.device)

        self.scheduler = PNDMScheduler.from_config(MODEL, subfolder="scheduler")
        self.scheduler.prk_timesteps = np.array([])

        if load_trace_unet:
            traced_unet = load_traced_unet(
                MODEL,
                subfolder="unet_traced",
                filename="unet_traced.pt",
                dtype=torch.float32,
                device=self.device
            )

            if traced_unet is not None:
                print("Loaded Traced UNet")
                self.unet = traced_unet

    @torch.no_grad()
    def encode_images(self, images):
        return self.vae.encode(images).latent_dist.sample() * 0.18215

    @torch.no_grad()
    def decode_latents(self, latents):
        return self.vae.decode(latents / 0.18215).sample

    def forward(self, latents, text_embeddings, timesteps):
        result = self.unet.forward(
            latents,
            timesteps,
            text_embeddings
        ).sample
        return result

    def get_extra_kwargs(self, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
        return extra_step_kwargs

    # =========================================== Riffusion specific methods ===========================================
    def original_timestep_riffusion(self, alpha, denoising_a, denoising_b, inference_steps):
        strength = (1 - alpha) * denoising_a + alpha * denoising_b

        offset = self.scheduler.config.get("steps_offset", 0)
        init_timestep = int(inference_steps * strength) + offset
        init_timestep = min(init_timestep, inference_steps)

        timesteps = self.scheduler.timesteps[-init_timestep]
        timesteps = torch.tensor([timesteps], device=self.device)

        return timesteps, init_timestep, offset

    def noise_image_riffusion(self, latents, alpha, timesteps, dtype):
        noise_a = torch.randn(latents.shape, device=self.device, dtype=dtype)
        noise_b = torch.randn(latents.shape, device=self.device, dtype=dtype)
        noise = self.slerp(alpha, noise_a, noise_b)
        latents = self.scheduler.add_noise(latents, noise, timesteps)
        return latents

    def get_text_embeddings_riffusion(self, alpha, text_prompt_start, text_prompt_end):
        embed_start = self.text_transform.embed_text(text_prompt_start)
        embed_end = self.text_transform.embed_text(text_prompt_end)
        text_embed = embed_start * (1.0 - alpha) + embed_end * alpha
        return text_embed, embed_start.dtype

    def get_conditional_text_embeddings_riffusion(
            self,
            alpha,
            text_prompt_start,
            text_prompt_end,
            num_images_per_prompt,
            guidance_scale
    ):
        text_embed, latents_dtype = self.get_text_embeddings_riffusion(alpha, text_prompt_start, text_prompt_end)

        bs_embed, seq_len, _ = text_embed.shape
        text_embed = text_embed.repeat(1, num_images_per_prompt, 1)
        text_embed = text_embed.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if guidance_scale > 1:
            uncond_tokens = [""]
            uncond_input = self.text_transform.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.text_transform.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            uncond_ids = uncond_input.input_ids.to(self.device)
            uncond_embed = self.text_transform.text_encoder(uncond_ids)[0]
            uncond_embed = uncond_embed.repeat_interleave(bs_embed * num_images_per_prompt, dim=0)

            text_embed = torch.cat([uncond_embed, text_embed])

        return text_embed, latents_dtype

    @torch.inference_mode()
    def transfer_style_riffusion(
            self,
            init_image,
            text_prompt_start,
            text_prompt_end,
            inference_steps=50,
            denoising_a=0.75,
            denoising_b=0.75,
            guidance_a=4.0,
            guidance_b=4.0,
            alpha=0.65,
            eta=0.00,
            num_images_per_prompt=1,
    ):
        self.scheduler.set_timesteps(inference_steps)
        guidance_scale = guidance_a * (1.0 - alpha) + guidance_b * alpha

        text_embed, latents_dtype = self.get_conditional_text_embeddings_riffusion(
            alpha,
            text_prompt_start,
            text_prompt_end,
            num_images_per_prompt,
            guidance_scale
        )

        image_torch = self.preprocess_image(init_image).to(device=self.device, dtype=latents_dtype)
        init_latents = self.encode_images(image_torch)

        # Partial diffusion
        timesteps, init_timestep, offset = self.original_timestep_riffusion(alpha, denoising_a, denoising_b,
                                                                            inference_steps)
        init_latents = self.noise_image_riffusion(init_latents, alpha, timesteps, dtype=latents_dtype)

        extra_step_kwargs = self.get_extra_kwargs(eta)
        t_start = max(inference_steps - init_timestep + offset, 0)
        timesteps = self.scheduler.timesteps[t_start:].to(self.device)

        latents = init_latents
        for t in tqdm(timesteps, total=len(timesteps)):
            with torch.amp.autocast("cuda"):
                latent_input = torch.cat([latents, latents]) if guidance_scale > 1 else latents
                latent_input = self.scheduler.scale_model_input(latent_input, t)

                pred_noise = self.unet(latent_input, t, text_embed).sample

                if guidance_scale > 1:
                    pred_noise_uncond, pred_noise_text = pred_noise.chunk(2)
                    pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_text - pred_noise_uncond)

                latents = self.scheduler.step(pred_noise, t, latents, **extra_step_kwargs).prev_sample

        decoded_image = self.decode_latents(latents)
        image = (decoded_image / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).squeeze().numpy()
        image = self.numpy_to_pil(image)[0]

        return image

    # =========================================== Riffusion specific methods ===========================================

    @staticmethod
    def preprocess_image(image: Image.Image) -> torch.Tensor:
        w, h = image.size
        transformer = transforms.Compose([
            transforms.Resize((h, w - w % 32), interpolation=Image.LANCZOS),
            transforms.ToTensor()
        ])
        image_torch = transformer(image).permute(0, 2, 1).unsqueeze(0)
        return 2.0 * image_torch - 1.0

    @staticmethod
    def slerp(t: float, v0: torch.Tensor, v1: torch.Tensor, dot_threshold: float = 0.9995) -> torch.Tensor:
        if not isinstance(v0, np.ndarray):
            inputs_are_torch = True
            input_device = v0.device
            v0 = v0.cpu().numpy()
            v1 = v1.cpu().numpy()

        dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
        if np.abs(dot) > dot_threshold:
            v2 = (1 - t) * v0 + t * v1
        else:
            theta_0 = np.arccos(dot)
            sin_theta_0 = np.sin(theta_0)
            theta_t = theta_0 * t
            sin_theta_t = np.sin(theta_t)
            s0 = np.sin(theta_0 - theta_t) / sin_theta_0
            s1 = sin_theta_t / sin_theta_0
            v2 = s0 * v0 + s1 * v1

        if inputs_are_torch:
            v2 = torch.from_numpy(v2).to(input_device)

        return v2

    @staticmethod
    def numpy_to_pil(images):
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    @torch.inference_mode()
    def transfer_style(
            self,
            mel_spectrogram,
            text_prompt,
            inference_steps=50,
            strength=0.65,
            scale=4.0,
            tve=False,
            bias_reduction=False
    ):
        self.scheduler.set_timesteps(inference_steps)
        self.vae.eval()
        self.unet.eval()
        self.text_transform.tve.eval()

        image_torch = self.preprocess_image(mel_spectrogram).to(device=self.device)
        init_latents = self.encode_images(image_torch)

        offset = self.scheduler.config.get("steps_offset", 0)
        init_timestep = int(inference_steps * strength) + offset
        init_timestep = min(init_timestep, inference_steps)

        t = self.scheduler.timesteps[-init_timestep]
        t = torch.tensor([t], device=self.device)
        noise = torch.randn(init_latents.shape, device=self.device)

        if bias_reduction:
            latents = self.partial_diffusion(init_latents, noise, text_prompt, t, tve)
        else:
            latents = self.scheduler.add_noise(init_latents, noise, t)

        t_start = max(inference_steps - init_timestep + offset + 1, 0)
        timesteps = self.scheduler.timesteps[t_start:].to(self.device)
        for t in tqdm(timesteps, total=len(timesteps)):
            t = torch.tensor([t], device=self.device)
            with torch.amp.autocast("cuda"):
                latent_input = torch.cat([latents, latents]) if scale > 1 else latents
                latent_input = self.scheduler.scale_model_input(latent_input, t)

                text_embed, _ = self.get_text_embed_guided(text_prompt, scale, t, tve=tve)
                pred_noise = self.unet(latent_input, t, text_embed).sample

                if scale > 1:
                    pred_noise_uncond, pred_noise_text = pred_noise.chunk(2)
                    pred_noise = pred_noise_uncond + scale * (pred_noise_text - pred_noise_uncond)

                latents = self.scheduler.step(pred_noise, t, latents).prev_sample

        decoded_image = self.decode_latents(latents)
        image = (decoded_image / 2 + 0.5).clamp(0, 1).cpu().permute(0, 3, 2, 1).squeeze().numpy()
        image = self.numpy_to_pil(image)[0]

        return image

    def partial_diffusion(self, init_latents, noise, text_prompt, t, tve):
        latents = self.scheduler.add_noise(init_latents, noise, t)
        with torch.amp.autocast("cuda"):
            text_embed, _ = self.get_text_embed(text_prompt, t, tve=tve)
            pred_noise = self.unet(latents, t, text_embed).sample

        return self.scheduler.add_noise(init_latents, pred_noise, t)

    def get_text_embed_guided(self, text_prompt, guidance_scale, t, tve=False):
        text_embed, latents_dtype = self.get_text_embed(text_prompt, t, tve=tve)

        bs_embed, seq_len, _ = text_embed.shape
        text_embed = text_embed.repeat(1, 1, 1)
        text_embed = text_embed.view(bs_embed, seq_len, -1)

        # Negative prompt for guidance
        if guidance_scale > 1:
            uncond_embed, _ = self.get_text_embed("", t, tve=tve)
            uncond_embed = uncond_embed.repeat_interleave(bs_embed, dim=0)
            uncond_embed = uncond_embed.repeat(1, 1, 1)

            text_embed = torch.cat([uncond_embed, text_embed])
        return text_embed, latents_dtype

    def get_text_embed(self, text_prompt, t, tve=False):
        text_embed = self.text_transform.embed_text(text_prompt)
        if tve:
            text_embed = self.text_transform.tve(t, text_embed)
        return text_embed, text_embed.dtype
