import numpy as np
import torch
from PIL.Image import Image
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import PNDMScheduler
from transformers import CLIPTextModel, CLIPTokenizer


def slerp(t: float, v0: torch.Tensor, v1: torch.Tensor, dot_threshold: float = 0.9995) -> torch.Tensor:
    """
    Helper function to spherically interpolate two arrays v1 v2.
    """
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


class MusicTransfer(torch.nn.Module):
    def __init__(self):
        super(MusicTransfer, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4).to(self.device)
        self.unet = UNet2DConditionModel().to(self.device)
        self.generator_a = torch.Generator(device=self.device)
        self.generator_b = torch.Generator(device=self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.scheduler: PNDMScheduler = PNDMScheduler().from_pretrained("google/ddpm-cat-256")

    def inference(self, img: Image):
        torch_img = torch.tensor(img).to(self.device)
        latent_dist = self.vae.encode(torch_img).latent_dist
        init_latents = latent_dist.sample(generator=self.generator)

        # Dodaj noise
        alpha = 0.5
        strength_a: float = 0.8
        strength_b: float = 0.8
        num_inference_steps: int = 50
        self.scheduler.set_timesteps(num_inference_steps)

        # Initialization
        strength = strength_a * (1 - alpha) + strength_b * alpha
        offset = self.scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)

        batch_size = 1
        num_images_per_prompt = 1
        timesteps = self.scheduler.timesteps[-init_timestep]
        timesteps = torch.tensor(
            [timesteps] * batch_size * num_images_per_prompt, device=self.device
        )

        # Add noise to latents
        noise_a = torch.rand(init_latents.shape, device=self.device)
        noise_b = torch.rand(init_latents.shape, device=self.device)
        noise = slerp(alpha, noise_a, noise_b)
        init_latents = self.scheduler.add_noise(init_latents, noise, timesteps)
        latents = init_latents.clone()

        t_start = max(num_inference_steps - init_timestep + offset, 0)
        timesteps = self.scheduler.timesteps[t_start:].to(self.device)
        for i in range(timesteps):
            # Predictaj noise unetom i s text_embedom
            # Izracunaj prethodni korak
            pass

        image = self.vae.decode(init_latents)

        return image

    def embed_text(self, text):
        text_input = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            embed = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return embed
