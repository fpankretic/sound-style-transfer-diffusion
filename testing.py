import torch

from diffusers.models import AutoencoderKL

generator = torch.Generator()

print(generator.seed())

vae = AutoencoderKL(
    in_channels=1,
    out_channels=1,
    latent_channels=3,
)

in_x = torch.tensor([1.2]).reshape(1, 1, 1, 1)
out_latent_dists = vae.encode(in_x).latent_dist

print(out_latent_dists.sample(generator=generator))
print(out_latent_dists.sample(generator=generator))
