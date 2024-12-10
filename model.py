import torch

from diffusers.models import AutoencoderKL, UNet2DConditionModel


class MusicTransfer(torch.nn.Module):
    def __init__(
            self
    ):
        super(MusicTransfer, self).__init__()
        self.vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4)
        self.unet = UNet2DConditionModel()
        self.generator = torch.Generator()

    def forward(self, img):
        latent_dist = self.vae.encode(img).latent_dist
        latents = latent_dist.sample(generator=self.generator)

        # Dodaj noise
        # scheduler.add_noise(latents, noise, timesteps)

        t = 10
        for i in range(t):
            # Predictaj noise unetom i s text_embedom
            # Izracunaj prethodni korak
            pass

        image = self.vae.decode(latents)

        return image
