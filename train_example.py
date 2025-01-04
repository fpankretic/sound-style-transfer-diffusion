import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import CLIPTokenizer, CLIPTextModel


class DiffusionModel(nn.Module):
    def __init__(self, latent_dim=4, text_dim=512):
        super(DiffusionModel, self).__init__()
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        self.diffusion = UNet2DConditionModel(
            sample_size=64,  # Image size
            in_channels=latent_dim,  # Channels from VAE
            out_channels=latent_dim,
            cross_attention_dim=text_dim,
            layers_per_block=2,
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def forward(self, latents, text_embeddings, timesteps):
        result = self.diffusion(
            sample=latents,
            timestep=timesteps,
            encoder_hidden_states=text_embeddings
        ).sample

        return result

    def encode_images(self, images):
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
        return latents

    def decode_latents(self, latents):
        with torch.no_grad():
            images = self.vae.decode(latents / 0.18215).sample
        return images


def get_mnist_dataloader(batch_size=16):
    transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])
    mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    return DataLoader(mnist, batch_size=batch_size, shuffle=True)


def precompute_text_embeddings(tokenizer, text_encoder, device):
    text_embeddings = {}
    for i in range(10):
        with torch.no_grad():
            prompt = f"A digit {i}"
            tokens = tokenizer(prompt, padding=True, return_tensors="pt").to(device)
            text_embeddings[i] = text_encoder(**tokens).last_hidden_state
    return text_embeddings


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model = DiffusionModel(text_dim=512).to(device)
    model.load_state_dict(torch.load("data/diffusion_model.pth"))

    text_embeddings = precompute_text_embeddings(tokenizer, text_encoder, device)

    num_epochs = 10
    batch_size = 16
    dataloader = get_mnist_dataloader(batch_size=batch_size)

    # Optimizer and Diffusion Scheduler
    optimizer = torch.optim.AdamW(model.diffusion.parameters(), lr=1e-4)
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

    loss_fn = MSELoss()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs}")
        model.diffusion.train()
        for step, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels

            # VAE Encode
            with torch.no_grad():
                latents = model.vae.encode(images).latent_dist.sample() * 0.18215

            noise = torch.randn_like(latents)

            label_embeddings = [text_embeddings[label.item()] for label in labels]
            label_embeddings = torch.stack(label_embeddings).squeeze(dim=1).to(device)

            # Sample random timesteps
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps,
                                      (images.size(0),), device=device, dtype=torch.int64)

            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            # Predict noise
            pred_noise = model(noisy_latents, label_embeddings, timesteps)
            loss = loss_fn(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            if step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")
                #torch.save(model.state_dict(), "data/diffusion_model.pth")
