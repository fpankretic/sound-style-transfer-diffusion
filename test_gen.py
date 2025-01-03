import matplotlib.pyplot as plt
import torch
from diffusers import DDPMScheduler
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel

from module2 import DiffusionModel

# Initialize the components
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
model = DiffusionModel(text_dim=512).to(device)
model.load_state_dict(torch.load("data/diffusion_model.pth"))


# Generate image for a given text prompt
def generate_image(prompt, model, tokenizer, text_encoder, scheduler, device):
    # Tokenize and encode the prompt
    with torch.no_grad():
        tokens = tokenizer(prompt, padding=True, return_tensors="pt").to(device)
        text_embeddings = text_encoder(**tokens).last_hidden_state

    # Start with random noise
    latents = torch.randn(1, 4, 64, 64).to(device)  # Batch size = 1, latent shape = (4, 64, 64)
    image = model.decode_latents(latents).permute(0, 2, 3, 1).cpu().squeeze().numpy()
    plt.imshow(image)
    plt.axis("off")
    plt.show()

    # Iteratively denoise
    model.diffusion.eval()
    for t in tqdm(reversed(range(1000))):
        with torch.no_grad():
            print(f"Timestep: {t}")

            # Create timestep tensor
            timesteps = torch.tensor([t], dtype=torch.int64, device=device)

            # Predict noise
            pred_noise = model(latents, text_embeddings, timesteps)

            # Remove noise
            latents = scheduler.step(pred_noise, t, latents).prev_sample

        if t % 50 == 0:
            image = model.decode_latents(latents).clip(0, 1).permute(0, 2, 3, 1).cpu().squeeze().numpy()
            plt.imshow(image)
            plt.axis("off")
            plt.show()

    # Decode latents to an image
    decoded_images = model.decode_latents(latents)
    image = decoded_images.clip(0, 1).permute(0, 2, 3, 1).cpu().squeeze().numpy()
    return image


# Generate an image for a specific prompt
prompt = "A digit 3"
image = generate_image(prompt, model, tokenizer, text_encoder, scheduler, device)

# Visualize the generated image
plt.imshow(image)
plt.axis("off")
plt.show()
