import torch
from diffusers import DDPMScheduler
from torchvision.utils import save_image
from transformers import CLIPTokenizer, CLIPTextModel

def generate_image(model, text_prompt, output_path="generated_image.png"):
    # Load tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

    # Prepare text embeddings
    tokens = tokenizer([text_prompt], padding=True, return_tensors="pt")
    text_embeddings = text_encoder(**tokens).last_hidden_state

    # Initialize scheduler
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Sample random noise in latent space
    latents = torch.randn(1, 4, 64, 64)  # 4 latent channels for AutoencoderKL

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    latents = latents.to(device)
    text_embeddings = text_embeddings.to(device)

    # Reverse diffusion process
    for t in reversed(scheduler.timesteps):
        with torch.no_grad():
            # Predict noise at timestep `t`
            noise_pred = model.diffusion(
                sample=latents,
                timestep=torch.tensor([t]).to(device),
                encoder_hidden_states=text_embeddings,
            ).sample

            # Compute previous latent
            latents = scheduler.step(noise_pred, t, latents).prev_sample

    # Decode latents to image
    with torch.no_grad():
        latents = latents / 0.18215  # Unscaling (per Stable Diffusion convention)
        generated_image = model.vae.decode(latents).sample

    # Save or display the image
    save_image((generated_image + 1) / 2, output_path)  # Scale back to [0, 1]
    print(f"Image saved to {output_path}")

# Example usage
if __name__ == "__main__":
    trained_model = DiffusionModel()  # Replace with your trained model
    trained_model.load_state_dict(torch.load("trained_model.pth"))  # Load trained weights
    generate_image(trained_model, text_prompt="A digit 5", output_path="digit_5.png")
