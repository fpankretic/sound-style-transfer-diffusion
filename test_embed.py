import torch
from transformers import CLIPTextModel, CLIPTokenizer


def embed_text(text):
    text_input = tokenizer(
        text,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        embed = text_encoder(text_input.input_ids.to("cuda"))[0]
    return embed


tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")

text = "A photo of a cat."
embed = embed_text(text)
print(embed)