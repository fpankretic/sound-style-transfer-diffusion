import torch
import math
from torch import nn
from transformers import CLIPTokenizer, CLIPTextModel


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]  # 160
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # 320
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    return embedding


class TVE(nn.Module):
    def __init__(self, token_dim=768, num_att_layers=8):
        super(TVE, self).__init__()

        time_embed_dim = token_dim * 4

        self.timestep_proj = nn.Sequential(
            nn.Linear(token_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, token_dim)
        )

        self.att = nn.MultiheadAttention(token_dim, num_att_layers, dropout=0.2)
        self.cross_att = nn.MultiheadAttention(token_dim, num_att_layers, dropout=0.2)
        self.net = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(token_dim, token_dim)
        )

        # TODO: Try without
        self.norm = nn.LayerNorm(token_dim)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)


    def forward(self, timestep, text_embed):
        # t_e = self.embed(timestep)
        t_e = timestep_embedding(timestep, text_embed.shape[-1])
        t_e = self.timestep_proj(t_e)

        v_0 = t_e + text_embed
        v_0 = self.norm(v_0)

        v_i = self.att(v_0, v_0, v_0)[0]
        v_i = self.cross_att(v_i, v_0, v_0)[0]
        v_i = self.net(v_i)

        return v_i


class TextTransform(nn.Module):
    def __init__(self, MODEL="riffusion/riffusion-model-v1"):
        super(TextTransform, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.text_encoder = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder")
        self.text_encoder = self.text_encoder.to(self.device)

        self.tokenizer = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
        self.tve = TVE().to(self.device)

    @torch.no_grad()
    def embed_text(self, prompt):
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        prompt_embeds = self.text_encoder(text_input_ids.to(self.device))
        prompt_embeds = prompt_embeds[0]
        prompt_embeds_dtype = self.text_encoder.dtype
        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=self.device)

        return prompt_embeds

    def forward(self, timestep, text):
        text_embed = self.embed_text(text)
        return self.tve(timestep, text_embed)
