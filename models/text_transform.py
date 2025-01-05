import torch
from torch import nn
from transformers import CLIPTokenizer, CLIPTextModel


class TVE(nn.Module):
    def __init__(self, embed_dim=768, num_att_layers=4, seq_len=77):
        super(TVE, self).__init__()
        self.embed = nn.Embedding(1000, embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.att = nn.MultiheadAttention(embed_dim, num_att_layers)
        self.cross_att = nn.MultiheadAttention(embed_dim, num_att_layers)
        self.ff = nn.Linear(embed_dim, embed_dim)
        self.batch_norm = nn.BatchNorm1d(seq_len)

    def forward(self, timestep, text_embed):
        t_e = self.embed(timestep)
        t_e = self.linear(t_e)
        v_0 = t_e + text_embed.to(t_e.dtype)
        v_1, _ = self.att(v_0, v_0, v_0)
        v_i, _ = self.cross_att(v_1, v_0, v_0)
        v_i = self.ff(v_i)
        v_i = self.batch_norm(v_i)
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
