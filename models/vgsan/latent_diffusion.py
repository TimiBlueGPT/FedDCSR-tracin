import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------
# 1. 时间步 embedding
# ---------------------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: [B]
        return: [B, dim]
        """
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


# ---------------------------------------
# 2. ϵ 预测网络：Transformer-based UNet
# ---------------------------------------
class EpsTransformer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=4*hidden,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 把时间 embedding 加进去
        self.time_mlp = nn.Linear(hidden, hidden)

        self.out = nn.Linear(hidden, hidden)

    def forward(self, x, t_emb):
        """
        x: [B, L, D]
        t_emb: [B, D]
        """
        # 扩展时间 embedding 到序列长度
        t_expand = t_emb.unsqueeze(1).repeat(1, x.size(1), 1)
        h = x + self.time_mlp(t_expand)
        h = self.transformer(h)
        return self.out(h)


# ---------------------------------------
# 3. Diffusion 训练与采样
# ---------------------------------------
class DiffusionModel(nn.Module):
    def __init__(self, latent_dim=128, time_steps=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_steps = time_steps

        # Beta schedule
        betas = torch.linspace(1e-4, 0.02, time_steps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(1.0 - betas, dim=0))

        # 网络
        self.time_embed = SinusoidalTimeEmbedding(latent_dim)
        self.eps_model = EpsTransformer(latent_dim)

    # q_sample：加噪声 q(x_t | x_0)
    def q_sample(self, x_0, t, noise=None):
        """
        x_0: [B, L, D]
        t: [B]
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_ac = self.alphas_cumprod[t].sqrt().view(-1, 1, 1)
        sqrt_one_minus_ac = (1 - self.alphas_cumprod[t]).sqrt().view(-1, 1, 1)

        return sqrt_ac * x_0 + sqrt_one_minus_ac * noise

    # 训练 loss：预测噪声 ε
    def p_losses(self, x_0):
        """
        x_0: [B, L, D]
        """
        B = x_0.size(0)
        device = x_0.device

        t = torch.randint(0, self.time_steps, (B,), device=device)

        noise = torch.randn_like(x_0)
        x_noisy = self.q_sample(x_0, t, noise)

        t_emb = self.time_embed(t)

        pred_noise = self.eps_model(x_noisy, t_emb)

        loss = F.mse_loss(pred_noise, noise)
        return loss

    # 采样（第三阶段增强使用）
    @torch.no_grad()
    def sample(self, x_shape):
        """
        输入 z_e 的 shape，用 DDPM reverse 采样
        """
        B, L, D = x_shape
        x = torch.randn(B, L, D).to(self.betas.device)

        for i in reversed(range(self.time_steps)):
            t = torch.full((B,), i, device=x.device)
            t_emb = self.time_embed(t)

            eps = self.eps_model(x, t_emb)

            alpha = self.alphas[i]
            alpha_cp = self.alphas_cumprod[i]
            beta = self.betas[i]

            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1 / alpha.sqrt()) * (
                x - (beta / (1 - alpha_cp).sqrt()) * eps
            ) + beta.sqrt() * noise

        return x


class FullModel(nn.Module):
    def __init__(self, main_model, diffusion_model):
        super().__init__()
        self.main = main_model
        self.diffusion = diffusion_model

    def forward(self, seqs, neg_seqs=None, aug_seqs=None, train_diffusion=False):

        if not train_diffusion:
            # 第一阶段：使用原来的 DisenVGSAN.forward
            return self.main(seqs, neg_seqs, aug_seqs)

        else:
            #print("当前为训练diffusion阶段")
            # 第二阶段：训练 diffusion，只使用 z_e
            with torch.no_grad():
                z_e = self.main.get_z_e(seqs)  # 必须提前写好
            diff_loss = self.diffusion.p_losses(z_e)
            return diff_loss
