import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class NoisePredictionNetwork(nn.Module):
    """
    Neural Network epsilon_theta(x_t, t, s_t) that predicts the noise 
    injected at timestep t, conditioned on the noisy action and current state.
    """
    def __init__(self, state_dim, action_dim, time_dim=64, hidden_dim=256):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim)
        )
        
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Main network processes concatenated (noisy_action, state_features, time_features)
        self.net = nn.Sequential(
            nn.Linear(action_dim + hidden_dim + time_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, noisy_action, time, state):
        t_emb = self.time_mlp(time)
        s_emb = self.state_mlp(state)
        
        # Concatenate x_t, s_t features, and t features
        x = torch.cat([noisy_action, s_emb, t_emb], dim=-1)
        noise_pred = self.net(x)
        return noise_pred


class StateConditionedDiffusion(nn.Module):
    """
    SDA-MARL Diffusion Model for positive sample generation.
    Handles forward diffusion and state-conditioned reverse denoising.
    """
    def __init__(self, state_dim, action_dim, T=100, beta_start=1e-4, beta_end=0.02, device="cpu"):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.T = T
        self.device = device
        
        # Noise Prediction Network (epsilon_theta)
        self.model = NoisePredictionNetwork(state_dim, action_dim).to(device)
        
        # Define Noise Schedule (Linear)
        self.betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Forward diffusion calculation parameters
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
    def forward_diffusion(self, x_0, t, noise=None):
        """
        Step 1: Forward Diffusion in Action Space
        q(x_t | x_0) = N(x_t ; sqrt(alpha_bar_t)*x_0, (1-alpha_bar_t)I)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        
        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
        return x_t, noise

    def compute_loss(self, state, action):
        """
        Step 3: Training the State-Conditioned Diffusion Policy
        L_train = E[ || epsilon - epsilon_theta(x_t, t, s_t) ||^2 ]
        """
        batch_size = state.shape[0]
        
        # Randomly sample timesteps for each batch element
        t = torch.randint(0, self.T, (batch_size,), device=self.device).long()
        
        # Generate noisy action at timestep t
        noise = torch.randn_like(action)
        x_t, true_noise = self.forward_diffusion(action, t, noise)
        
        # Predict noise using epsilon_theta
        predicted_noise = self.model(x_t, t.float(), state)
        
        # MSE Loss
        loss = F.mse_loss(predicted_noise, true_noise)
        return loss

    @torch.no_grad()
    def sample(self, state):
        """
        Step 2: State-Conditioned Reverse Denoising and Sampling
        Iteratively reconstructs action from pure noise.
        """
        batch_size = state.shape[0]
        
        # Start from pure Gaussian noise in action space
        x = torch.randn((batch_size, self.action_dim), device=self.device)
        
        for i in reversed(range(0, self.T)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.model(x, t.float(), state)
            
            alpha_t = self.alphas[t][:, None]
            alpha_bar_t = self.alphas_cumprod[t][:, None]
            beta_t = self.betas[t][:, None]
            
            # Eq 15: Reverse transition step
            if i > 0:
                noise = torch.randn_like(x)
                sigma_t = torch.sqrt(beta_t) # Simplified variance scheduling
            else:
                noise = torch.zeros_like(x)
                sigma_t = torch.zeros_like(beta_t)
                
            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise) + sigma_t * noise
            
        # Optional: Clamping to valid action bounds (e.g. [-1, 1])
        x = torch.clamp(x, -1.0, 1.0)
        return x
