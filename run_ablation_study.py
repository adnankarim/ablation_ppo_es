"""
================================================================================
COMPREHENSIVE ABLATION STUDY: ES vs PPO Hyperparameters Across Dimensions
================================================================================

This script performs comprehensive hyperparameter ablation studies for both 
Evolution Strategies (ES) and PPO-DDMEC across multiple dimensions (1D to 30D).

Ablation Strategy:
- For each dimension (1D, 2D, 5D, 10D, 20D, 30D):
  * ES Ablations: sigma, learning rate (population=10 fixed)
  * PPO Ablations: kl_weight, ppo_clip, learning rate
- All experiments use pretrained DDPM models
- Comprehensive logging to WandB and local files
- Detailed plots and metrics tracking
- Overall summary comparing all configurations

Author: Automated Ablation Pipeline
Date: December 2024
================================================================================
"""

import os
import sys
import json
import csv
import datetime
import argparse
import time
import shutil
import itertools
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Optional seaborn (not critical for functionality)
try:
    import seaborn as sns
    sns.set_palette("husl")
except Exception:
    sns = None

# Set style (fallback if seaborn style unavailable)
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    try:
        plt.style.use('seaborn-whitegrid')
    except Exception:
        pass  # Use default matplotlib style

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb not available. Install with: pip install wandb")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


# ============================================================================
# INFORMATION-THEORETIC METRICS CALCULATOR
# ============================================================================

class InformationMetrics:
    """Compute information-theoretic metrics for coupling evaluation."""
    
    @staticmethod
    def entropy_gaussian(std: float) -> float:
        """Entropy of 1D Gaussian: H(X) = 0.5 * ln(2πeσ²)"""
        return 0.5 * np.log(2 * np.pi * np.e * (std ** 2 + 1e-8))
    
    @staticmethod
    def entropy_multidim_gaussian(cov_matrix: np.ndarray) -> float:
        """Entropy of multivariate Gaussian: H(X) = 0.5 * ln((2πe)^d * det(Σ))
        
        Returns total entropy (not normalized per dimension).
        """
        cov_matrix = np.atleast_2d(cov_matrix)
        d = cov_matrix.shape[0]
        
        # Add regularization for numerical stability
        reg_strength = max(1e-4, 1e-3 * np.sqrt(d))  # Scale with sqrt(dim)
        reg_cov = cov_matrix + np.eye(d) * reg_strength
        
        # Clamp diagonal to reasonable range
        diag_vals = np.diag(reg_cov).copy()
        diag_vals = np.clip(diag_vals, 0.01, 10000)  # Reasonable variance range
        np.fill_diagonal(reg_cov, diag_vals)
        
        try:
            # For high dimensions, use log-det to avoid overflow
            sign, logdet = np.linalg.slogdet(reg_cov)
            
            if sign <= 0 or not np.isfinite(logdet):
                # Fallback: use product of diagonal (assumes independence)
                logdet = np.sum(np.log(diag_vals))
            
            # H(X) = 0.5 * (d * log(2πe) + log(det(Σ)))
            entropy = 0.5 * (d * np.log(2 * np.pi * np.e) + logdet)
            
        except Exception as e:
            # If anything fails, use diagonal approximation
            logdet = np.sum(np.log(diag_vals))
            entropy = 0.5 * (d * np.log(2 * np.pi * np.e) + logdet)
        
        # Clamp to reasonable range per dimension
        # For unit variance Gaussian: H ≈ 0.5*log(2πe) ≈ 1.42 per dimension
        max_entropy_per_dim = 6.0  # Allow variance up to ~exp(10) ≈ 22000
        min_entropy_per_dim = -2.0  # Allow variance down to ~exp(-6) ≈ 0.002
        
        max_entropy = d * max_entropy_per_dim
        min_entropy = d * min_entropy_per_dim
        
        return np.clip(entropy, min_entropy, max_entropy)
    
    @staticmethod
    def kl_divergence_gaussian(mu_p: np.ndarray, std_p: np.ndarray, 
                                mu_q: np.ndarray, std_q: np.ndarray,
                                per_dimension: bool = True) -> float:
        """KL divergence between two Gaussians: KL(p||q)
        
        Args:
            per_dimension: If True, returns average KL per dimension for fair comparison
        """
        d = len(mu_p) if hasattr(mu_p, '__len__') else 1
        mu_p = np.atleast_1d(mu_p).astype(float)
        mu_q = np.atleast_1d(mu_q).astype(float)
        
        # Clamp std to reasonable range to avoid numerical issues
        std_p = np.atleast_1d(std_p).astype(float)
        std_q = np.atleast_1d(std_q).astype(float)
        std_p = np.clip(std_p, 0.01, 100.0)  # Clamp between 0.01 and 100
        std_q = np.clip(std_q, 0.01, 100.0)
        
        # Clamp means to reasonable range
        mu_p = np.clip(mu_p, -1000, 1000)
        mu_q = np.clip(mu_q, -1000, 1000)
        
        var_p = std_p ** 2
        var_q = std_q ** 2
        
        # KL formula with numerical stability
        mean_diff_sq = np.minimum((mu_p - mu_q) ** 2, 10000)  # Cap squared diff
        var_ratio = np.clip(var_p / var_q, 1e-6, 1e6)  # Cap variance ratio
        log_var_ratio = np.clip(np.log(var_q / var_p), -20, 20)  # Cap log ratio
        
        kl_per_dim = 0.5 * (mean_diff_sq / var_q + var_ratio - 1 + log_var_ratio)
        kl_total = np.sum(kl_per_dim)
        
        if per_dimension:
            return max(0.0, float(kl_total / d))  # Average per dimension
        else:
            return max(0.0, float(kl_total))
    
    @staticmethod
    def joint_entropy_from_samples(x: np.ndarray, y: np.ndarray) -> float:
        """Estimate joint entropy H(X,Y) from samples using Gaussian assumption."""
        xy = np.column_stack([x.reshape(-1, x.shape[-1]) if x.ndim > 1 else x.reshape(-1, 1),
                              y.reshape(-1, y.shape[-1]) if y.ndim > 1 else y.reshape(-1, 1)])
        
        d = xy.shape[1]
        
        # Clean the data first
        valid_mask = np.all(np.isfinite(xy), axis=1)
        if np.sum(valid_mask) < max(10, d * 2):  # Need enough samples
            return d * InformationMetrics.entropy_gaussian(1.0)
        xy = xy[valid_mask]
        
        # Clamp extreme values
        xy = np.clip(xy, -100, 100)
        
        # Compute covariance with regularization
        try:
            cov = np.cov(xy.T)
            
            # Ensure cov is 2D
            if cov.ndim == 0:
                cov = np.array([[cov]])
            
            # Check for NaN or Inf in covariance
            if not np.all(np.isfinite(cov)):
                return d * InformationMetrics.entropy_gaussian(1.0)
            
            # Add stronger ridge regularization for high dimensions
            reg_strength = max(1e-4, 1e-3 * np.sqrt(d))  # Stronger regularization
            cov = cov + np.eye(d) * reg_strength
            
            return InformationMetrics.entropy_multidim_gaussian(cov)
        except Exception as e:
            # Fallback: assume independence
            return d * InformationMetrics.entropy_gaussian(1.0)
    
    @staticmethod
    def compute_all_metrics(
        x1_gen: np.ndarray,
        x2_gen: np.ndarray,
        x1_true: np.ndarray,
        x2_true: np.ndarray,
        target_mu1: float = 2.0,
        target_mu2: float = 10.0,
        target_std: float = 1.0,
        dim: int = 1,
    ) -> Dict[str, float]:
        """Compute all information-theoretic metrics."""
        
        # Ensure proper shape
        if x1_gen.ndim == 1:
            x1_gen = x1_gen.reshape(-1, 1)
            x2_gen = x2_gen.reshape(-1, 1)
            x1_true = x1_true.reshape(-1, 1)
            x2_true = x2_true.reshape(-1, 1)
        
        # CRITICAL: Clean generated samples - remove NaN/Inf and clamp to reasonable range
        def clean_samples(x: np.ndarray, expected_mean: float) -> np.ndarray:
            """Clean samples by handling NaN/Inf and clamping extremes."""
            x = x.copy()
            # Replace NaN/Inf with expected mean
            nan_mask = ~np.isfinite(x)
            if np.any(nan_mask):
                x[nan_mask] = expected_mean
            # Clamp to reasonable range (within 50 std of expected mean)
            x = np.clip(x, expected_mean - 50, expected_mean + 50)
            return x
        
        x1_gen = clean_samples(x1_gen, target_mu1)
        x2_gen = clean_samples(x2_gen, target_mu2)
        
        # Learned statistics
        mu1_learned = np.nanmean(x1_gen, axis=0)
        mu2_learned = np.nanmean(x2_gen, axis=0)
        std1_learned = np.nanstd(x1_gen, axis=0) + 1e-8
        std2_learned = np.nanstd(x2_gen, axis=0) + 1e-8
        
        # Ensure minimum std to avoid numerical issues with untrained models
        std1_learned = np.maximum(std1_learned, 0.3)
        std2_learned = np.maximum(std2_learned, 0.3)
        
        # Target statistics
        mu1_target = np.full(dim, target_mu1)
        mu2_target = np.full(dim, target_mu2)
        std_target = np.full(dim, target_std)
        
        # KL Divergences (per-dimension for fair comparison across dims)
        # NOTE: This measures marginal distribution quality P(X) vs target
        # For synthetic Gaussian marginals with correct coupling (X2 ≈ X1 + 8),
        # KL should generally stay LOW or decrease as model learns coupling correctly.
        # KL may increase if: conditional model distorts marginal variance, sampling
        # instability dominates, or reward pushes "shortcut" solutions.
        # Use KL as a sanity check, not an expectation of increase.
        kl_1 = InformationMetrics.kl_divergence_gaussian(mu1_learned, std1_learned, mu1_target, std_target, per_dimension=True)
        kl_2 = InformationMetrics.kl_divergence_gaussian(mu2_learned, std2_learned, mu2_target, std_target, per_dimension=True)
        
        # Marginal Entropies (use learned statistics, but with minimum std guard)
        # For multivariate Gaussian with diagonal covariance (independent dims)
        h_x1 = InformationMetrics.entropy_multidim_gaussian(np.diag(std1_learned ** 2))
        h_x2 = InformationMetrics.entropy_multidim_gaussian(np.diag(std2_learned ** 2))
        
        # Joint entropies for conditional quality (true -> generated)
        h_joint_21 = InformationMetrics.joint_entropy_from_samples(x2_true, x1_gen)
        h_joint_12 = InformationMetrics.joint_entropy_from_samples(x1_true, x2_gen)
        h_joint = (h_joint_21 + h_joint_12) / 2
        
        # Directional MI - measures conditional generation quality
        # I(X;Y) = H(X) + H(Y) - H(X,Y)
        # I(X2_true; X1_gen) measures how much X2_true tells us about X1_gen
        h_x1_true = InformationMetrics.entropy_multidim_gaussian(np.eye(dim) * (target_std ** 2))
        h_x2_true = InformationMetrics.entropy_multidim_gaussian(np.eye(dim) * (target_std ** 2))
        
        # Independent joint entropy (theoretical bound if X and Y were independent)
        h_independent_joint = h_x1_true + h_x2_true
        
        mi_21 = max(0, h_x2_true + h_x1 - h_joint_21)  # I(X2_true; X1_gen)
        mi_12 = max(0, h_x1_true + h_x2 - h_joint_12)  # I(X1_true; X2_gen)
        
        # Mutual Information - symmetric true->generated MI (consistent with conditional entropy terms)
        # This measures how well the conditional models learned the coupling
        mutual_info = 0.5 * (mi_21 + mi_12)
        
        # Clamp MI to reasonable range (per dimension): [0, 6*dim] 
        # For independent: MI ≈ 0, for fully dependent: MI ≈ H(X)
        mutual_info = float(np.clip(mutual_info, 0.0, 6.0 * dim))
        
        # Also compute gen->gen MI for reference (optional, kept for backward compatibility)
        h_joint_gen = InformationMetrics.joint_entropy_from_samples(x1_gen, x2_gen)
        mi_gen_gen = max(0, h_x1 + h_x2 - h_joint_gen)
        mi_gen_gen = float(np.clip(mi_gen_gen, 0.0, 6.0 * dim))
        
        # Conditional Entropy H(X|Y) = H(X,Y) - H(Y)
        h_x1_given_x2 = max(0, h_joint_21 - h_x2_true)
        h_x2_given_x1 = max(0, h_joint_12 - h_x1_true)
        
        # Theoretical bounds
        h_theoretical = InformationMetrics.entropy_multidim_gaussian(np.eye(dim) * (target_std ** 2))
        
        # Correlation
        corrs = []
        for d in range(dim):
            c1 = np.corrcoef(x2_true[:, d], x1_gen[:, d])[0, 1]
            c2 = np.corrcoef(x1_true[:, d], x2_gen[:, d])[0, 1]
            if np.isfinite(c1):
                corrs.append(c1)
            if np.isfinite(c2):
                corrs.append(c2)
        correlation = np.mean(corrs) if corrs else 0.0
        
        # MAE
        mae_21 = np.abs(x1_gen - (x2_true - 8.0)).mean()
        mae_12 = np.abs(x2_gen - (x1_true + 8.0)).mean()
        mae = (mae_21 + mae_12) / 2
        
        return {
            'kl_x1_per_dim': float(kl_1),  # Average KL per dimension for X1 marginal
            'kl_x2_per_dim': float(kl_2),  # Average KL per dimension for X2 marginal
            'kl_marginals_per_dim_sum': float(kl_1 + kl_2),  # Sum of per-dim KLs across both marginals (NOT total KL)
            # Legacy aliases for backward compatibility
            'kl_div_1': float(kl_1),
            'kl_div_2': float(kl_2),
            'kl_div_1_per_dim': float(kl_1),
            'kl_div_2_per_dim': float(kl_2),
            'kl_per_dim_sum': float(kl_1 + kl_2),
            'kl_div_total': float(kl_1 + kl_2),
            'entropy_x1': float(h_x1),
            'entropy_x2': float(h_x2),
            'joint_entropy': float(h_joint),
            'mutual_information': float(mutual_info),
            'mi_gen_gen': float(mi_gen_gen),  # Generated->generated MI (for reference)
            'mi_x2_to_x1': float(mi_21),
            'mi_x1_to_x2': float(mi_12),
            'h_x1_given_x2': float(h_x1_given_x2),
            'h_x2_given_x1': float(h_x2_given_x1),
            'h_theoretical': float(h_theoretical),
            'h_independent_joint': float(h_independent_joint),  # Added missing metric
            'correlation': float(correlation),
            'mae': float(mae),
            'mae_x2_to_x1': float(mae_21),
            'mae_x1_to_x2': float(mae_12),
            'mu1_learned': float(np.mean(mu1_learned)),
            'mu2_learned': float(np.mean(mu2_learned)),
            'std1_learned': float(np.mean(std1_learned)),
            'std2_learned': float(np.mean(std2_learned)),
        }


# ============================================================================
# ABLATION CONFIGURATION
# ============================================================================

@dataclass
class AblationConfig:
    """Configuration for ablation study."""
    
    # Dimensions to test (1D to 30D)
    dimensions: List[int] = field(default_factory=lambda: [1, 2, 5, 10, 20, 30])
    
    # DDPM pretraining (use pretrained models)
    ddpm_epochs: int = 200
    ddpm_lr: float = 1e-3
    ddpm_batch_size: int = 128  # Increased from 64 for 2x speedup
    ddpm_timesteps: int = 100  # Reduced from 1000 for PPO feasibility
    ddpm_hidden_dim: int = 128
    ddpm_num_samples: int = 50000
    
    # Coupling training
    coupling_epochs: int = 14
    coupling_batch_size: int = 128  # Increased from 64 for 2x speedup
    coupling_num_samples: int = 30000
    warmup_epochs: int = 15  # Increased for stability in high dimensions
    warmup_lr: float = 1e-4  # Fixed LR for warmup (separate from ES/PPO ablation LR)
    num_sampling_steps: int = 100
    ppo_updates_per_epoch: int = 20  # CRITICAL: Reduce PPO compute cost (was iterating over full dataset)
    
    # ES Ablations (population size fixed)
    es_population_size: int = 15  # Fixed population size for ES
    es_sigma_values: List[float] = field(default_factory=lambda: [0.001, 0.002, 0.005, 0.01])
    es_lr_values: List[float] = field(default_factory=lambda: [0.0001, 0.0002, 0.0005, 0.001])  # Reduced to prevent divergence
    
    # PPO Ablations (tighter grid for stability)
    ppo_kl_weight_values: List[float] = field(default_factory=lambda: [1e-4, 3e-4, 1e-3, 3e-3])
    ppo_clip_values: List[float] = field(default_factory=lambda: [0.02, 0.05, 0.1])
    ppo_lr_values: List[float] = field(default_factory=lambda: [1e-5, 2e-5, 5e-5, 1e-4])
    
    # Output
    output_dir: str = "ablation_results"
    use_wandb: bool = True
    wandb_project: str = "ddmec-ablation"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    # Model management
    reuse_pretrained: bool = True  # Reuse existing pretrained models if available
    retrain_ddpm: bool = False  # If True, ignore reuse_pretrained and force retrain
    
    # Optional run filters (for job arrays / partial runs)
    only_dim: Optional[int] = None  # e.g., 10
    only_method: Optional[str] = None  # "ES" or "PPO"
    max_es_configs: Optional[int] = None  # e.g., 4
    es_config_idx: Optional[int] = None  # e.g., 0..len(es_configs)-1
    max_ppo_configs: Optional[int] = None  # e.g., 8
    ppo_config_idx: Optional[int] = None  # e.g., 0..len(ppo_configs)-1
    
    # Logging
    log_every: int = 1
    plot_every: int = 3


# ============================================================================
# MODELS (Same as main script)
# ============================================================================

class MultiDimMLP(nn.Module):
    """MLP for multi-dimensional DDPM that predicts noise."""
    
    def __init__(self, dim: int, hidden_dim: int = 128, time_embed_dim: int = 64):
        super().__init__()
        self.dim = dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
        )
        
        # Main network
        self.net = nn.Sequential(
            nn.Linear(dim + time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, t_scale: float, condition: torch.Tensor = None) -> torch.Tensor:
        t_norm = t.float().unsqueeze(-1) / t_scale
        t_emb = self.time_embed(t_norm)
        inp = torch.cat([x, t_emb], dim=-1)
        return self.net(inp)


class ConditionalMultiDimMLP(nn.Module):
    """Conditional MLP for DDMEC/ES coupling."""
    
    def __init__(self, dim: int, hidden_dim: int = 128, time_embed_dim: int = 64):
        super().__init__()
        self.dim = dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
        )
        
        # Main network
        self.net = nn.Sequential(
            nn.Linear(2 * dim + time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, t_scale: float, condition: torch.Tensor = None) -> torch.Tensor:
        t_norm = t.float().unsqueeze(-1) / t_scale
        t_emb = self.time_embed(t_norm)
        
        if condition is None:
            condition = torch.zeros_like(x)
        
        inp = torch.cat([x, condition, t_emb], dim=-1)
        return self.net(inp)


class MultiDimDDPM:
    """DDPM for arbitrary dimensions."""
    
    def __init__(
        self,
        dim: int,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        device: str = "cuda",
        conditional: bool = False,
    ):
        self.dim = dim
        self.timesteps = timesteps
        self.device = torch.device(device)
        self.conditional = conditional
        
        # Beta schedule - scale beta_end with timesteps (critical for 100 steps)
        # beta_end=0.02 is typical for 1000 steps; scale down for fewer steps
        scaled_beta_end = beta_end * (timesteps / 1000.0) if timesteps < 1000 else beta_end
        betas = torch.linspace(beta_start, scaled_beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_schedule(betas, alphas, alphas_cumprod)
        
        # Time scale for normalization (correct: use timesteps-1, not hardcoded 1000)
        self.t_scale = float(max(1, timesteps - 1))
        
        # Model
        if conditional:
            self.model = ConditionalMultiDimMLP(dim, hidden_dim).to(self.device)
        else:
            self.model = MultiDimMLP(dim, hidden_dim).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    
    def register_schedule(self, betas, alphas, alphas_cumprod):
        """Register diffusion schedule with DDPM posterior coefficients."""
        self.betas = betas.to(self.device)
        self.alphas = alphas.to(self.device)
        self.alphas_cumprod = alphas_cumprod.to(self.device)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(self.device)
        
        # DDPM posterior coefficients (for unified reverse kernel)
        # prev abar (with abar_prev[0] = 1)
        abar_prev = torch.cat([torch.ones(1, device=self.device), self.alphas_cumprod[:-1]], dim=0)
        self.alphas_cumprod_prev = abar_prev
        
        # DDPM posterior variance: beta_t * (1-abar_prev)/(1-abar_t)
        self.posterior_variance = self.betas * (1.0 - abar_prev) / (1.0 - self.alphas_cumprod + 1e-8)
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-8)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance)
        
        # posterior mean coefficients
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(abar_prev) / (1.0 - self.alphas_cumprod + 1e-8)
        )
        self.posterior_mean_coef2 = (
            (1.0 - abar_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod + 1e-8)
        )
    
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """Forward diffusion: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        return sqrt_alpha_t * x0 + sqrt_one_minus_alpha_t * noise
    
    def train_step(self, x0: torch.Tensor, condition: torch.Tensor = None) -> float:
        """Single training step."""
        self.model.train()
        batch_size = x0.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)
        
        # Sample noise
        noise = torch.randn_like(x0)
        
        # Forward diffusion
        x_t = self.q_sample(x0, t, noise)
        
        # Predict noise
        if self.conditional:
            noise_pred = self.model(x_t, t, self.t_scale, condition)
        else:
            noise_pred = self.model(x_t, t, self.t_scale)
        
        # Loss
        loss = nn.functional.mse_loss(noise_pred, noise)
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def sample(self, num_samples: int, condition: torch.Tensor = None, num_steps: int = None) -> torch.Tensor:
        """Sample from the model using DDPM with numerical stability."""
        self.model.eval()
        
        if num_steps is None:
            num_steps = self.timesteps
        
        # Start from noise
        x = torch.randn(num_samples, self.dim, device=self.device)
        
        # Prepare condition safely (prevents memory blowup from unsafe repeat)
        if condition is not None:
            if condition.dim() == 1:
                condition = condition.unsqueeze(0)
            
            n_cond = int(condition.shape[0])
            
            # If a single conditioning vector is provided, broadcast it.
            if n_cond == 1 and num_samples > 1:
                condition = condition.repeat(num_samples, 1)
            
            # If we have per-sample conditioning, it must match num_samples.
            elif n_cond != num_samples:
                if n_cond > num_samples:
                    # More conditions than samples: slice to match
                    condition = condition[:num_samples]
                else:
                    # Fewer conditions than samples: tile to match
                    reps = int(math.ceil(num_samples / n_cond))
                    condition = condition.repeat(reps, 1)[:num_samples]
        
        # Reverse diffusion - use same kernel as PPO (DDPM posterior)
        # CRITICAL: Never include t=0 in sampling loop (matches PPO/ES rollouts which iterate t=T-1...1)
        # Note: num_steps can be used for faster evaluation, but PPO training uses full timesteps
        steps = min(int(num_steps), self.timesteps) if num_steps is not None else self.timesteps
        
        # Build explicit timestep indices (ONLY t >= 1, never t=0)
        if steps == self.timesteps:
            # Full timesteps: iterate from T-1 down to 1 (skip t=0)
            t_indices = list(reversed(range(1, self.timesteps)))
        else:
            # Reduced steps: use linspace from T-1 down to 1 (end at 1, not 0)
            t_indices = np.linspace(self.timesteps - 1, 1, steps).round().astype(int)
            t_indices = np.unique(t_indices)  # remove duplicates due to rounding
            t_indices = list(reversed(t_indices))  # reverse to go from high to low
        
        for t_int in t_indices:
            t = torch.full((num_samples,), t_int, device=self.device, dtype=torch.long)
            
            # Predict noise
            if self.conditional:
                eps = self.model(x, t, self.t_scale, condition)
            else:
                eps = self.model(x, t, self.t_scale)
            
            # Clamp noise prediction to prevent explosion
            eps = torch.clamp(eps, -10.0, 10.0)
            
            # Use same p_mean_std() as PPO (unified kernel)
            mean, std = self.p_mean_std(x, t, eps)
            
            # CRITICAL: Clamp std to match training (PPO/ES rollout uses clamp_min(1e-4))
            # This ensures evaluation sampling behaves the same as training
            std = std.clamp_min(1e-4)
            
            # Robustness check: if model blew up, return zeros
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
                return torch.zeros(num_samples, self.dim, device=self.device)
            
            # Sample next state (t_int is always >= 1 here, so always add noise)
            x = mean + std * torch.randn_like(x)
            
            # Final clamp to prevent runaway values
            x = torch.clamp(x, -100.0, 100.0)
        
        return x
    
    def save(self, path: str):
        """Save model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'dim': self.dim,
            'timesteps': self.timesteps,
            'conditional': self.conditional,
        }, path)
    
    def load(self, path: str):
        """Load model with safety checks."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Verify checkpoint matches current model configuration
        assert checkpoint['dim'] == self.dim, \
            f"Checkpoint dim {checkpoint['dim']} != model dim {self.dim}"
        assert checkpoint['timesteps'] == self.timesteps, \
            f"Checkpoint timesteps {checkpoint['timesteps']} != model timesteps {self.timesteps}"
        assert checkpoint['conditional'] == self.conditional, \
            f"Checkpoint conditional {checkpoint['conditional']} != model conditional {self.conditional}"
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    @torch.no_grad()
    def p_mean_std(self, x_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor):
        """
        DDPM posterior reverse kernel p_theta(x_{t-1} | x_t, cond) using standard DDPM posterior.
        This matches the kernel used in sample() for consistency.
        
        CRITICAL: This function should NEVER be called with t=0.
        In DDPM, transitions are from x_t to x_{t-1} for t > 0 only.
        Sampling/rollouts should iterate t = T-1 ... 1 (never include t=0).
        """
        # Safety guard: t=0 should never appear in rollouts/sampling
        if torch.any(t == 0):
            raise RuntimeError("p_mean_std called with t==0; sampling/rollouts should only use t>=1.")
        
        abar_t = self.alphas_cumprod[t].view(-1, 1)
        
        # x0 estimate from epsilon prediction
        x0_hat = (x_t - torch.sqrt(1.0 - abar_t) * eps_pred) / (torch.sqrt(abar_t) + 1e-8)
        x0_hat = torch.clamp(x0_hat, -50.0, 50.0)

        # DDPM posterior mean: coef1 * x0_hat + coef2 * x_t
        coef1 = self.posterior_mean_coef1[t].view(-1, 1)
        coef2 = self.posterior_mean_coef2[t].view(-1, 1)
        mean = coef1 * x0_hat + coef2 * x_t

        # DDPM posterior variance
        var = self.posterior_variance[t].view(-1, 1)
        std = torch.sqrt(var).clamp_min(1e-8)
        
        return mean, std


# ============================================================================
# HELPER FUNCTIONS FOR PPO
# ============================================================================

def _set_seed(seed: int, deterministic: bool = False):
    """
    Unified seed setting for reproducibility (used by both PPO and ES).
    
    Args:
        seed: Random seed value
        deterministic: If True, enable CUDNN deterministic mode (slower but bitwise reproducible)
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    np.random.seed(seed % (2**32 - 1))

def _normal_logprob(x, mean, std):
    """Diagonal Gaussian log p(x | mean, std)."""
    var = std * std
    return (-0.5 * ((x - mean) ** 2 / (var + 1e-8) + 2.0 * torch.log(std + 1e-8) + math.log(2 * math.pi))).sum(dim=-1)

def _gaussian_kl_diag(mean_p, std_p, mean_q, std_q):
    """
    KL( N_p || N_q ) for diagonal Gaussians, per-sample.
    
    Returns KL divergence summed over dimensions (per-sample).
    """
    var_p = std_p.pow(2)
    var_q = std_q.pow(2)
    return 0.5 * (
        (var_p / (var_q + 1e-8)) +
        ((mean_q - mean_p).pow(2) / (var_q + 1e-8)) -
        1.0 +
        2.0 * (torch.log(std_q + 1e-8) - torch.log(std_p + 1e-8))
    ).sum(dim=-1)


# ============================================================================
# ES TRAINER
# ============================================================================

class ESTrainer:
    """
    Evolution Strategies trainer using SAME unpaired objective as PPO.
    
    Fair comparison: ES and PPO optimize identical objective J(θ) = E[reward] - λ*KL
    - Same data: unpaired marginals (independent samples)
    - Same reward: contrastive reward (r_pos - r_neg)
    - Same constraint: KL penalty to anchor model
    - Budget matching: normalized by rollout samples
    """
    
    def __init__(
        self,
        actor: MultiDimDDPM,  # Conditional model being optimized
        anchor: MultiDimDDPM,  # Frozen unconditional pretrained model (KL constraint)
        scorer: MultiDimDDPM,  # Other conditional model (fixed, provides reward)
        population_size: int = 10,
        sigma: float = 0.002,
        lr: float = 0.001,
        kl_coef: float = 1e-3,  # Same as PPO
        mc_reward_steps: int = 4,  # Same as PPO
        device: str = "cuda"
    ):
        self.actor = actor
        self.anchor = anchor
        self.scorer = scorer
        self.population_size = population_size
        self.sigma = sigma
        self.lr = lr
        self.kl_coef = kl_coef
        self.mc_reward_steps = mc_reward_steps
        self.device = torch.device(device)
        
        # Freeze anchor and scorer (they don't change during ES)
        self.anchor.model.eval()
        for p in self.anchor.model.parameters():
            p.requires_grad = False
        self.scorer.model.eval()
        for p in self.scorer.model.parameters():
            p.requires_grad = False
    
    @torch.no_grad()
    def _collect_rollout(self, y_cond: torch.Tensor):
        """Collect rollout using current actor (returns final x0 only, for reward)."""
        self.actor.model.eval()
        B = y_cond.shape[0]
        x = torch.randn(B, self.actor.dim, device=self.device)
        
        # Rollout reverse diffusion chain
        for t_int in reversed(range(1, self.actor.timesteps)):  # Skip t=0
            t = torch.full((B,), t_int, device=self.device, dtype=torch.long)
            if self.actor.conditional:
                eps = self.actor.model(x, t, self.actor.t_scale, y_cond)
            else:
                eps = self.actor.model(x, t, self.actor.t_scale)
            
            mean, std = self.actor.p_mean_std(x, t, eps)
            std = std.clamp_min(1e-4)
            
            z = torch.randn_like(x)
            x_prev = mean + std * z
            x_prev = torch.clamp(x_prev, -100.0, 100.0)  # Match PPO
            x = x_prev
        
        return x  # Final x0
    
    @torch.no_grad()
    def _collect_rollout_with_buffers(self, y_cond: torch.Tensor):
        """
        Collect rollout with state buffers (same as PPO).
        
        Returns:
            x0: Final sample
            X_t: Stacked rollout states [T*B, dim]
            T_t: Stacked timesteps [T*B]
        """
        self.actor.model.eval()
        B = y_cond.shape[0]
        x = torch.randn(B, self.actor.dim, device=self.device)
        
        xs_t = []
        ts = []
        
        # Rollout reverse diffusion chain (skip t=0, same as PPO)
        for t_int in reversed(range(1, self.actor.timesteps)):
            t = torch.full((B,), t_int, device=self.device, dtype=torch.long)
            
            if self.actor.conditional:
                eps = self.actor.model(x, t, self.actor.t_scale, y_cond)
            else:
                eps = self.actor.model(x, t, self.actor.t_scale)
            
            mean, std = self.actor.p_mean_std(x, t, eps)
            std = std.clamp_min(1e-4)
            
            z = torch.randn_like(x)
            x_prev = mean + std * z
            x_prev = torch.clamp(x_prev, -100.0, 100.0)  # Match PPO
            
            # Store state BEFORE transition (x is x_t, x_prev is x_{t-1})
            xs_t.append(x)
            ts.append(t)
            
            x = x_prev
        
        x0 = x
        X_t = torch.cat(xs_t, dim=0)  # [T*B, dim]
        T_t = torch.cat(ts, dim=0)     # [T*B]
        
        return x0, X_t, T_t
    
    @torch.no_grad()
    def _compute_reward(self, y_target: torch.Tensor, x_condition: torch.Tensor) -> torch.Tensor:
        """Compute contrastive reward (same as PPO)."""
        B = y_target.shape[0]
        
        def score(y, xcond):
            """Score function: negative denoising MSE (proxy for log-likelihood)."""
            total = torch.zeros(B, device=self.device)
            for _ in range(self.mc_reward_steps):
                t = torch.randint(0, self.scorer.timesteps, (B,), device=self.device)
                noise = torch.randn_like(y)
                y_t = self.scorer.q_sample(y, t, noise)
                if self.scorer.conditional:
                    eps_pred = self.scorer.model(y_t, t, self.scorer.t_scale, xcond)
                else:
                    eps_pred = self.scorer.model(y_t, t, self.scorer.t_scale)
                mse = ((eps_pred - noise) ** 2).mean(dim=-1)  # per-sample
                total += (-mse)
            return total / float(self.mc_reward_steps)
        
        # Positive: correct pairing
        r_pos = score(y_target, x_condition)
        
        # Negative: shuffled pairing (breaks coupling)
        x_shuf = x_condition[torch.randperm(B, device=self.device)]
        r_neg = score(y_target, x_shuf)
        
        # Contrastive reward: difference forces conditioning dependence
        return r_pos - r_neg
    
    @torch.no_grad()
    def _compute_anchor_kl(self, X_t: torch.Tensor, T_t: torch.Tensor, y_cond: torch.Tensor) -> float:
        """
        Compute KL penalty to anchor on rollout states X_t (same as PPO).
        
        CRITICAL: Must compute KL on actual rollout states X_t, not on x0.
        This matches PPO's KL computation exactly.
        
        Args:
            X_t: Rollout states [T*B, dim]
            T_t: Rollout timesteps [T*B]
            y_cond: Conditioning variable [B, dim]
        
        Returns:
            KL penalty (scalar)
        """
        B = y_cond.shape[0]
        T = X_t.shape[0] // B  # Number of transitions per sample
        
        # Repeat conditioning for each transition
        y_rep = y_cond.repeat(T, 1)  # [T*B, dim]
        
        # Actor stats (conditional)
        if self.actor.conditional:
            eps_actor = self.actor.model(X_t, T_t, self.actor.t_scale, y_rep)
        else:
            eps_actor = self.actor.model(X_t, T_t, self.actor.t_scale)
        mean_actor, std_actor = self.actor.p_mean_std(X_t, T_t, eps_actor)
        std_actor = std_actor.clamp_min(1e-4)
        
        # Anchor stats (unconditional)
        if self.anchor.conditional:
            eps_anchor = self.anchor.model(X_t, T_t, self.anchor.t_scale, None)
        else:
            eps_anchor = self.anchor.model(X_t, T_t, self.anchor.t_scale)
        mean_anchor, std_anchor = self.anchor.p_mean_std(X_t, T_t, eps_anchor)
        std_anchor = std_anchor.clamp_min(1e-4)
        
        # Full Gaussian KL (same formula as PPO)
        kl = _gaussian_kl_diag(mean_actor, std_actor, mean_anchor, std_anchor).mean()
        
        return float(kl.item())
    
    def compute_fitness(self, params_vec: torch.Tensor, y_cond: torch.Tensor, seed: int = None) -> float:
        """
        Compute fitness J(θ) = E[reward] - λ*KL (same objective as PPO).
        
        Args:
            params_vec: Parameter vector (flattened model parameters)
            y_cond: Conditioning variable (unpaired marginal sample)
            seed: Random seed for common random numbers (variance reduction)
        
        Returns:
            Fitness value (higher is better)
        """
        # Check for NaN/Inf
        if not torch.all(torch.isfinite(params_vec)):
            return -float('inf')
        
        # Load parameters into model
        vector_to_parameters(params_vec, self.actor.model.parameters())
        
        # Set seed for common random numbers (if provided)
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        self.actor.model.eval()
        with torch.no_grad():
            # Collect rollout with state buffers (for KL computation on X_t)
            x0, X_t, T_t = self._collect_rollout_with_buffers(y_cond)
            
            # Compute reward (contrastive) on final x0
            reward = self._compute_reward(y_cond, x0)
            reward_mean = reward.mean().item()
            
            # Compute KL penalty on rollout states X_t (CRITICAL: same as PPO)
            kl_penalty = self._compute_anchor_kl(X_t, T_t, y_cond)
            
            # Fitness = reward - λ*KL (same as PPO objective)
            fitness = reward_mean - self.kl_coef * kl_penalty
            
            if not np.isfinite(fitness):
                return -float('inf')
        
        return float(fitness)
    
    def train_step(self, y_cond: torch.Tensor, seed: int = None) -> Dict[str, float]:
        """
        Single ES training step using antithetic sampling.
        
        Returns:
            Dict with 'loss' (negative fitness), 'reward_mean', 'kl_penalty'
        """
        # Get current parameters
        base_params = parameters_to_vector(self.actor.model.parameters()).detach()
        
        # Generate population with antithetic sampling (variance reduction)
        half_pop = max(1, self.population_size // 2)
        param_noises = torch.randn(half_pop, base_params.numel(), device=self.device)
        param_noises = torch.cat([param_noises, -param_noises], dim=0)  # Antithetic pairs
        param_noises = param_noises[:self.population_size]  # Trim if odd
        
        # Evaluate fitness for each population member (with common random numbers)
        fitnesses = []
        rewards = []
        kls = []
        
        for i in range(param_noises.shape[0]):
            # Set parameters
            vec = base_params + self.sigma * param_noises[i]
            
            # Use common random seed for variance reduction
            eval_seed = seed if seed is not None else None
            
            # Compute fitness (reward - λ*KL)
            fitness = self.compute_fitness(vec, y_cond, seed=eval_seed)
            fitnesses.append(fitness)
            
            # Track components for logging (compute for base params once per step)
            # Note: This is the reward/KL for the CURRENT actor (base params), not the population
            # Population fitnesses are used for gradient estimation, but we log base params for clarity
            if i == 0:  # Only compute components once (for efficiency)
                # Recompute with base params to get reward/KL (KL on rollout states X_t)
                vector_to_parameters(base_params, self.actor.model.parameters())
                self.actor.model.eval()
                with torch.no_grad():
                    x0, X_t, T_t = self._collect_rollout_with_buffers(y_cond)
                    reward = self._compute_reward(y_cond, x0)
                    kl_penalty = self._compute_anchor_kl(X_t, T_t, y_cond)
                    rewards.append(float(reward.mean().item()))
                    kls.append(float(kl_penalty))
        
        # Restore base parameters
        vector_to_parameters(base_params, self.actor.model.parameters())
        
        fitnesses = np.array(fitnesses)
        
        # Check for NaN/Inf
        if not np.all(np.isfinite(fitnesses)):
            valid_mask = np.isfinite(fitnesses)
            if np.any(valid_mask):
                worst_fitness = np.min(fitnesses[valid_mask])
                fitnesses[~valid_mask] = worst_fitness
            else:
                fitnesses = np.zeros_like(fitnesses)
        
        # Update parameters using ES gradient estimate
        fitness_std = np.std(fitnesses)
        if fitness_std < 1e-10:
            # No variance - skip update
            return {
                'loss': float(-np.mean(fitnesses)),
                'reward_mean': float(np.mean(rewards)) if rewards else 0.0,
                'kl_penalty': float(np.mean(kls)) if kls else 0.0,
            }
        
        # Normalize fitnesses
        fit_tensor = torch.tensor(fitnesses, device=self.device)
        fit_tensor = (fit_tensor - fit_tensor.mean()) / (fit_tensor.std() + 1e-8)
        
        # Compute gradient estimate: g = (1/(N*σ)) * Σ (F_k * ε_k)
        # For antithetic: g = (1/(N*σ)) * Σ ((F+_k - F-_k) * ε_k)
        grad = (fit_tensor.view(-1, 1) * param_noises).mean(dim=0) / (self.sigma + 1e-12)
        
        # Gradient clipping
        grad_norm = torch.norm(grad)
        max_grad_norm = 1.0
        if grad_norm > max_grad_norm:
            grad = grad * (max_grad_norm / grad_norm)
        
        # Apply update
        new_params = base_params + self.lr * grad
        
        # Check for NaN/Inf
        if not torch.all(torch.isfinite(new_params)):
            return {
                'loss': float(-np.mean(fitnesses)),
                'reward_mean': float(np.mean(rewards)) if rewards else 0.0,
                'kl_penalty': float(np.mean(kls)) if kls else 0.0,
            }
        
        # Load updated parameters
        vector_to_parameters(new_params, self.actor.model.parameters())
        self.actor.model.train()
        
        # Return metrics: fitness stats from population, reward/KL from base params
        return {
            'loss': float(-np.mean(fitnesses)),  # Negative fitness as "loss" (from population)
            'fitness_mean': float(np.mean(fitnesses)),  # Population fitness mean
            'fitness_std': float(np.std(fitnesses)),  # Population fitness std
            'reward_base': float(np.mean(rewards)) if rewards else 0.0,  # Reward for base params (logged)
            'kl_base': float(np.mean(kls)) if kls else 0.0,  # KL for base params (logged)
            # Legacy keys for compatibility
            'reward_mean': float(np.mean(rewards)) if rewards else 0.0,
            'kl_penalty': float(np.mean(kls)) if kls else 0.0,
        }


# ============================================================================
# MLE WITH ANCHOR TRAINER (Old implementation - kept for reference)
# ============================================================================

class MLEWithAnchorTrainer:
    """
    Old MLE-based trainer (kept for reference/compatibility).
    This is NOT PPO - it's just MLE denoising loss + anchor MSE constraint.
    """
    
    def __init__(
        self,
        cond_model: MultiDimDDPM,       # Current conditional model (theta or phi)
        pretrain_model: MultiDimDDPM,   # Static anchor for KL constraint (theta* or phi*)
        kl_weight: float = 1e-3,        # Lambda in paper Eq. 9 (used in loss)
        ppo_clip: float = 0.2,          # [UNUSED] Kept for API compatibility
        lr: float = 2e-5,
        device: str = "cuda"
    ):
        self.cond_model = cond_model
        self.pretrain_model = pretrain_model
        self.kl_weight = kl_weight  # Lambda in paper Eq. 9
        self.ppo_clip = ppo_clip    
        self.device = torch.device(device)
        
        # Freeze the pretrained anchor model
        self.pretrain_model.model.eval()
        for p in self.pretrain_model.model.parameters():
            p.requires_grad = False
        
        # Separate optimizer for conditional model
        self.optimizer = torch.optim.Adam(self.cond_model.model.parameters(), lr=lr)
    
    def train_step(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> float:
        """
        Single MLE training step implementing DDMEC Equation 9 and 11.
        
        L = reconstruction_loss + lambda * KL[p_theta || p_theta_*]
        
        Where:
        - reconstruction_loss: Minimizes conditional entropy (Eq. 9)
        - KL constraint: MSE between conditional and unconditional predictions (Eq. 11)
        """
        self.cond_model.model.train()
        batch_size = x_batch.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.cond_model.timesteps, (batch_size,), device=self.device)
        
        # Sample noise
        noise = torch.randn_like(x_batch)
        
        # Forward diffusion
        x_t = self.cond_model.q_sample(x_batch, t, noise)
        
        # 1. Reconstruction Loss (The "Reward" / Conditional Entropy)
        # Minimizing this approximates minimizing -log p(x|y)
        if self.cond_model.conditional:
            noise_pred_current = self.cond_model.model(x_t, t, self.cond_model.t_scale, y_batch)
        else:
            noise_pred_current = self.cond_model.model(x_t, t, self.cond_model.t_scale)
        reconstruction_loss = nn.functional.mse_loss(noise_pred_current, noise)
        
        # 2. Marginal Constraint (KL Divergence Penalty)
        # As per Eq. 11, this is the MSE between the conditional model
        # and the frozen unconditional anchor
        with torch.no_grad():
            if self.pretrain_model.conditional:
                noise_pred_anchor = self.pretrain_model.model(x_t, t, self.pretrain_model.t_scale, None)
            else:
                noise_pred_anchor = self.pretrain_model.model(x_t, t, self.pretrain_model.t_scale)
            
        divergence_from_anchor = nn.functional.mse_loss(noise_pred_current, noise_pred_anchor)
        
        # ===== Total Loss (Equation 9) =====
        # Combine Reward + Lambda * KL
        total_loss = reconstruction_loss + (self.kl_weight * divergence_from_anchor)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()


# ============================================================================
# REAL PPO TRAINER (DDMEC PPO-style)
# ============================================================================

class DDMECPPOTrainer:
    """
    Real PPO over the reverse diffusion kernel:
      policy:  p_theta(x_{t-1} | x_t, cond) = N(mean_theta, sigma_t^2 I)

    Reward is provided by the OTHER conditional model (specular scorer),
    estimated via denoising-MSE-based log-likelihood proxy.
    """

    def __init__(
        self,
        actor: MultiDimDDPM,          # conditional model being updated
        anchor: MultiDimDDPM,         # frozen unconditional pretrained model (KL constraint)
        scorer: MultiDimDDPM,         # other conditional model (fixed during this phase)
        kl_coef: float = 1e-3,
        clip_eps: float = 0.2,
        ppo_epochs: int = 4,
        rollout_steps: int = None,    # if None, use actor.timesteps
        mc_reward_steps: int = 4,
        lr: float = 1e-5,
        device: str = "cuda",
    ):
        self.actor = actor
        self.anchor = anchor
        self.scorer = scorer

        self.kl_coef = kl_coef
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.mc_reward_steps = mc_reward_steps
        self.device = torch.device(device)

        self.rollout_steps = rollout_steps if rollout_steps is not None else actor.timesteps
        if self.rollout_steps != actor.timesteps:
            raise ValueError("For correctness, set rollout_steps == actor.timesteps (no skipping).")

        # freeze anchor always
        self.anchor.model.eval()
        for p in self.anchor.model.parameters():
            p.requires_grad = False

        self.opt = torch.optim.Adam(self.actor.model.parameters(), lr=lr)
        self.reward_ema = 0.0
        self.reward_ema_beta = 0.95

    @torch.no_grad()
    def _estimate_reward(self, y_target: torch.Tensor, x_condition: torch.Tensor) -> torch.Tensor:
        """
        Contrastive reward: log p_scorer(y_target | x_condition) - log p_scorer(y_target | x_condition_shuffled)
        
        This prevents degenerate equilibrium where both actor and scorer ignore conditioning.
        By taking the difference, we force the reward to depend on the conditioning signal.
        """
        self.scorer.model.eval()
        B = y_target.shape[0]

        def score(y, xcond):
            """Score function: negative denoising MSE (proxy for log-likelihood)."""
            total = torch.zeros(B, device=self.device)
            for _ in range(self.mc_reward_steps):
                t = torch.randint(0, self.scorer.timesteps, (B,), device=self.device)
                noise = torch.randn_like(y)
                y_t = self.scorer.q_sample(y, t, noise)
                if self.scorer.conditional:
                    eps_pred = self.scorer.model(y_t, t, self.scorer.t_scale, xcond)
                else:
                    eps_pred = self.scorer.model(y_t, t, self.scorer.t_scale)
                mse = ((eps_pred - noise) ** 2).mean(dim=-1)  # per-sample
                total += (-mse)
            return total / float(self.mc_reward_steps)

        # Positive: correct pairing
        r_pos = score(y_target, x_condition)
        
        # Negative: shuffled pairing (breaks coupling)
        x_shuf = x_condition[torch.randperm(B, device=self.device)]
        r_neg = score(y_target, x_shuf)

        # Contrastive reward: difference forces conditioning dependence
        return r_pos - r_neg

    @torch.no_grad()
    def _collect_rollout(self, y_cond: torch.Tensor):
        """
        Rollout the reverse diffusion chain and store transitions:
          (x_t, t, x_{t-1}, old_logp)
        """
        self.actor.model.eval()
        B = y_cond.shape[0]
        x = torch.randn(B, self.actor.dim, device=self.device)

        xs_t = []
        ts = []
        xs_prev = []
        old_logps = []

        # Exclude t=0 from PPO objective (deterministic transition, awkward logprob)
        for t_int in reversed(range(1, self.actor.timesteps)):  # Start from 1, skip t=0
            t = torch.full((B,), t_int, device=self.device, dtype=torch.long)
            if self.actor.conditional:
                eps = self.actor.model(x, t, self.actor.t_scale, y_cond)
            else:
                eps = self.actor.model(x, t, self.actor.t_scale)

            mean, std = self.actor.p_mean_std(x, t, eps)
            
            # Clamp std to minimum for stability
            std = std.clamp_min(1e-4)

            z = torch.randn_like(x)
            x_prev = mean + std * z
            # Clamp rollout states to prevent explosion (matches sample() behavior)
            x_prev = torch.clamp(x_prev, -100.0, 100.0)

            logp = _normal_logprob(x_prev, mean, std)

            xs_t.append(x)
            ts.append(t)
            xs_prev.append(x_prev)
            old_logps.append(logp)

            x = x_prev
        
        # Final step: deterministic x_0 (not included in PPO objective)
        # Just store final x for reward computation

        x0 = x
        return x0, xs_t, ts, xs_prev, old_logps

    def train_step(self, y_cond: torch.Tensor, seed: Optional[int] = None) -> Dict[str, float]:
        """
        One DDMEC PPO update for actor using scorer reward.
        y_cond is REAL sample from the marginal of the conditioning variable.
        seed: Optional random seed for reproducibility (matches ES for fairness).
        """
        # Set seed if provided (for fairness with ES common random numbers)
        if seed is not None:
            _set_seed(seed)
        
        # 1) rollout using current actor
        x_gen, xs_t, ts, xs_prev, old_logps = self._collect_rollout(y_cond)

        # 2) reward from fixed scorer: log p_scorer(y_cond | x_gen)
        with torch.no_grad():
            reward = self._estimate_reward(y_cond, x_gen)

            # baseline for advantage
            r_mean = reward.mean().item()
            self.reward_ema = self.reward_ema_beta * self.reward_ema + (1 - self.reward_ema_beta) * r_mean
            adv = reward - self.reward_ema
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # normalize

        # stack rollout buffers (note: t=0 excluded, so T = timesteps - 1)
        num_transitions = len(xs_t)  # Should be timesteps - 1
        X_t = torch.cat(xs_t, dim=0)           # [T*B, dim]
        T_t = torch.cat(ts, dim=0)             # [T*B]
        X_prev = torch.cat(xs_prev, dim=0)     # [T*B, dim]
        OLD_LP = torch.cat(old_logps, dim=0)   # [T*B]
        # CRITICAL: Scale advantage by 1/num_transitions to prevent early timesteps from dominating
        # (since same scalar advantage is repeated for all transitions in a trajectory)
        ADV = adv.repeat(num_transitions) / float(num_transitions)  # [T*B] - normalized per transition

        # Cache anchor stats once (they depend only on X_t, T_t which don't change during PPO epochs)
        # This saves repeated anchor forward passes (significant speedup)
        with torch.no_grad():
            if self.anchor.conditional:
                eps_anchor = self.anchor.model(X_t, T_t, self.anchor.t_scale, None)
            else:
                eps_anchor = self.anchor.model(X_t, T_t, self.anchor.t_scale)
            mean_anchor, std_anchor = self.anchor.p_mean_std(X_t, T_t, eps_anchor)
            std_anchor = std_anchor.clamp_min(1e-4)  # For safety

        # Detach and prepare conditioning for PPO epochs (constant across epochs)
        if self.actor.conditional:
            cond_rep = y_cond.detach().repeat(num_transitions, 1)
        else:
            cond_rep = None

        # 3) PPO epochs
        self.actor.model.train()
        for _ in range(self.ppo_epochs):
            if self.actor.conditional:
                eps_new = self.actor.model(X_t, T_t, self.actor.t_scale, cond_rep)
            else:
                eps_new = self.actor.model(X_t, T_t, self.actor.t_scale)
            mean_new, std_new = self.actor.p_mean_std(X_t, T_t, eps_new)
            std_new = std_new.clamp_min(1e-4)  # CRITICAL: match rollout clamp
            new_lp = _normal_logprob(X_prev, mean_new, std_new)

            # Clamp log-prob difference to prevent ratio explosion
            delta = (new_lp - OLD_LP).clamp(-20, 20)
            ratio = torch.exp(delta)
            clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
            ppo_obj = torch.min(ratio * ADV, clipped * ADV).mean()

            # anchor constraint (uses cached anchor stats)
            # Use full Gaussian KL (same formula as ES for fairness)
            anchor_kl = _gaussian_kl_diag(mean_new, std_new, mean_anchor, std_anchor).mean()

            loss = -(ppo_obj) + self.kl_coef * anchor_kl

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.model.parameters(), 1.0)
            self.opt.step()

        return {
            "loss": float(loss.item()),
            "reward_mean": float(reward.mean().item()),
            "reward_std": float(reward.std().item()),
        }


# ============================================================================
# ABLATION RUNNER
# ============================================================================

class AblationRunner:
    """Runs the complete ablation study."""
    
    def __init__(self, config: AblationConfig):
        self.config = config
        
        # Set seeds
        _set_seed(config.seed)
        
        # Create output directories
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(config.output_dir, f"run_{timestamp}")
        self.models_dir = os.path.join(self.output_dir, "models")
        self.plots_dir = os.path.join(self.output_dir, "plots")
        self.logs_dir = os.path.join(self.output_dir, "logs")
        
        # Persistent pretrained models directory (shared across runs)
        self.pretrained_models_dir = os.path.join(config.output_dir, "pretrained_models")
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.pretrained_models_dir, exist_ok=True)
        
        # Initialize wandb if available
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=config.wandb_project,
                name=f"ablation_{timestamp}",
                config=asdict(config)
            )
        
        # Store results
        self.all_results = {}
        
        print(f"Ablation study initialized")
        print(f"Output directory: {self.output_dir}")
        print(f"Pretrained models directory: {self.pretrained_models_dir}")
        print(f"Device: {config.device}")
        print(f"Dimensions: {config.dimensions}")
        print(f"Reuse pretrained models: {config.reuse_pretrained}")
        print(f"WandB: {'Enabled' if config.use_wandb and WANDB_AVAILABLE else 'Disabled'}")
    
    def smart_load_weights(self, cond_ddpm: MultiDimDDPM, base_ddpm: MultiDimDDPM, dim: int,
                          random_cond_init: bool = False, cond_scale: float = 1e-3):
        """
        Initialize ConditionalMultiDimMLP from MultiDimMLP:
          - time_embed copied exactly
          - net layers copied where shapes match
          - first net layer: copy x + time parts, init condition part (zeros or small random)
        
        Args:
            cond_ddpm: Conditional model to load weights into
            base_ddpm: Unconditional pretrained model to load from
            dim: Dimension of the data
            random_cond_init: If True, use small random weights (for PPO bootstrapping)
                            If False, use zeros (for ES warmup)
            cond_scale: Scale for random initialization (default 1e-3, use 1e-2 for high dims)
        """
        cond_m = cond_ddpm.model
        base_m = base_ddpm.model

        # 1) Copy time embedding exactly
        cond_m.time_embed.load_state_dict(base_m.time_embed.state_dict())

        # 2) Copy all shared layers except the first Linear, which has different input width
        # net: [Linear0, SiLU, Linear1, SiLU, Linear2, SiLU, Linear3]
        # Only Linear0 differs (input dim)
        for idx in [2, 4, 6]:  # indices of Linear layers after the first
            cond_m.net[idx].load_state_dict(base_m.net[idx].state_dict())

        # 3) Handle first Linear weight mapping
        # base Linear0: in = dim + t_emb
        # cond Linear0: in = 2*dim + t_emb  (x, condition, t_emb)
        base_L0 = base_m.net[0]
        cond_L0 = cond_m.net[0]

        with torch.no_grad():
            # Zero everything first
            cond_L0.weight.zero_()
            cond_L0.bias.copy_(base_L0.bias)

            # base order: [x | t_emb]
            # cond order: [x | cond | t_emb]
            t_emb_dim = base_L0.weight.shape[1] - dim

            # Copy x block
            cond_L0.weight[:, :dim].copy_(base_L0.weight[:, :dim])

            # Copy time-embedding block (base tail) into conditional tail
            cond_L0.weight[:, 2*dim:2*dim + t_emb_dim].copy_(base_L0.weight[:, dim:dim + t_emb_dim])

            # Init condition block
            if random_cond_init:
                # Scale cond_scale with dimension for better signal in high dims
                # Higher dims need larger init scale to provide measurable conditional signal
                if dim <= 10:
                    mult = 1.0
                elif dim <= 20:
                    mult = 3.0
                else:
                    mult = 10.0  # makes 1e-3 -> 1e-2 for 30D
                
                effective_scale = cond_scale * mult
                cond_L0.weight[:, dim:2*dim].normal_(mean=0.0, std=effective_scale)
                print(f"    [SMART LOAD] Spliced random cond weights (scale={effective_scale:.1e})")
            else:
                cond_L0.weight[:, dim:2*dim].zero_()
                print(f"    [SMART LOAD] Spliced zero weights for conditioning input")
    
    def run(self):
        """Run the complete ablation study."""
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE ABLATION STUDY")
        print("="*80 + "\n")
        
        start_time = time.time()
        
        # Filter dimensions if only_dim is specified
        dimensions_to_run = self.config.dimensions
        if hasattr(self.config, 'only_dim') and self.config.only_dim is not None:
            if self.config.only_dim not in self.config.dimensions:
                print(f"WARNING: --only-dim {self.config.only_dim} not in dimensions list. Ignoring.")
            else:
                dimensions_to_run = [self.config.only_dim]
        
        # For each dimension
        for dim in dimensions_to_run:
            print(f"\n{'='*80}")
            print(f"DIMENSION: {dim}D")
            print(f"{'='*80}\n")
            
            dim_results = {
                'ES': [],
                'PPO': []
            }
            
            # Pretrain DDPM for this dimension
            print(f"[{dim}D] Step 1: Pretraining DDPM...")
            ddpm_x1, ddpm_x2 = self._pretrain_ddpm(dim)
            
            # CRITICAL: Create single frozen eval set per dimension (shared across all configs)
            # This ensures fair comparison - all configs evaluated on identical test data
            num_eval = 5000 if dim >= 20 else 1000
            eval_x1_true = torch.randn(num_eval, dim, device=self.config.device) * 1.0 + 2.0
            eval_x2_true = eval_x1_true + 8.0 + 0.1 * torch.randn_like(eval_x1_true)
            print(f"  [EVAL] Created frozen eval set: {num_eval} samples (shared across all configs)")
            
            # ES Ablations
            if hasattr(self.config, 'only_method') and self.config.only_method == "PPO":
                print(f"\n[{dim}D] Step 2: ES Ablations... SKIPPED (--only-method PPO)")
                dim_results['ES'] = []
            else:
                print(f"\n[{dim}D] Step 2: ES Ablations...")
                es_configs = list(itertools.product(
                    self.config.es_sigma_values,
                    self.config.es_lr_values
                ))
                
                # Limit configs if max_es_configs is specified
                if hasattr(self.config, 'max_es_configs') and self.config.max_es_configs is not None:
                    es_configs = es_configs[:self.config.max_es_configs]
                    print(f"  [LIMIT] Running only first {len(es_configs)} ES configs (--max-es-configs)")
                
                # Select single config if es_config_idx is specified (for job arrays)
                if hasattr(self.config, 'es_config_idx') and self.config.es_config_idx is not None:
                    if self.config.es_config_idx >= len(es_configs):
                        print(f"  [ERROR] --es-config-idx {self.config.es_config_idx} >= {len(es_configs)} configs. Skipping ES.")
                        dim_results['ES'] = []
                    else:
                        es_configs = [es_configs[self.config.es_config_idx]]
                        print(f"  [SELECT] Running only ES config {self.config.es_config_idx}: {es_configs[0]}")
                
                es_results = self._run_es_experiment(
                    dim, ddpm_x1, ddpm_x2, es_configs, eval_x1_true, eval_x2_true
                )
                dim_results['ES'] = es_results
            
            # PPO Ablations
            if hasattr(self.config, 'only_method') and self.config.only_method == "ES":
                print(f"\n[{dim}D] Step 3: PPO Ablations... SKIPPED (--only-method ES)")
                dim_results['PPO'] = []
            else:
                print(f"\n[{dim}D] Step 3: PPO Ablations...")
                ppo_configs = list(itertools.product(
                    self.config.ppo_kl_weight_values,
                    self.config.ppo_clip_values,
                    self.config.ppo_lr_values
                ))
                
                # Limit configs if max_ppo_configs is specified
                if hasattr(self.config, 'max_ppo_configs') and self.config.max_ppo_configs is not None:
                    ppo_configs = ppo_configs[:self.config.max_ppo_configs]
                    print(f"  [LIMIT] Running only first {len(ppo_configs)} PPO configs (--max-ppo-configs)")
                
                # Select single config if ppo_config_idx is specified (for job arrays)
                if hasattr(self.config, 'ppo_config_idx') and self.config.ppo_config_idx is not None:
                    if self.config.ppo_config_idx >= len(ppo_configs):
                        print(f"  [ERROR] --ppo-config-idx {self.config.ppo_config_idx} >= {len(ppo_configs)} configs. Skipping PPO.")
                        dim_results['PPO'] = []
                    else:
                        ppo_configs = [ppo_configs[self.config.ppo_config_idx]]
                        print(f"  [SELECT] Running only PPO config {self.config.ppo_config_idx}: {ppo_configs[0]}")
                
                ppo_results = self._run_ppo_experiment(
                    dim, ddpm_x1, ddpm_x2, ppo_configs, eval_x1_true, eval_x2_true
                )
                dim_results['PPO'] = ppo_results
            
            # Store results
            self.all_results[dim] = dim_results
            
            # Generate dimension summary
            self._generate_dimension_summary(dim)
        
        # Generate overall summary
        print("\n" + "="*80)
        print("GENERATING OVERALL SUMMARY")
        print("="*80 + "\n")
        self._generate_overall_summary()
        
        total_time = time.time() - start_time
        print(f"\nTotal ablation time: {total_time/3600:.2f} hours")
        print(f"Results saved to: {self.output_dir}")
        
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.finish()
    
    def _pretrain_ddpm(self, dim: int) -> Tuple[MultiDimDDPM, MultiDimDDPM]:
        """Pretrain DDPM models for X1 and X2."""
        # Check if models already exist in persistent directory
        # CRITICAL: Include timesteps and hidden_dim in filename to prevent cache collisions
        tag = f"{dim}d_T{self.config.ddpm_timesteps}_H{self.config.ddpm_hidden_dim}"
        model_x1_path = os.path.join(self.pretrained_models_dir, f"ddpm_x1_{tag}.pt")
        model_x2_path = os.path.join(self.pretrained_models_dir, f"ddpm_x2_{tag}.pt")
        
        # Generate training data
        x1_data = torch.randn(self.config.ddpm_num_samples, dim) * 1.0 + 2.0
        x2_data = torch.randn(self.config.ddpm_num_samples, dim) * 1.0 + 10.0
        
        # DDPM for X1
        if (self.config.reuse_pretrained and not self.config.retrain_ddpm and os.path.exists(model_x1_path)):
            print(f"  [LOAD] Loading pretrained DDPM X1 from {model_x1_path}")
            ddpm_x1 = MultiDimDDPM(
                dim=dim,
                timesteps=self.config.ddpm_timesteps,
                hidden_dim=self.config.ddpm_hidden_dim,
                lr=self.config.ddpm_lr,
                device=self.config.device,
                conditional=False
            )
            ddpm_x1.load(model_x1_path)
        else:
            if os.path.exists(model_x1_path):
                print(f"  [SKIP] Existing model found but --retrain-ddpm flag set. Training from scratch...")
            else:
                print(f"  [TRAIN] No pretrained model found. Training DDPM X1...")
            
            ddpm_x1 = MultiDimDDPM(
                dim=dim,
                timesteps=self.config.ddpm_timesteps,
                hidden_dim=self.config.ddpm_hidden_dim,
                lr=self.config.ddpm_lr,
                device=self.config.device,
                conditional=False
            )
            
            dataset = TensorDataset(x1_data)
            dataloader = DataLoader(dataset, batch_size=self.config.ddpm_batch_size, shuffle=True)
            
            for epoch in range(self.config.ddpm_epochs):
                losses = []
                for batch in dataloader:
                    x = batch[0].to(self.config.device)
                    loss = ddpm_x1.train_step(x)
                    losses.append(loss)
                
                if (epoch + 1) % 20 == 0:
                    print(f"    Epoch {epoch+1}/{self.config.ddpm_epochs}, Loss: {np.mean(losses):.4f}")
            
            ddpm_x1.save(model_x1_path)
            print(f"  [SAVE] Saved DDPM X1 to {model_x1_path}")
        
        # DDPM for X2
        if (self.config.reuse_pretrained and not self.config.retrain_ddpm and os.path.exists(model_x2_path)):
            print(f"  [LOAD] Loading pretrained DDPM X2 from {model_x2_path}")
            ddpm_x2 = MultiDimDDPM(
                dim=dim,
                timesteps=self.config.ddpm_timesteps,
                hidden_dim=self.config.ddpm_hidden_dim,
                lr=self.config.ddpm_lr,
                device=self.config.device,
                conditional=False
            )
            ddpm_x2.load(model_x2_path)
        else:
            if os.path.exists(model_x2_path):
                print(f"  [SKIP] Existing model found but --retrain-ddpm flag set. Training from scratch...")
            else:
                print(f"  [TRAIN] No pretrained model found. Training DDPM X2...")
            
            ddpm_x2 = MultiDimDDPM(
                dim=dim,
                timesteps=self.config.ddpm_timesteps,
                hidden_dim=self.config.ddpm_hidden_dim,
                lr=self.config.ddpm_lr,
                device=self.config.device,
                conditional=False
            )
            
            dataset = TensorDataset(x2_data)
            dataloader = DataLoader(dataset, batch_size=self.config.ddpm_batch_size, shuffle=True)
            
            for epoch in range(self.config.ddpm_epochs):
                losses = []
                for batch in dataloader:
                    x = batch[0].to(self.config.device)
                    loss = ddpm_x2.train_step(x)
                    losses.append(loss)
                
                if (epoch + 1) % 20 == 0:
                    print(f"    Epoch {epoch+1}/{self.config.ddpm_epochs}, Loss: {np.mean(losses):.4f}")
            
            ddpm_x2.save(model_x2_path)
            print(f"  [SAVE] Saved DDPM X2 to {model_x2_path}")
        
        return ddpm_x1, ddpm_x2
    
    def _run_es_experiment(
        self,
        dim: int,
        ddpm_x1: MultiDimDDPM,
        ddpm_x2: MultiDimDDPM,
        es_configs: List[Tuple[float, float]],
        eval_x1_true: torch.Tensor,
        eval_x2_true: torch.Tensor
    ) -> List[Dict]:
        """Run ES experiments for all configs (using shared frozen eval set)."""
        results = []
        
        for config_idx, (sigma, lr) in enumerate(es_configs):
            print(f"\n  ES Config {config_idx+1}/{len(es_configs)}: sigma={sigma}, lr={lr}")
            result = self._run_single_es_experiment(
                dim, ddpm_x1, ddpm_x2, sigma, lr, config_idx, eval_x1_true, eval_x2_true
            )
            results.append(result)
        
        return results
    
    def _run_single_es_experiment(
        self,
        dim: int,
        ddpm_x1: MultiDimDDPM,
        ddpm_x2: MultiDimDDPM,
        sigma: float,
        lr: float,
        config_idx: int,
        eval_x1_true: torch.Tensor,
        eval_x2_true: torch.Tensor
    ) -> Dict:
        """Run single ES experiment."""
        # Create checkpoint directory
        checkpoint_dir = os.path.join(self.plots_dir, f'checkpoints_ES_{dim}D_config_{config_idx}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create conditional models and initialize from pretrained unconditional models
        # ES now uses same objective as PPO, so no separate warmup phase needed
        cond_x1 = MultiDimDDPM(
            dim=dim,
            timesteps=self.config.ddpm_timesteps,
            hidden_dim=self.config.ddpm_hidden_dim,
            lr=lr,  # Use ES LR directly
            device=self.config.device,
            conditional=True
        )
        # Apply smart loading with RANDOM init for ES (same as PPO - needed for reward signal)
        # ES now uses same contrastive reward as PPO, so needs non-zero conditioning weights
        self.smart_load_weights(cond_x1, ddpm_x1, dim, random_cond_init=True, cond_scale=1e-3)
        print(f"    [INIT] Initialized cond_x1 from pretrained DDPM X1 (random cond weights for ES-Reward)")
        
        cond_x2 = MultiDimDDPM(
            dim=dim,
            timesteps=self.config.ddpm_timesteps,
            hidden_dim=self.config.ddpm_hidden_dim,
            lr=lr,  # Use ES LR (no separate warmup phase needed)
            device=self.config.device,
            conditional=True
        )
        # Apply smart loading with RANDOM init for ES (same as PPO)
        self.smart_load_weights(cond_x2, ddpm_x2, dim, random_cond_init=True, cond_scale=1e-3)
        print(f"    [INIT] Initialized cond_x2 from pretrained DDPM X2 (random cond weights for ES-Reward)")
        
        # Generate training data - UNPAIRED (same as PPO for fair comparison)
        # ES now uses same unpaired objective as PPO: J(θ) = E[reward] - λ*KL
        # Both methods optimize identical objective on identical data regime
        # NOTE: We don't pre-allocate x1_data/x2_data here - ES samples batches on-the-fly to save VRAM
        # (especially important in high-D runs like 30D+)
        
        # No dataloader needed - ES samples batches on-the-fly (like PPO)
        
        # Initialize metrics tracking
        epoch_metrics = []
        
        # Use provided frozen eval set (shared across all configs for fair comparison)
        # Evaluate INITIAL state (before any training)
        print(f"    Evaluating initial state...")
        initial_metrics = self._evaluate_coupling(dim, cond_x1, cond_x2, x1_true=eval_x1_true, x2_true=eval_x2_true)
        
        initial_metrics['epoch'] = 0
        initial_metrics['loss'] = 0.0  # Will be computed by ES trainers
        initial_metrics['sigma'] = sigma
        initial_metrics['lr'] = lr
        initial_metrics['phase'] = 'initial'
        epoch_metrics.append(initial_metrics)
        
        print(f"    Initial: KL/dim={initial_metrics['kl_div_total']:.4f}, "
              f"MI={initial_metrics['mutual_information']:.4f}, "
              f"Corr={initial_metrics['correlation']:.4f}")
        
        # Log initial state to wandb
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                f'ES/{dim}D/config_{config_idx}/initial/kl_total': initial_metrics['kl_div_total'],
                f'ES/{dim}D/config_{config_idx}/initial/mutual_information': initial_metrics['mutual_information'],
                f'ES/{dim}D/config_{config_idx}/initial/correlation': initial_metrics['correlation'],
                f'ES/{dim}D/config_{config_idx}/initial/mae': initial_metrics['mae'],
            })
        
        # Create ES-Reward trainers (same objective as PPO)
        # Use same KL weight as PPO for fair comparison (can be made configurable later)
        kl_weight = 1e-3  # Default, same as PPO
        es_x1 = ESTrainer(
            actor=cond_x1,
            anchor=ddpm_x1,
            scorer=cond_x2,  # Fixed scorer (specular)
            population_size=self.config.es_population_size,
            sigma=sigma,
            lr=lr,
            kl_coef=kl_weight,
            mc_reward_steps=4,  # Same as PPO
            device=self.config.device
        )
        es_x2 = ESTrainer(
            actor=cond_x2,
            anchor=ddpm_x2,
            scorer=cond_x1,  # Fixed scorer (specular)
            population_size=self.config.es_population_size,
            sigma=sigma,
            lr=lr,
            kl_coef=kl_weight,
            mc_reward_steps=4,  # Same as PPO
            device=self.config.device
        )
        
        # Training loop: ES-Reward with specular alternation (same as PPO)
        # Budget matching: ES uses N fitness evals per step, PPO uses U updates per epoch
        # For fairness, we match total rollout samples (not epochs)
        
        # CRITICAL: Match rollout budget per epoch (Option A: match number of rollouts)
        # PPO: 2 phases per epoch (x1 phase + x2 phase), each phase does ppo_updates_per_epoch rollouts
        #   Total PPO rollouts per epoch = 2 * ppo_updates_per_epoch
        # ES: 2 phases per step (x1 phase + x2 phase), each phase does population_size rollouts
        #   Total ES rollouts per step = 2 * population_size
        # For fairness, match rollouts: ES_steps * (2 * pop_size) = 2 * ppo_updates_per_epoch
        #   => ES_steps = ppo_updates_per_epoch / pop_size
        num_es_steps = max(1, self.config.ppo_updates_per_epoch // self.config.es_population_size)
        
        if True:  # Always print budget info
            ppo_rollouts_per_epoch = 2 * self.config.ppo_updates_per_epoch
            es_rollouts_per_step = 2 * self.config.es_population_size
            total_es_rollouts = num_es_steps * es_rollouts_per_step
            print(f"    [BUDGET] ES steps per epoch: {num_es_steps} (pop={self.config.es_population_size}, "
                  f"PPO_rollouts/epoch={ppo_rollouts_per_epoch}, ES_rollouts/step={es_rollouts_per_step}, "
                  f"total_ES_rollouts={total_es_rollouts})")
        
        for epoch in range(self.config.coupling_epochs):
            epoch_logs = []
            
            for step in range(num_es_steps):
                # Sample unpaired marginals (same as PPO)
                batch_size = self.config.coupling_batch_size
                x1_batch = torch.randn(batch_size, dim, device=self.config.device) * 1.0 + 2.0
                x2_batch = torch.randn(batch_size, dim, device=self.config.device) * 1.0 + 10.0
                
                # Common random seed for variance reduction
                seed = epoch * 10000 + step
                
                # Phase A: update θ = P(X1|X2), scorer = φ fixed
                # Note: requires_grad toggles don't affect ES (it uses vector_to_parameters, not autograd)
                # But kept for code clarity/symmetry with PPO
                log_a = es_x1.train_step(y_cond=x2_batch, seed=seed)
                
                # Phase B: update φ = P(X2|X1), scorer = θ fixed
                # Use different seed for phase B (but deterministic)
                log_b = es_x2.train_step(y_cond=x1_batch, seed=seed + 1000000)
                
                epoch_logs.append(0.5 * (log_a.get('reward_mean', 0.0) + log_b.get('reward_mean', 0.0)))
            
            # Evaluate coupling quality after both phases
            metrics = self._evaluate_coupling(dim, cond_x1, cond_x2, x1_true=eval_x1_true, x2_true=eval_x2_true)
            metrics['epoch'] = epoch + 1
            metrics['loss'] = float(-np.mean(epoch_logs))  # Higher reward => lower "loss" for plotting
            metrics['sigma'] = sigma
            metrics['lr'] = lr
            metrics['phase'] = 'es'
            epoch_metrics.append(metrics)
            
            # Generate checkpoint plot (only every plot_every epochs to reduce IO)
            if (epoch + 1) % self.config.plot_every == 0 or (epoch + 1) == self.config.coupling_epochs:
                self._plot_checkpoint(epoch_metrics, checkpoint_dir, epoch, 'ES', dim, f'σ={sigma}, lr={lr}')
            
            # Log to wandb
            if self.config.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    f'ES/{dim}D/config_{config_idx}/epoch': epoch,
                    f'ES/{dim}D/config_{config_idx}/loss': metrics['loss'],
                    f'ES/{dim}D/config_{config_idx}/kl_total': metrics['kl_div_total'],
                    f'ES/{dim}D/config_{config_idx}/correlation': metrics['correlation'],
                    f'ES/{dim}D/config_{config_idx}/mae': metrics['mae'],
                })
            
            if (epoch + 1) % 3 == 0:
                print(f"    Epoch {epoch+1}: Loss={metrics['loss']:.4f}, KL/dim={metrics['kl_div_total']:.4f}, Corr={metrics['correlation']:.4f}")
            
            # Early stopping if ES diverges (100 per dim is very high)
            if metrics['kl_div_total'] > 100 or not np.isfinite(metrics['kl_div_total']):
                print(f"    [ERROR] ES diverged (KL/dim={metrics['kl_div_total']:.2e}), stopping early at epoch {epoch+1}")
                break
        
        # Final evaluation
        final_metrics = epoch_metrics[-1] if epoch_metrics else {'kl_div_total': float('inf'), 'correlation': 0.0, 'mae': float('inf'), 'loss': float('inf')}
        if epoch_metrics:
            final_metrics['history'] = epoch_metrics
        
        return final_metrics
    
    def _run_ppo_experiment(
        self,
        dim: int,
        ddpm_x1: MultiDimDDPM,
        ddpm_x2: MultiDimDDPM,
        ppo_configs: List[Tuple[float, float, float]],
        eval_x1_true: torch.Tensor,
        eval_x2_true: torch.Tensor
    ) -> List[Dict]:
        """Run PPO experiments for all configs (using shared frozen eval set)."""
        results = []
        
        for config_idx, (kl_weight, ppo_clip, lr) in enumerate(ppo_configs):
            print(f"\n  PPO Config {config_idx+1}/{len(ppo_configs)}: kl_weight={kl_weight:.1e}, clip={ppo_clip}, lr={lr}")
            result = self._run_single_ppo_experiment(
                dim, ddpm_x1, ddpm_x2, kl_weight, ppo_clip, lr, config_idx, eval_x1_true, eval_x2_true
            )
            result['kl_weight'] = kl_weight
            result['ppo_clip'] = ppo_clip
            result['lr'] = lr
            result['phase'] = 'ppo'
            results.append(result)
        
        return results
    
    def _run_single_ppo_experiment(
        self,
        dim: int,
        ddpm_x1: MultiDimDDPM,
        ddpm_x2: MultiDimDDPM,
        kl_weight: float,
        ppo_clip: float,
        lr: float,
        config_idx: int,
        eval_x1_true: torch.Tensor,
        eval_x2_true: torch.Tensor
    ) -> Dict:
        """Run single PPO experiment."""
        # Create checkpoint directory
        checkpoint_dir = os.path.join(self.plots_dir, f'checkpoints_PPO_{dim}D_config_{config_idx}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create conditional models and initialize from pretrained unconditional models
        cond_x1 = MultiDimDDPM(
            dim=dim,
            timesteps=self.config.ddpm_timesteps,
            hidden_dim=self.config.ddpm_hidden_dim,
            lr=lr,
            device=self.config.device,
            conditional=True
        )
        # Apply smart loading with RANDOM conditioning init for PPO bootstrapping
        # CRITICAL: Zero init makes scorer ignore condition → no reward signal → PPO can't learn
        self.smart_load_weights(cond_x1, ddpm_x1, dim, random_cond_init=True, cond_scale=1e-3)
        print(f"    [INIT] Initialized cond_x1 from pretrained DDPM X1 (random cond weights for PPO)")
        
        cond_x2 = MultiDimDDPM(
            dim=dim,
            timesteps=self.config.ddpm_timesteps,
            hidden_dim=self.config.ddpm_hidden_dim,
            lr=lr,
            device=self.config.device,
            conditional=True
        )
        # Apply smart loading with RANDOM conditioning init for PPO bootstrapping
        self.smart_load_weights(cond_x2, ddpm_x2, dim, random_cond_init=True, cond_scale=1e-3)
        print(f"    [INIT] Initialized cond_x2 from pretrained DDPM X2 (random cond weights for PPO)")
        
        # Create REAL PPO trainers with specular scoring
        # actor=cond_x1, scorer=cond_x2 (fixed during phase A)
        # actor=cond_x2, scorer=cond_x1 (fixed during phase B)
        ppo_x1 = DDMECPPOTrainer(
            actor=cond_x1, anchor=ddpm_x1, scorer=cond_x2,
            kl_coef=kl_weight, clip_eps=ppo_clip, lr=lr,
            ppo_epochs=4, rollout_steps=self.config.ddpm_timesteps,
            device=self.config.device
        )
        ppo_x2 = DDMECPPOTrainer(
            actor=cond_x2, anchor=ddpm_x2, scorer=cond_x1,
            kl_coef=kl_weight, clip_eps=ppo_clip, lr=lr,
            ppo_epochs=4, rollout_steps=self.config.ddpm_timesteps,
            device=self.config.device
        )
        
        # Training loop
        epoch_metrics = []
        
        # Generate frozen eval set for consistent evaluation across epochs
        # Increase samples for high dims to stabilize MI estimation (30D needs more samples for 60D Gaussian)
        num_eval = 5000 if dim >= 20 else 1000
        eval_x1_true = torch.randn(num_eval, dim, device=self.config.device) * 1.0 + 2.0
        eval_x2_true = eval_x1_true + 8.0 + 0.1 * torch.randn_like(eval_x1_true)
        
        # Evaluate INITIAL state (epoch 0)
        print(f"    Evaluating initial state (before PPO training)...")
        initial_metrics = self._evaluate_coupling(dim, cond_x1, cond_x2, x1_true=eval_x1_true, x2_true=eval_x2_true)
        
        initial_metrics['epoch'] = 0
        initial_metrics['loss'] = 0.0  # Will be computed from reward
        initial_metrics['kl_weight'] = kl_weight
        initial_metrics['ppo_clip'] = ppo_clip
        initial_metrics['lr'] = lr
        initial_metrics['phase'] = 'initial'
        epoch_metrics.append(initial_metrics)
        
        print(f"    Initial: KL/dim={initial_metrics['kl_div_total']:.4f}, "
              f"MI={initial_metrics['mutual_information']:.4f}, "
              f"Corr={initial_metrics['correlation']:.4f}")
        
        # Log initial state to wandb
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                f'PPO/{dim}D/config_{config_idx}/initial/kl_total': initial_metrics['kl_div_total'],
                f'PPO/{dim}D/config_{config_idx}/initial/mutual_information': initial_metrics['mutual_information'],
                f'PPO/{dim}D/config_{config_idx}/initial/correlation': initial_metrics['correlation'],
                f'PPO/{dim}D/config_{config_idx}/initial/mae': initial_metrics['mae'],
            })
        
        # ===== REAL PPO COOPERATIVE TRAINING LOOP (Specular Alternation) =====
        # CRITICAL: Sample marginals on-the-fly instead of iterating full dataset
        # This reduces compute by ~20-50x while preserving unpaired marginals setting
        for epoch in range(self.config.coupling_epochs):
            epoch_logs = []
            
            # Sample fresh unpaired marginals for each update (much faster than dataloader)
            for step in range(self.config.ppo_updates_per_epoch):
                # Sample unpaired marginals on-the-fly
                x1_batch = torch.randn(self.config.coupling_batch_size, dim, device=self.config.device) * 1.0 + 2.0
                x2_batch = torch.randn(self.config.coupling_batch_size, dim, device=self.config.device) * 1.0 + 10.0
                
                # Use same seed pattern as ES for fairness
                seed = epoch * 10000 + step
                
                # -------------------------
                # Phase A: update theta = P(X1|X2), scorer = phi fixed
                # -------------------------
                for p in cond_x2.model.parameters():
                    p.requires_grad = False
                for p in cond_x1.model.parameters():
                    p.requires_grad = True
                
                log_a = ppo_x1.train_step(y_cond=x2_batch, seed=seed)
                
                # -------------------------
                # Phase B: update phi = P(X2|X1), scorer = theta fixed
                # -------------------------
                for p in cond_x1.model.parameters():
                    p.requires_grad = False
                for p in cond_x2.model.parameters():
                    p.requires_grad = True
                
                # Use different seed for phase B (but deterministic)
                log_b = ppo_x2.train_step(y_cond=x1_batch, seed=seed + 1000000)
                
                epoch_logs.append(0.5 * (log_a["reward_mean"] + log_b["reward_mean"]))
            
            # Evaluate coupling quality after both phases
            metrics = self._evaluate_coupling(dim, cond_x1, cond_x2, x1_true=eval_x1_true, x2_true=eval_x2_true)
            metrics['epoch'] = epoch + 1  # Epochs: 1, 2, 3... (epoch 0 is initial)
            metrics['loss'] = float(-np.mean(epoch_logs))  # Higher reward => lower "loss" for plotting
            metrics['kl_weight'] = kl_weight
            metrics['ppo_clip'] = ppo_clip
            metrics['lr'] = lr
            metrics['phase'] = 'ppo'
            epoch_metrics.append(metrics)
            
            # Generate checkpoint plot (only every plot_every epochs to reduce IO)
            if (epoch + 1) % self.config.plot_every == 0 or (epoch + 1) == self.config.coupling_epochs:
                self._plot_checkpoint(epoch_metrics, checkpoint_dir, epoch, 'PPO', dim, f'kl_w={kl_weight:.1e}, clip={ppo_clip}, lr={lr}')
            
            # Log to wandb
            if self.config.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    f'PPO/{dim}D/config_{config_idx}/epoch': epoch,
                    f'PPO/{dim}D/config_{config_idx}/loss': metrics['loss'],
                    f'PPO/{dim}D/config_{config_idx}/kl_total': metrics['kl_div_total'],
                    f'PPO/{dim}D/config_{config_idx}/correlation': metrics['correlation'],
                    f'PPO/{dim}D/config_{config_idx}/mae': metrics['mae'],
                })
            
            if (epoch + 1) % 3 == 0:
                print(f"    Epoch {epoch+1}: Loss={metrics['loss']:.4f}, KL/dim={metrics['kl_div_total']:.4f}, Corr={metrics['correlation']:.4f}")
        
        # Final evaluation
        final_metrics = epoch_metrics[-1]
        final_metrics['history'] = epoch_metrics
        
        return final_metrics
    
    def _evaluate_coupling(self, dim: int, cond_x1: MultiDimDDPM, cond_x2: MultiDimDDPM, 
                          is_initial: bool = False, x1_true: torch.Tensor = None, 
                          x2_true: torch.Tensor = None) -> Dict:
        """
        Evaluate coupling quality with robust error handling.
        
        Args:
            dim: Dimension
            cond_x1: Conditional model for X1|X2
            cond_x2: Conditional model for X2|X1
            is_initial: Whether this is initial evaluation
            x1_true: Optional frozen test set for X1 (if None, generates new)
            x2_true: Optional frozen test set for X2 (if None, generates new)
        
        Note: We don't shuffle conditioning at initialization because:
        - Shuffling breaks the model's ability to generate reasonable samples (KL explodes)
        - Initial MI ~0.3-0.4 is acceptable (from pretrained model capability)
        - What matters is MI INCREASE during training (0.3 → 2.0+)
        """
        # If caller provided frozen eval sets, evaluate exactly on that size.
        # CRITICAL: Must match provided set size to prevent memory blowup in sample()
        if x1_true is not None and x2_true is not None:
            assert x1_true.shape[0] == x2_true.shape[0], "x1_true/x2_true must have same N"
            num_eval = int(x1_true.shape[0])
        else:
            # Generate new test set with appropriate size for dimension
            num_eval = 5000 if dim >= 20 else 1000
            x1_true = torch.randn(num_eval, dim, device=self.config.device) * 1.0 + 2.0
            x2_true = x1_true + 8.0 + 0.1 * torch.randn_like(x1_true)  # Add noise for finite MI
        
        try:
            
            # Sample from conditional models with error handling
            # Always use proper conditioning (no shuffling)
            with torch.no_grad():
                x1_gen = cond_x1.sample(num_eval, x2_true, self.config.num_sampling_steps)
                x2_gen = cond_x2.sample(num_eval, x1_true, self.config.num_sampling_steps)
            
            # Convert to numpy
            x1_gen_np = x1_gen.cpu().numpy()
            x2_gen_np = x2_gen.cpu().numpy()
            x1_true_np = x1_true.cpu().numpy()
            x2_true_np = x2_true.cpu().numpy()
            
            # Check for catastrophic failure (all NaN)
            if not np.any(np.isfinite(x1_gen_np)) or not np.any(np.isfinite(x2_gen_np)):
                # Return worst-case metrics
                return {
                    'kl_div_1': 100.0,
                    'kl_div_2': 100.0,
                    'kl_div_total': 200.0,
                    'entropy_x1': 0.0,
                    'entropy_x2': 0.0,
                    'joint_entropy': 0.0,
                    'mutual_information': 0.0,
                    'mi_x2_to_x1': 0.0,
                    'mi_x1_to_x2': 0.0,
                    'mi_gen_gen': 0.0,
                    'h_x1_given_x2': 0.0,
                    'h_x2_given_x1': 0.0,
                    'h_theoretical': dim * 1.42,
                    'h_independent_joint': dim * 2.84,
                    'correlation': 0.0,
                    'mae': 100.0,
                    'mae_x2_to_x1': 100.0,
                    'mae_x1_to_x2': 100.0,
                    'mu1_learned': 0.0,
                    'mu2_learned': 0.0,
                    'std1_learned': 0.0,
                    'std2_learned': 0.0,
                    'dim': dim,
                }
            
            # Compute metrics
            metrics = InformationMetrics.compute_all_metrics(
                x1_gen_np,
                x2_gen_np,
                x1_true_np,
                x2_true_np,
                dim=dim
            )
            
            # Add dimension to metrics for scoring normalization
            metrics['dim'] = dim
            
            # Add MI per-dimension as first-class metric (useful for high-dim analysis)
            metrics['mutual_information_per_dim'] = metrics.get('mutual_information', 0.0) / dim if dim > 0 else 0.0
            
            return metrics
            
        except Exception as e:
            print(f"    [ERROR] Evaluation error: {e}")
            # Return default worst-case metrics
            return {
                'kl_div_1': 100.0,
                'kl_div_2': 100.0,
                'kl_div_total': 200.0,
                'entropy_x1': 0.0,
                'entropy_x2': 0.0,
                'joint_entropy': 0.0,
                'mutual_information': 0.0,
                'mi_x2_to_x1': 0.0,
                'mi_x1_to_x2': 0.0,
                'mi_gen_gen': 0.0,
                'h_x1_given_x2': 0.0,
                'h_x2_given_x1': 0.0,
                'h_theoretical': dim * 1.42,
                'h_independent_joint': dim * 2.84,
                'correlation': 0.0,
                'mae': 100.0,
                'mae_x2_to_x1': 100.0,
                'mae_x1_to_x2': 100.0,
                'mu1_learned': 0.0,
                'mu2_learned': 0.0,
                'std1_learned': 0.0,
                'std2_learned': 0.0,
                'dim': dim,
            }
    
    def _save_metrics_to_csv(self, epoch_metrics: List[Dict], checkpoint_dir: str, method: str, dim: int):
        """Save all metrics to CSV for later analysis."""
        import csv
        
        if not epoch_metrics:
            return
        
        csv_path = os.path.join(checkpoint_dir, 'metrics.csv')
        
        # Union of keys across all epochs (stable + complete)
        # Prevents losing columns if later epochs add keys or initial has fewer
        keys = set()
        for m in epoch_metrics:
            keys |= set(m.keys())
        keys.discard('history')  # Exclude history (not CSV-serializable)
        
        # Keep epoch/phase first, rest sorted for consistency
        rest = sorted([k for k in keys if k not in ('epoch', 'phase')])
        fieldnames = ['epoch', 'phase'] + rest
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for metrics in epoch_metrics:
                # Use .get() with empty string default for missing keys
                row = {k: metrics.get(k, "") for k in fieldnames}
                writer.writerow(row)
    
    def _plot_checkpoint(self, epoch_metrics: List[Dict], checkpoint_dir: str, epoch: int, method: str, dim: int, config_str: str):
        """Generate comprehensive checkpoint plot (including warmup epochs with shading)."""
        if len(epoch_metrics) < 2:
            return
        
        # Include ALL epochs for full picture
        all_metrics = epoch_metrics
        warmup_metrics = [m for m in all_metrics if m.get('phase') in ['initial', 'warmup']]
        training_metrics = [m for m in all_metrics if m.get('phase') not in ['initial', 'warmup']]
        
        if len(all_metrics) < 2:
            return  # Not enough data yet
        
        # Save metrics to CSV only on plot checkpoints (reduces IO)
        # CSV is rewritten each time, but only when we actually plot
        self._save_metrics_to_csv(all_metrics, checkpoint_dir, method, dim)
        
        epochs = [m['epoch'] for m in all_metrics]
        
        # Find warmup/training boundary for vertical line
        warmup_boundary = None
        if warmup_metrics and training_metrics:
            warmup_end = max([m['epoch'] for m in warmup_metrics])
            training_start = min([m['epoch'] for m in training_metrics])
            warmup_boundary = (warmup_end + training_start) / 2.0
        
        # Create comprehensive plot with ALL metrics
        fig = plt.figure(figsize=(24, 16))
        gs = GridSpec(4, 4, figure=fig)
        
        # Helper function to plot with warmup/training boundary
        def plot_metric(ax, all_epochs, all_values, ylabel, title, color='blue'):
            """Plot metric with phase boundary line."""
            ax.plot(all_epochs, all_values, color=color, linewidth=2, 
                   marker='o', markersize=5, alpha=0.8, label=method)
            
            # Add vertical line at warmup/training boundary
            if warmup_boundary is not None:
                ax.axvline(x=warmup_boundary, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, label='Warmup|Training')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Row 1: Primary metrics
        ax1 = fig.add_subplot(gs[0, 0])
        plot_metric(ax1, epochs, [m['loss'] for m in all_metrics], 
                   'Loss', 'Training Loss', 'navy')
        
        ax2 = fig.add_subplot(gs[0, 1])
        plot_metric(ax2, epochs, [m['kl_div_total'] for m in all_metrics], 
                   'KL Divergence (per-dim)', 'KL Divergence', 'darkred')
        
        ax3 = fig.add_subplot(gs[0, 2])
        plot_metric(ax3, epochs, [m['correlation'] for m in all_metrics], 
                   'Correlation', 'Correlation', 'darkgreen')
        
        ax4 = fig.add_subplot(gs[0, 3])
        plot_metric(ax4, epochs, [m['mae'] for m in all_metrics], 
                   'MAE', 'Mean Absolute Error', 'purple')
        
        # Row 2: KL components and MAE directional
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.plot(epochs, [m['kl_div_1'] for m in all_metrics], 'b-', linewidth=2, marker='o', label='KL(X1)', markersize=5)
        ax5.plot(epochs, [m['kl_div_2'] for m in all_metrics], 'r-', linewidth=2, marker='o', label='KL(X2)', markersize=5)
        if warmup_boundary is not None:
            ax5.axvline(x=warmup_boundary, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('KL Divergence (per-dim)')
        ax5.set_title('Individual KL Components')
        ax5.legend()
        ax5.grid(True)
        
        ax6 = fig.add_subplot(gs[1, 1])
        ax6.plot(epochs, [m['mae_x2_to_x1'] for m in all_metrics], 'b-', linewidth=2, marker='o', label='X2→X1', markersize=5)
        ax6.plot(epochs, [m['mae_x1_to_x2'] for m in all_metrics], 'r-', linewidth=2, marker='o', label='X1→X2', markersize=5)
        if warmup_boundary is not None:
            ax6.axvline(x=warmup_boundary, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('MAE')
        ax6.set_title('Directional MAE')
        ax6.legend()
        ax6.grid(True)
        
        ax7 = fig.add_subplot(gs[1, 2])
        plot_metric(ax7, epochs, [m['mutual_information'] for m in all_metrics], 
                   'Mutual Information', 'Mutual Information I(X;Y)', 'purple')
        
        ax8 = fig.add_subplot(gs[1, 3])
        ax8.plot(epochs, [m['mi_x2_to_x1'] for m in all_metrics], 'b-', linewidth=2, marker='o', label='I(X2;X1)', markersize=5)
        ax8.plot(epochs, [m['mi_x1_to_x2'] for m in all_metrics], 'r-', linewidth=2, marker='o', label='I(X1;X2)', markersize=5)
        if warmup_boundary is not None:
            ax8.axvline(x=warmup_boundary, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
        ax8.set_xlabel('Epoch')
        ax8.set_ylabel('MI')
        ax8.set_title('Directional Mutual Information')
        ax8.legend()
        ax8.grid(True)
        
        # Row 3: Entropy metrics
        ax9 = fig.add_subplot(gs[2, 0])
        ax9.plot(epochs, [m['entropy_x1'] for m in all_metrics], 'b-', linewidth=2, marker='o', label='H(X1)', markersize=5)
        ax9.plot(epochs, [m['entropy_x2'] for m in all_metrics], 'r-', linewidth=2, marker='o', label='H(X2)', markersize=5)
        ax9.plot(epochs, [m['h_theoretical'] for m in all_metrics], 'g--', linewidth=2, label='H(theoretical)')
        if warmup_boundary is not None:
            ax9.axvline(x=warmup_boundary, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
        ax9.set_xlabel('Epoch')
        ax9.set_ylabel('Entropy')
        ax9.set_title('Marginal Entropies')
        ax9.legend()
        ax9.grid(True)
        
        ax10 = fig.add_subplot(gs[2, 1])
        ax10.plot(epochs, [m['joint_entropy'] for m in all_metrics], 'purple', linewidth=2, marker='o', markersize=5)
        if warmup_boundary is not None:
            ax10.axvline(x=warmup_boundary, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
        ax10.set_xlabel('Epoch')
        ax10.set_ylabel('Joint Entropy')
        ax10.set_title('Joint Entropy H(X,Y)')
        ax10.grid(True)
        
        ax11 = fig.add_subplot(gs[2, 2])
        ax11.plot(epochs, [m['h_x1_given_x2'] for m in all_metrics], 'b-', linewidth=2, marker='o', label='H(X1|X2)', markersize=5)
        ax11.plot(epochs, [m['h_x2_given_x1'] for m in all_metrics], 'r-', linewidth=2, marker='o', label='H(X2|X1)', markersize=5)
        if warmup_boundary is not None:
            ax11.axvline(x=warmup_boundary, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
        ax11.set_xlabel('Epoch')
        ax11.set_ylabel('Conditional Entropy')
        ax11.set_title('Conditional Entropies')
        ax11.legend()
        ax11.grid(True)
        
        ax12 = fig.add_subplot(gs[2, 3])
        # Info-theoretic relationship: I(X;Y) = H(X) - H(X|Y)
        mi_vals = [m['mutual_information'] for m in all_metrics]
        h_x1_vals = [m['entropy_x1'] for m in all_metrics]
        h_x1_given_x2_vals = [m['h_x1_given_x2'] for m in all_metrics]
        derived_mi = [max(0, h_x1_vals[i] - h_x1_given_x2_vals[i]) for i in range(len(epochs))]
        ax12.plot(epochs, mi_vals, 'b-', linewidth=2, marker='o', label='MI (computed)', markersize=5)
        ax12.plot(epochs, derived_mi, 'r--', linewidth=2, label='H(X1)-H(X1|X2)', markersize=5)
        if warmup_boundary is not None:
            ax12.axvline(x=warmup_boundary, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
        ax12.set_xlabel('Epoch')
        ax12.set_ylabel('MI')
        ax12.set_title('MI Consistency Check')
        ax12.legend()
        ax12.grid(True)
        
        # Row 4: Learned statistics
        ax13 = fig.add_subplot(gs[3, 0])
        ax13.plot(epochs, [m['mu1_learned'] for m in all_metrics], 'b-', linewidth=2, marker='o', label='μ1 learned', markersize=5)
        ax13.axhline(y=2.0, color='b', linestyle='--', label='μ1 target (2.0)')
        ax13.plot(epochs, [m['mu2_learned'] for m in all_metrics], 'r-', linewidth=2, marker='o', label='μ2 learned', markersize=5)
        ax13.axhline(y=10.0, color='r', linestyle='--', label='μ2 target (10.0)')
        if warmup_boundary is not None:
            ax13.axvline(x=warmup_boundary, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
        ax13.set_xlabel('Epoch')
        ax13.set_ylabel('Mean')
        ax13.set_title('Learned Means vs Targets')
        ax13.legend()
        ax13.grid(True)
        
        ax14 = fig.add_subplot(gs[3, 1])
        ax14.plot(epochs, [m['std1_learned'] for m in all_metrics], 'b-', linewidth=2, marker='o', label='σ1 learned', markersize=5)
        ax14.axhline(y=1.0, color='b', linestyle='--', label='σ1 target (1.0)')
        ax14.plot(epochs, [m['std2_learned'] for m in all_metrics], 'r-', linewidth=2, marker='o', label='σ2 learned', markersize=5)
        ax14.axhline(y=1.0, color='r', linestyle='--', label='σ2 target (1.0)')
        if warmup_boundary is not None:
            ax14.axvline(x=warmup_boundary, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
        ax14.set_xlabel('Epoch')
        ax14.set_ylabel('Std Dev')
        ax14.set_title('Learned Std Devs vs Targets')
        ax14.legend()
        ax14.grid(True)
        
        # Summary metrics table
        ax15 = fig.add_subplot(gs[3, 2:])
        ax15.axis('off')
        latest = all_metrics[-1]
        initial = all_metrics[0] if all_metrics else latest
        
        summary_text = f"""
{method} Training Progress - {dim}D
Config: {config_str}

Total Epochs: {len(all_metrics)} (warmup: {len(warmup_metrics)}, training: {len(training_metrics)})

Latest Metrics (Epoch {latest['epoch']}):
━━━━━━━━━━━━━━━━━━━━━━━━━━
Primary:
  Loss:         {latest['loss']:.4f}
  KL/dim:       {latest['kl_div_total']:.4f}
  Correlation:  {latest['correlation']:.4f}
  MAE:          {latest['mae']:.4f}

Information Theory:
  MI:           {latest['mutual_information']:.4f} (init: {initial.get('mutual_information', 0):.4f})
  H(X1):        {latest['entropy_x1']:.4f}
  H(X2):        {latest['entropy_x2']:.4f}
  H(X,Y):       {latest['joint_entropy']:.4f}
  H(X1|X2):     {latest['h_x1_given_x2']:.4f}

Learned vs Target:
  μ1: {latest['mu1_learned']:.3f} (target: 2.0)
  μ2: {latest['mu2_learned']:.3f} (target: 10.0)
  σ1: {latest['std1_learned']:.3f} (target: 1.0)
  σ2: {latest['std2_learned']:.3f} (target: 1.0)
        """
        ax15.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                 verticalalignment='center', transform=ax15.transAxes,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Use actual epoch number from metrics (includes warmup epochs)
        actual_epoch = latest['epoch']
        plt.suptitle(f'{method} Training Checkpoint - Epoch {actual_epoch} ({dim}D) [Dashed Line = Warmup|Training]', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Generate filename based on actual epoch number from metrics
        if actual_epoch < 0:
            plot_filename = f'warmup_{abs(actual_epoch):02d}.png'
        else:
            plot_filename = f'epoch_{actual_epoch:03d}.png'
        
        plot_path = os.path.join(checkpoint_dir, plot_filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _score_config(self, metrics: Dict) -> float:
        """
        Score configuration: lower is better.
        Uses MAE (primary) + correlation/MI (secondary) with mild KL constraint.
        This prevents selecting models that ignore conditioning (low KL but high MAE).
        
        CRITICAL: MI is normalized by dimension to prevent dimension bias.
        """
        mae = metrics.get('mae', float('inf'))
        kl = metrics.get('kl_div_total', float('inf'))
        corr = metrics.get('correlation', 0.0)
        mi = metrics.get('mutual_information', 0.0)
        dim = max(1, int(metrics.get('dim', 1)))  # Get dimension from metrics
        
        # Normalize MI by dimension to prevent dimension bias
        mi_norm = mi / dim
        
        # Penalize high MAE (coupling failure), high KL (marginal divergence)
        # Reward high correlation and MI (coupling success)
        score = (
            3.0 * mae +                    # coupling accuracy (most important)
            0.2 * kl -                     # keep marginals sane (mild constraint)
            1.0 * corr -                   # reward coupling
            0.2 * mi_norm                  # reward mutual information (normalized per dim)
        )
        return score
    
    def _generate_dimension_summary(self, dim: int):
        """Generate summary for a specific dimension."""
        results = self.all_results[dim]
        
        # Find best configurations using composite score (not just KL)
        # KL alone can select models that ignore conditioning
        has_es = len(results['ES']) > 0
        has_ppo = len(results['PPO']) > 0
        
        best_es = min(results['ES'], key=self._score_config) if has_es else None
        best_ppo = min(results['PPO'], key=self._score_config) if has_ppo else None
        
        summary_path = os.path.join(self.logs_dir, f"summary_{dim}d.txt")
        
        with open(summary_path, 'w') as f:
            f.write(f"="*80 + "\n")
            f.write(f"ABLATION SUMMARY: {dim}D\n")
            f.write(f"="*80 + "\n\n")
            
            if has_es:
                f.write("BEST ES CONFIGURATION:\n")
                f.write(f"  sigma: {best_es['sigma']}\n")
                f.write(f"  lr: {best_es['lr']}\n")
                f.write(f"  KL Total: {best_es['kl_div_total']:.4f}\n")
                f.write(f"  Correlation: {best_es['correlation']:.4f}\n")
                f.write(f"  MAE: {best_es['mae']:.4f}\n\n")
            else:
                f.write("BEST ES CONFIGURATION: (skipped --only-method PPO)\n\n")
            
            if has_ppo:
                f.write("BEST PPO CONFIGURATION:\n")
                f.write(f"  kl_weight: {best_ppo['kl_weight']:.1e}\n")
                f.write(f"  ppo_clip: {best_ppo['ppo_clip']}\n")
                f.write(f"  lr: {best_ppo['lr']}\n")
                f.write(f"  KL Total: {best_ppo['kl_div_total']:.4f}\n")
                f.write(f"  Correlation: {best_ppo['correlation']:.4f}\n")
                f.write(f"  MAE: {best_ppo['mae']:.4f}\n\n")
            else:
                f.write("BEST PPO CONFIGURATION: (skipped --only-method ES)\n\n")
            
            if has_es and has_ppo:
                f.write(f"WINNER: {'ES' if self._score_config(best_es) < self._score_config(best_ppo) else 'PPO'}\n")
            elif has_es:
                f.write("WINNER: ES (only method run)\n")
            elif has_ppo:
                f.write("WINNER: PPO (only method run)\n")
            else:
                f.write("WINNER: (no methods run)\n")
        
        print(f"  Dimension summary saved: {summary_path}")
        
        # Save all results to CSV for this dimension
        self._save_dimension_results_to_csv(dim, results)
        
        # Generate plots
        self._plot_dimension_ablations(dim)
    
    def _save_dimension_results_to_csv(self, dim: int, results: Dict):
        """Save all configuration results to CSV for later analysis."""
        import csv
        
        # Save ES results
        es_csv_path = os.path.join(self.logs_dir, f"es_results_{dim}d.csv")
        if results['ES']:
            fieldnames = [k for k in results['ES'][0].keys() if k != 'history']
            with open(es_csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for result in results['ES']:
                    row = {k: v for k, v in result.items() if k in fieldnames}
                    writer.writerow(row)
            print(f"  ES results CSV saved: {es_csv_path}")
        
        # Save PPO results
        ppo_csv_path = os.path.join(self.logs_dir, f"ppo_results_{dim}d.csv")
        if results['PPO']:
            fieldnames = [k for k in results['PPO'][0].keys() if k != 'history']
            with open(ppo_csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for result in results['PPO']:
                    row = {k: v for k, v in result.items() if k in fieldnames}
                    writer.writerow(row)
            print(f"  PPO results CSV saved: {ppo_csv_path}")
    
    def _plot_dimension_ablations(self, dim: int):
        """Generate comprehensive ablation plots for a dimension with ALL metrics."""
        results = self.all_results[dim]
        
        # Helper to aggregate PPO results over clip (best over clip for each kl_weight, lr)
        def best_over_clip(rows):
            """Select best config across clip values using scoring function."""
            return min(rows, key=self._score_config)
        
        # Aggregate PPO results: for each (kl_weight, lr), take best over clip
        ppo_agg = {}
        for r in results['PPO']:
            key = (r['kl_weight'], r['lr'])
            ppo_agg.setdefault(key, []).append(r)
        ppo_best = [best_over_clip(v) for v in ppo_agg.values()] if len(results['PPO']) > 0 else []
        
        # Main ablation plot (expanded from 9 to 16 subplots + text panel)
        fig = plt.figure(figsize=(28, 24))
        gs = GridSpec(5, 4, figure=fig)  # 5 rows to accommodate text panel
        
        # ES: sigma vs KL
        ax1 = fig.add_subplot(gs[0, 0])
        if len(results['ES']) == 0:
            ax1.text(0.5, 0.5, f"ES skipped ({dim}D)", ha="center", va="center", fontsize=12)
            ax1.axis("off")
        else:
            sigma_vals = sorted(set(r['sigma'] for r in results['ES']))
            for sigma in sigma_vals:
                data = [r for r in results['ES'] if r['sigma'] == sigma]
                lrs = [r['lr'] for r in data]
                kls = [r['kl_div_total'] for r in data]
                ax1.plot(lrs, kls, marker='o', label=f'σ={sigma}')
            ax1.set_xlabel('Learning Rate')
            ax1.set_ylabel('KL Total')
            ax1.set_title(f'ES: LR vs KL ({dim}D)')
            ax1.set_xscale('log')
            ax1.legend()
            ax1.grid(True)
        
        # ES: sigma vs Correlation
        ax2 = fig.add_subplot(gs[0, 1])
        if len(results['ES']) == 0:
            ax2.text(0.5, 0.5, f"ES skipped ({dim}D)", ha="center", va="center", fontsize=12)
            ax2.axis("off")
        else:
            sigma_vals = sorted(set(r['sigma'] for r in results['ES']))
            for sigma in sigma_vals:
                data = [r for r in results['ES'] if r['sigma'] == sigma]
                lrs = [r['lr'] for r in data]
                corrs = [r['correlation'] for r in data]
                ax2.plot(lrs, corrs, marker='o', label=f'σ={sigma}')
            ax2.set_xlabel('Learning Rate')
            ax2.set_ylabel('Correlation')
            ax2.set_title(f'ES: LR vs Correlation ({dim}D)')
            ax2.set_xscale('log')
            ax2.legend()
            ax2.grid(True)
        
        # ES: sigma vs MAE
        ax3 = fig.add_subplot(gs[0, 2])
        if len(results['ES']) == 0:
            ax3.text(0.5, 0.5, f"ES skipped ({dim}D)", ha="center", va="center", fontsize=12)
            ax3.axis("off")
        else:
            sigma_vals = sorted(set(r['sigma'] for r in results['ES']))
            for sigma in sigma_vals:
                data = [r for r in results['ES'] if r['sigma'] == sigma]
                lrs = [r['lr'] for r in data]
                maes = [r['mae'] for r in data]
                ax3.plot(lrs, maes, marker='o', label=f'σ={sigma}')
            ax3.set_xlabel('Learning Rate')
            ax3.set_ylabel('MAE')
            ax3.set_title(f'ES: LR vs MAE ({dim}D)')
            ax3.set_xscale('log')
            ax3.legend()
            ax3.grid(True)
        
        # PPO: kl_weight vs KL (aggregated over clip)
        ax4 = fig.add_subplot(gs[1, 0])
        if len(ppo_best) == 0:
            ax4.text(0.5, 0.5, f"PPO skipped ({dim}D)", ha="center", va="center", fontsize=12)
            ax4.axis("off")
        else:
            kl_weights = sorted(set(r['kl_weight'] for r in ppo_best))
            for kl_w in kl_weights:
                data = [r for r in ppo_best if r['kl_weight'] == kl_w]
                lrs = [r['lr'] for r in data]
                kls = [r['kl_div_total'] for r in data]
                ax4.plot(lrs, kls, marker='o', label=f'KL_w={kl_w:.1e}')
            ax4.set_xlabel('Learning Rate')
            ax4.set_ylabel('KL Total')
            ax4.set_title(f'PPO: LR vs KL ({dim}D, best over clip)')
            ax4.set_xscale('log')
            ax4.legend()
            ax4.grid(True)
        
        # PPO: kl_weight vs Correlation (aggregated over clip)
        ax5 = fig.add_subplot(gs[1, 1])
        if len(ppo_best) == 0:
            ax5.text(0.5, 0.5, f"PPO skipped ({dim}D)", ha="center", va="center", fontsize=12)
            ax5.axis("off")
        else:
            kl_weights = sorted(set(r['kl_weight'] for r in ppo_best))
            for kl_w in kl_weights:
                data = [r for r in ppo_best if r['kl_weight'] == kl_w]
                lrs = [r['lr'] for r in data]
                corrs = [r['correlation'] for r in data]
                ax5.plot(lrs, corrs, marker='o', label=f'KL_w={kl_w:.1e}')
            ax5.set_xlabel('Learning Rate')
            ax5.set_ylabel('Correlation')
            ax5.set_title(f'PPO: LR vs Correlation ({dim}D, best over clip)')
            ax5.set_xscale('log')
            ax5.legend()
            ax5.grid(True)
        
        # PPO: clip vs KL
        ax6 = fig.add_subplot(gs[1, 2])
        if len(results['PPO']) == 0:
            ax6.text(0.5, 0.5, f"PPO skipped ({dim}D)", ha="center", va="center", fontsize=12)
            ax6.axis("off")
        else:
            clips = sorted(set(r['ppo_clip'] for r in results['PPO']))
            for clip in clips:
                data = [r for r in results['PPO'] if r['ppo_clip'] == clip]
                lrs = [r['lr'] for r in data]
                kls = [r['kl_div_total'] for r in data]
                ax6.plot(lrs, kls, marker='o', label=f'Clip={clip}')
            ax6.set_xlabel('Learning Rate')
            ax6.set_ylabel('KL Total')
            ax6.set_title(f'PPO: Clip vs KL ({dim}D)')
            ax6.set_xscale('log')
            ax6.legend()
            ax6.grid(True)
        
        # Heatmap: ES sigma vs lr (KL)
        ax7 = fig.add_subplot(gs[2, 0])
        if len(results['ES']) == 0:
            ax7.text(0.5, 0.5, f"No ES results ({dim}D)", ha="center", va="center", fontsize=12)
            ax7.axis("off")
        else:
            sigma_vals = sorted(set(r['sigma'] for r in results['ES']))
            lr_vals = sorted(set(r['lr'] for r in results['ES']))
            if len(sigma_vals) == 0 or len(lr_vals) == 0:
                ax7.text(0.5, 0.5, f"No ES results ({dim}D)", ha="center", va="center", fontsize=12)
                ax7.axis("off")
            else:
                heatmap_data = np.zeros((len(sigma_vals), len(lr_vals)))
                for i, sigma in enumerate(sigma_vals):
                    for j, lr in enumerate(lr_vals):
                        matching = [r for r in results['ES'] if r['sigma'] == sigma and r['lr'] == lr]
                        if matching:
                            heatmap_data[i, j] = matching[0]['kl_div_total']
                        else:
                            heatmap_data[i, j] = np.nan
                im = ax7.imshow(heatmap_data, aspect='auto', cmap='viridis')
                ax7.set_xticks(range(len(lr_vals)))
                ax7.set_yticks(range(len(sigma_vals)))
                ax7.set_xticklabels([f'{lr:.4f}' for lr in lr_vals], rotation=45)
                ax7.set_yticklabels([f'{sigma:.4f}' for sigma in sigma_vals])
                ax7.set_xlabel('Learning Rate')
                ax7.set_ylabel('Sigma')
                ax7.set_title(f'ES: KL Heatmap ({dim}D)')
                plt.colorbar(im, ax=ax7)
        
        # Heatmap: PPO kl_weight vs lr (KL, aggregated over clip)
        ax8 = fig.add_subplot(gs[2, 1])
        if len(ppo_best) == 0:
            ax8.text(0.5, 0.5, f"No PPO results ({dim}D)", ha="center", va="center", fontsize=12)
            ax8.axis("off")
        else:
            kl_weights = sorted(set(r['kl_weight'] for r in ppo_best))
            lr_vals_ppo = sorted(set(r['lr'] for r in ppo_best))
            if len(kl_weights) == 0 or len(lr_vals_ppo) == 0:
                ax8.text(0.5, 0.5, f"No PPO results ({dim}D)", ha="center", va="center", fontsize=12)
                ax8.axis("off")
            else:
                heatmap_data = np.zeros((len(kl_weights), len(lr_vals_ppo)))
                for i, kl_w in enumerate(kl_weights):
                    for j, lr in enumerate(lr_vals_ppo):
                        matching = [r for r in ppo_best if r['kl_weight'] == kl_w and r['lr'] == lr]
                        if matching:
                            heatmap_data[i, j] = matching[0]['kl_div_total']
                        else:
                            heatmap_data[i, j] = np.nan
                im = ax8.imshow(heatmap_data, aspect='auto', cmap='viridis')
                ax8.set_xticks(range(len(lr_vals_ppo)))
                ax8.set_yticks(range(len(kl_weights)))
                ax8.set_xticklabels([f'{lr:.5f}' for lr in lr_vals_ppo], rotation=45)
                ax8.set_yticklabels([f'{kl_w:.1e}' for kl_w in kl_weights])
                ax8.set_xlabel('Learning Rate')
                ax8.set_ylabel('KL Weight')
                ax8.set_title(f'PPO: KL Heatmap ({dim}D, best over clip)')
                plt.colorbar(im, ax=ax8)
        
        # Check which methods exist
        has_es = len(results['ES']) > 0
        has_ppo = len(results['PPO']) > 0
        
        best_es = min(results['ES'], key=self._score_config) if has_es else None
        best_ppo = min(results['PPO'], key=self._score_config) if has_ppo else None
        
        # Helper to safely get values from potentially None best configs
        def safe_get(best, key, default=0.0):
            """Safely get value from best config, returning default if None."""
            return float(best.get(key, default)) if best is not None else float(default)
        
        # Best configs comparison
        ax9 = fig.add_subplot(gs[2, 2])
        metrics = ['kl_div_total', 'correlation', 'mae', 'mutual_information']
        es_vals = [safe_get(best_es, m, default=0.0) for m in metrics]
        ppo_vals = [safe_get(best_ppo, m, default=0.0) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        ax9.bar(x - width/2, es_vals, width, label='Best ES')
        ax9.bar(x + width/2, ppo_vals, width, label='Best PPO')
        ax9.set_xticks(x)
        ax9.set_xticklabels(metrics, rotation=45, ha='right')
        ax9.set_ylabel('Value')
        ax9.set_title(f'Best Configs Comparison ({dim}D)')
        ax9.legend()
        ax9.grid(True, axis='y')
        
        # Row 4: Additional comprehensive metrics
        # Entropy metrics comparison
        ax10 = fig.add_subplot(gs[2, 3])
        entropy_metrics = ['entropy_x1', 'entropy_x2', 'joint_entropy']
        es_entropy = [safe_get(best_es, m, default=0.0) for m in entropy_metrics]
        ppo_entropy = [safe_get(best_ppo, m, default=0.0) for m in entropy_metrics]
        x = np.arange(len(entropy_metrics))
        ax10.bar(x - width/2, es_entropy, width, label='Best ES')
        ax10.bar(x + width/2, ppo_entropy, width, label='Best PPO')
        ax10.set_xticks(x)
        ax10.set_xticklabels(['H(X1)', 'H(X2)', 'H(X,Y)'], rotation=45, ha='right')
        ax10.set_ylabel('Entropy')
        ax10.set_title(f'Entropy Metrics ({dim}D)')
        ax10.legend()
        ax10.grid(True, axis='y')
        
        # Conditional entropy comparison
        ax11 = fig.add_subplot(gs[3, 0])
        cond_entropy_metrics = ['h_x1_given_x2', 'h_x2_given_x1']
        es_cond_entropy = [safe_get(best_es, m, default=0.0) for m in cond_entropy_metrics]
        ppo_cond_entropy = [safe_get(best_ppo, m, default=0.0) for m in cond_entropy_metrics]
        x = np.arange(len(cond_entropy_metrics))
        ax11.bar(x - width/2, es_cond_entropy, width, label='Best ES')
        ax11.bar(x + width/2, ppo_cond_entropy, width, label='Best PPO')
        ax11.set_xticks(x)
        ax11.set_xticklabels(['H(X1|X2)', 'H(X2|X1)'], rotation=45, ha='right')
        ax11.set_ylabel('Conditional Entropy')
        ax11.set_title(f'Conditional Entropies ({dim}D)')
        ax11.legend()
        ax11.grid(True, axis='y')
        
        # Directional MI comparison
        ax12 = fig.add_subplot(gs[3, 1])
        mi_metrics = ['mi_x2_to_x1', 'mi_x1_to_x2']
        es_mi = [safe_get(best_es, m, default=0.0) for m in mi_metrics]
        ppo_mi = [safe_get(best_ppo, m, default=0.0) for m in mi_metrics]
        x = np.arange(len(mi_metrics))
        ax12.bar(x - width/2, es_mi, width, label='Best ES')
        ax12.bar(x + width/2, ppo_mi, width, label='Best PPO')
        ax12.set_xticks(x)
        ax12.set_xticklabels(['I(X2;X1)', 'I(X1;X2)'], rotation=45, ha='right')
        ax12.set_ylabel('MI')
        ax12.set_title(f'Directional MI ({dim}D)')
        ax12.legend()
        ax12.grid(True, axis='y')
        
        # Directional MAE comparison
        ax13 = fig.add_subplot(gs[3, 2])
        mae_metrics = ['mae_x2_to_x1', 'mae_x1_to_x2']
        es_mae_dir = [safe_get(best_es, m, default=0.0) for m in mae_metrics]
        ppo_mae_dir = [safe_get(best_ppo, m, default=0.0) for m in mae_metrics]
        x = np.arange(len(mae_metrics))
        ax13.bar(x - width/2, es_mae_dir, width, label='Best ES')
        ax13.bar(x + width/2, ppo_mae_dir, width, label='Best PPO')
        ax13.set_xticks(x)
        ax13.set_xticklabels(['MAE(X2→X1)', 'MAE(X1→X2)'], rotation=45, ha='right')
        ax13.set_ylabel('MAE')
        ax13.set_title(f'Directional MAE ({dim}D)')
        ax13.legend()
        ax13.grid(True, axis='y')
        
        # Individual KL components
        ax14 = fig.add_subplot(gs[3, 3])
        kl_metrics = ['kl_div_1', 'kl_div_2']
        es_kl_comp = [safe_get(best_es, m, default=0.0) for m in kl_metrics]
        ppo_kl_comp = [safe_get(best_ppo, m, default=0.0) for m in kl_metrics]
        x = np.arange(len(kl_metrics))
        ax14.bar(x - width/2, es_kl_comp, width, label='Best ES')
        ax14.bar(x + width/2, ppo_kl_comp, width, label='Best PPO')
        ax14.set_xticks(x)
        ax14.set_xticklabels(['KL(X1)', 'KL(X2)'], rotation=45, ha='right')
        ax14.set_ylabel('KL Divergence')
        ax14.set_title(f'Individual KL Components ({dim}D)')
        ax14.legend()
        ax14.grid(True, axis='y')
        
        # Learned statistics - means (use row 4 to avoid overlap with row 3)
        ax15 = fig.add_subplot(gs[4, :])
        ax15.axis('off')  # Hide axes for text display
        ax15.text(0.5, 0.9, 'Best Configuration Learned Statistics', ha='center', fontsize=12, fontweight='bold', transform=ax15.transAxes)
        
        if has_es and has_ppo:
            stats_text = f"""
ES Best Config (σ={best_es['sigma']}, lr={best_es['lr']}):
  μ1: {best_es['mu1_learned']:.3f} (target: 2.0)    μ2: {best_es['mu2_learned']:.3f} (target: 10.0)
  σ1: {best_es['std1_learned']:.3f} (target: 1.0)    σ2: {best_es['std2_learned']:.3f} (target: 1.0)

PPO Best Config (kl_w={best_ppo['kl_weight']:.1e}, clip={best_ppo['ppo_clip']}, lr={best_ppo['lr']}):
  μ1: {best_ppo['mu1_learned']:.3f} (target: 2.0)    μ2: {best_ppo['mu2_learned']:.3f} (target: 10.0)
  σ1: {best_ppo['std1_learned']:.3f} (target: 1.0)    σ2: {best_ppo['std2_learned']:.3f} (target: 1.0)
            """
        elif has_es:
            stats_text = f"""
ES Best Config (σ={best_es['sigma']}, lr={best_es['lr']}):
  μ1: {best_es['mu1_learned']:.3f} (target: 2.0)    μ2: {best_es['mu2_learned']:.3f} (target: 10.0)
  σ1: {best_es['std1_learned']:.3f} (target: 1.0)    σ2: {best_es['std2_learned']:.3f} (target: 1.0)

PPO: (skipped --only-method ES)
            """
        elif has_ppo:
            stats_text = f"""
ES: (skipped --only-method PPO)

PPO Best Config (kl_w={best_ppo['kl_weight']:.1e}, clip={best_ppo['ppo_clip']}, lr={best_ppo['lr']}):
  μ1: {best_ppo['mu1_learned']:.3f} (target: 2.0)    μ2: {best_ppo['mu2_learned']:.3f} (target: 10.0)
  σ1: {best_ppo['std1_learned']:.3f} (target: 1.0)    σ2: {best_ppo['std2_learned']:.3f} (target: 1.0)
            """
        else:
            stats_text = "\nNo methods run (both skipped)\n"
        ax15.text(0.1, 0.4, stats_text, fontsize=10, family='monospace',
                 verticalalignment='center', transform=ax15.transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        ax15.axis('off')
        
        plt.suptitle(f'{dim}D Comprehensive Ablation Study Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(self.plots_dir, f'ablation_{dim}d.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Plots saved: {plot_path}")
        
        # Log to wandb
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.log({f'{dim}D/ablation_plots': wandb.Image(plot_path)})
    
    def _generate_overall_summary(self):
        """Generate overall summary across all dimensions."""
        summary_path = os.path.join(self.output_dir, "OVERALL_SUMMARY.txt")
        
        with open(summary_path, 'w') as f:
            f.write("="*100 + "\n")
            f.write("COMPREHENSIVE ABLATION STUDY: OVERALL SUMMARY\n")
            f.write("="*100 + "\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output Directory: {self.output_dir}\n")
            f.write(f"Dimensions: {self.config.dimensions}\n")
            f.write("="*100 + "\n\n")
            
            # Best configurations per dimension
            f.write("BEST CONFIGURATIONS PER DIMENSION:\n")
            f.write("-"*100 + "\n\n")
            
            for dim in sorted(self.all_results.keys()):
                results = self.all_results[dim]
                has_es = len(results['ES']) > 0
                has_ppo = len(results['PPO']) > 0
                
                best_es = min(results['ES'], key=self._score_config) if has_es else None
                best_ppo = min(results['PPO'], key=self._score_config) if has_ppo else None
                
                f.write(f"{dim}D:\n")
                if has_es:
                    f.write(f"  ES  - sigma={best_es['sigma']:<6.4f} lr={best_es['lr']:<8.6f} "
                           f"KL={best_es['kl_div_total']:<8.4f} Corr={best_es['correlation']:<6.4f} MAE={best_es['mae']:<6.4f}\n")
                else:
                    f.write(f"  ES  - (skipped --only-method PPO)\n")
                if has_ppo:
                    f.write(f"  PPO - kl_w={best_ppo['kl_weight']:<6.1e} clip={best_ppo['ppo_clip']:<6.2f} "
                           f"lr={best_ppo['lr']:<8.6f} KL={best_ppo['kl_div_total']:<8.4f} Corr={best_ppo['correlation']:<6.4f} MAE={best_ppo['mae']:<6.4f}\n")
                else:
                    f.write(f"  PPO - (skipped --only-method ES)\n")
                if has_es and has_ppo:
                    f.write(f"  WINNER: {'ES' if self._score_config(best_es) < self._score_config(best_ppo) else 'PPO'}\n\n")
                elif has_es:
                    f.write(f"  WINNER: ES (only method run)\n\n")
                elif has_ppo:
                    f.write(f"  WINNER: PPO (only method run)\n\n")
                else:
                    f.write(f"  WINNER: (no methods run)\n\n")
            
            # Overall statistics
            f.write("\n" + "="*100 + "\n")
            f.write("OVERALL STATISTICS:\n")
            f.write("="*100 + "\n\n")
            
            es_wins = 0
            ppo_wins = 0
            
            for dim in self.all_results.keys():
                results = self.all_results[dim]
                has_es = len(results['ES']) > 0
                has_ppo = len(results['PPO']) > 0
                
                if has_es and has_ppo:
                    best_es = min(results['ES'], key=self._score_config)
                    best_ppo = min(results['PPO'], key=self._score_config)
                    if self._score_config(best_es) < self._score_config(best_ppo):
                        es_wins += 1
                    else:
                        ppo_wins += 1
                elif has_es:
                    es_wins += 1
                elif has_ppo:
                    ppo_wins += 1
            
            f.write(f"ES Wins: {es_wins}/{len(self.all_results)} dimensions\n")
            f.write(f"PPO Wins: {ppo_wins}/{len(self.all_results)} dimensions\n\n")
            
            f.write(f"OVERALL WINNER: {'ES' if es_wins > ppo_wins else 'PPO'}\n")
            
            f.write("\n" + "="*100 + "\n")
            f.write("END OF SUMMARY\n")
            f.write("="*100 + "\n")
        
        print(f"Overall summary saved: {summary_path}")
        
        # Save results to JSON
        json_path = os.path.join(self.output_dir, "all_results.json")
        
        # Convert results to JSON-serializable format
        json_results = {}
        for dim, results in self.all_results.items():
            json_results[str(dim)] = {
                'ES': [{k: v for k, v in r.items() if k != 'history'} for r in results['ES']],
                'PPO': [{k: v for k, v in r.items() if k != 'history'} for r in results['PPO']]
            }
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results JSON saved: {json_path}")
        
        # Generate overall comparison plots
        self._plot_overall_comparison()
    
    def _plot_overall_comparison(self):
        """Generate overall comparison plots across dimensions (improved version with CSV export)."""
        import csv
        
        dims = sorted(self.all_results.keys())
        
        # Helper: extract best config per dimension
        def _safe_min_by_score(rows):
            if not rows:
                return None
            return min(rows, key=self._score_config)
        
        best_es = {}
        best_ppo = {}
        for dim in dims:
            r = self.all_results[dim]
            best_es[dim] = _safe_min_by_score(r.get("ES", []))
            best_ppo[dim] = _safe_min_by_score(r.get("PPO", []))
        
        # Helper: extract series for plotting
        def _as_float(x, default=np.nan):
            try:
                return float(x) if x is not None else default
            except Exception:
                return default
        
        def series(best_map, key):
            xs, ys = [], []
            for d in dims:
                b = best_map.get(d)
                if b is None:
                    continue
                val = _as_float(b.get(key), np.nan)
                if np.isfinite(val):
                    xs.append(d)
                    ys.append(val)
            return xs, ys
        
        # Helper: determine winner per dimension
        def _winner_by_dim(dim):
            es = best_es.get(dim)
            ppo = best_ppo.get(dim)
            if es is None and ppo is None:
                return "NONE"
            if es is None:
                return "PPO"
            if ppo is None:
                return "ES"
            return "ES" if self._score_config(es) < self._score_config(ppo) else "PPO"
        
        # Helper: config string for CSV
        def _config_str(method, best):
            if best is None:
                return "(none)"
            if method == "ES":
                return f"σ={best.get('sigma')} lr={best.get('lr')}"
            return f"kl_w={best.get('kl_weight')} clip={best.get('ppo_clip')} lr={best.get('lr')}"
        
        # Save CSV of best configs per dimension
        overall_csv = os.path.join(self.output_dir, "overall_best_configs.csv")
        fieldnames = [
            "dim", "winner",
            "es_score", "es_kl", "es_corr", "es_mae", "es_mi", "es_mi_per_dim", "es_cfg",
            "ppo_score", "ppo_kl", "ppo_corr", "ppo_mae", "ppo_mi", "ppo_mi_per_dim", "ppo_cfg",
        ]
        with open(overall_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for dim in dims:
                es = best_es.get(dim)
                ppo = best_ppo.get(dim)
                winner = _winner_by_dim(dim)
                
                es_score = _as_float(self._score_config(es), np.nan) if es is not None else np.nan
                ppo_score = _as_float(self._score_config(ppo), np.nan) if ppo is not None else np.nan
                
                row = {
                    "dim": dim,
                    "winner": winner,
                    "es_score": es_score,
                    "es_kl": _as_float(es.get("kl_div_total"), np.nan) if es else np.nan,
                    "es_corr": _as_float(es.get("correlation"), np.nan) if es else np.nan,
                    "es_mae": _as_float(es.get("mae"), np.nan) if es else np.nan,
                    "es_mi": _as_float(es.get("mutual_information"), np.nan) if es else np.nan,
                    "es_mi_per_dim": _as_float(es.get("mutual_information_per_dim"), np.nan) if es else np.nan,
                    "es_cfg": _config_str("ES", es),
                    "ppo_score": ppo_score,
                    "ppo_kl": _as_float(ppo.get("kl_div_total"), np.nan) if ppo else np.nan,
                    "ppo_corr": _as_float(ppo.get("correlation"), np.nan) if ppo else np.nan,
                    "ppo_mae": _as_float(ppo.get("mae"), np.nan) if ppo else np.nan,
                    "ppo_mi": _as_float(ppo.get("mutual_information"), np.nan) if ppo else np.nan,
                    "ppo_mi_per_dim": _as_float(ppo.get("mutual_information_per_dim"), np.nan) if ppo else np.nan,
                    "ppo_cfg": _config_str("PPO", ppo),
                }
                w.writerow(row)
        
        print(f"Overall best-config CSV saved: {overall_csv}")
        
        # Extract series for plotting
        es_x_kl, es_y_kl = series(best_es, "kl_div_total")
        ppo_x_kl, ppo_y_kl = series(best_ppo, "kl_div_total")
        es_x_mae, es_y_mae = series(best_es, "mae")
        ppo_x_mae, ppo_y_mae = series(best_ppo, "mae")
        es_x_corr, es_y_corr = series(best_es, "correlation")
        ppo_x_corr, ppo_y_corr = series(best_ppo, "correlation")
        es_x_mipd, es_y_mipd = series(best_es, "mutual_information_per_dim")
        ppo_x_mipd, ppo_y_mipd = series(best_ppo, "mutual_information_per_dim")
        
        # Winner counts
        winners = [_winner_by_dim(d) for d in dims]
        es_wins = sum(1 for w in winners if w == "ES")
        ppo_wins = sum(1 for w in winners if w == "PPO")
        none_wins = sum(1 for w in winners if w == "NONE")
        
        # Create plots
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 3, figure=fig)
        
        # (1) Best KL vs dim
        ax1 = fig.add_subplot(gs[0, 0])
        if es_x_kl:
            ax1.plot(es_x_kl, es_y_kl, marker="o", linewidth=2, label="ES")
        if ppo_x_kl:
            ax1.plot(ppo_x_kl, ppo_y_kl, marker="s", linewidth=2, label="PPO")
        ax1.set_title("Best KL (per-dim) vs Dimension")
        ax1.set_xlabel("Dimension")
        ax1.set_ylabel("KL (per-dim)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # (2) Best MAE vs dim
        ax2 = fig.add_subplot(gs[0, 1])
        if es_x_mae:
            ax2.plot(es_x_mae, es_y_mae, marker="o", linewidth=2, label="ES")
        if ppo_x_mae:
            ax2.plot(ppo_x_mae, ppo_y_mae, marker="s", linewidth=2, label="PPO")
        ax2.set_title("Best MAE vs Dimension")
        ax2.set_xlabel("Dimension")
        ax2.set_ylabel("MAE")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # (3) Best Corr vs dim
        ax3 = fig.add_subplot(gs[0, 2])
        if es_x_corr:
            ax3.plot(es_x_corr, es_y_corr, marker="o", linewidth=2, label="ES")
        if ppo_x_corr:
            ax3.plot(ppo_x_corr, ppo_y_corr, marker="s", linewidth=2, label="PPO")
        ax3.set_title("Best Correlation vs Dimension")
        ax3.set_xlabel("Dimension")
        ax3.set_ylabel("Correlation")
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # (4) Best MI/dim vs dim (using per-dim MI for cross-dimension comparison)
        ax4 = fig.add_subplot(gs[1, 0])
        if es_x_mipd:
            ax4.plot(es_x_mipd, es_y_mipd, marker="o", linewidth=2, label="ES")
        if ppo_x_mipd:
            ax4.plot(ppo_x_mipd, ppo_y_mipd, marker="s", linewidth=2, label="PPO")
        ax4.set_title("Best Mutual Information per Dim vs Dimension")
        ax4.set_xlabel("Dimension")
        ax4.set_ylabel("MI / dim")
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # (5) Winner per dimension (categorical bar)
        ax5 = fig.add_subplot(gs[1, 1])
        w_map = {"ES": 1, "PPO": -1, "NONE": 0}
        y = [w_map[w] for w in winners]
        ax5.bar([str(d) for d in dims], y, color=['#1f77b4' if w == "ES" else '#ff7f0e' if w == "PPO" else 'gray' for w in winners])
        ax5.set_title("Winner per Dimension (ES=+1, PPO=-1)")
        ax5.set_xlabel("Dimension")
        ax5.set_ylabel("Winner")
        ax5.axhline(0, linewidth=1, color='black')
        ax5.grid(True, axis="y", alpha=0.3)
        
        # (6) Scatter: KL vs MAE for best configs
        ax6 = fig.add_subplot(gs[1, 2])
        for d in dims:
            es = best_es.get(d)
            if es is not None:
                kl_val = _as_float(es.get("kl_div_total"), np.nan)
                mae_val = _as_float(es.get("mae"), np.nan)
                if np.isfinite(kl_val) and np.isfinite(mae_val):
                    ax6.scatter(kl_val, mae_val, marker="o", s=100, alpha=0.6)
                    ax6.text(kl_val, mae_val, f"ES {d}", fontsize=8)
            ppo = best_ppo.get(d)
            if ppo is not None:
                kl_val = _as_float(ppo.get("kl_div_total"), np.nan)
                mae_val = _as_float(ppo.get("mae"), np.nan)
                if np.isfinite(kl_val) and np.isfinite(mae_val):
                    ax6.scatter(kl_val, mae_val, marker="x", s=100, alpha=0.6)
                    ax6.text(kl_val, mae_val, f"PPO {d}", fontsize=8)
        ax6.set_title("Best Configs: KL vs MAE")
        ax6.set_xlabel("KL (per-dim)")
        ax6.set_ylabel("MAE")
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f"Overall ES vs PPO Comparison | ES wins={es_wins}, PPO wins={ppo_wins}, none={none_wins}", 
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        plot_path = os.path.join(self.plots_dir, "overall_comparison.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"Overall comparison plot saved: {plot_path}")
        
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.log({"overall/overall_comparison": wandb.Image(plot_path)})


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Ablation Study: ES vs PPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dimensions
    parser.add_argument("--dimensions", type=int, nargs='+', default=[1, 2, 5, 10, 20, 30],
                       help="Dimensions to test")
    
    # Training parameters
    parser.add_argument("--coupling-epochs", type=int, default=14,
                       help="Number of epochs for coupling training (match config default)")
    parser.add_argument("--ddpm-epochs", type=int, default=200,
                       help="Number of epochs for DDPM pretraining")
    
    # ES ablation ranges (match AblationConfig defaults)
    parser.add_argument("--es-sigma-values", type=float, nargs='+', 
                       default=[0.001, 0.002, 0.005, 0.01],
                       help="ES sigma values to test")
    parser.add_argument("--es-lr-values", type=float, nargs='+',
                       default=[0.0001, 0.0002, 0.0005, 0.001],
                       help="ES learning rate values to test")
    
    # PPO ablation ranges (match AblationConfig defaults - CRITICAL: was [0.1,0.3,0.5,0.7])
    parser.add_argument("--ppo-kl-values", type=float, nargs='+',
                       default=[1e-4, 3e-4, 1e-3, 3e-3],
                       help="PPO KL weight values to test")
    parser.add_argument("--ppo-clip-values", type=float, nargs='+',
                       default=[0.02, 0.05, 0.1],
                       help="PPO clip values to test")
    parser.add_argument("--ppo-lr-values", type=float, nargs='+',
                       default=[1e-5, 2e-5, 5e-5, 1e-4],
                       help="PPO learning rate values to test")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="ablation_results",
                       help="Output directory")
    # WandB: mutually exclusive group for clarity
    wandb_group = parser.add_mutually_exclusive_group()
    wandb_group.add_argument("--use-wandb", dest="use_wandb", action="store_true",
                            help="Enable WandB logging")
    wandb_group.add_argument("--no-wandb", dest="use_wandb", action="store_false",
                            help="Disable WandB logging")
    parser.set_defaults(use_wandb=None)  # None means use config default
    parser.add_argument("--wandb-project", type=str, default="ddmec-ablation",
                       help="WandB project name")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Model management
    parser.add_argument("--retrain-ddpm", action="store_true",
                       help="Retrain DDPM models from scratch (ignore existing pretrained models)")
    parser.add_argument("--reuse-pretrained", dest='reuse_pretrained', action="store_true", default=True,
                       help="Reuse existing pretrained DDPM models if available (default)")
    parser.add_argument("--no-reuse-pretrained", dest='reuse_pretrained', action="store_false",
                       help="Same as --retrain-ddpm (train from scratch)")
    
    # Subsetting controls (for job arrays / incremental ablations)
    parser.add_argument("--only-dim", type=int, default=None,
                       help="Run only a single dimension (for job arrays)")
    parser.add_argument("--only-method", type=str, choices=["ES", "PPO", "BOTH"], default="BOTH",
                       help="Run only ES, only PPO, or both (default: BOTH)")
    parser.add_argument("--max-es-configs", type=int, default=None,
                       help="Limit number of ES configs to test (for quick runs)")
    parser.add_argument("--max-ppo-configs", type=int, default=None,
                       help="Limit number of PPO configs to test (for quick runs)")
    parser.add_argument("--es-config-idx", type=int, default=None,
                       help="Run only a single ES config by index (for job arrays, overrides --max-es-configs)")
    parser.add_argument("--ppo-config-idx", type=int, default=None,
                       help="Run only a single PPO config by index (for job arrays, overrides --max-ppo-configs)")
    
    args = parser.parse_args()
    
    # Handle retrain-ddpm flag (alias for no-reuse-pretrained)
    if args.retrain_ddpm:
        args.reuse_pretrained = False
    
    # Create config
    # Handle WandB default: use config default if argparse didn't set it
    # CRITICAL: AblationConfig.use_wandb is a Field object, need to instantiate to get actual value
    default_use_wandb = AblationConfig().use_wandb
    use_wandb = args.use_wandb if args.use_wandb is not None else default_use_wandb
    
    config = AblationConfig(
        dimensions=args.dimensions,
        coupling_epochs=args.coupling_epochs,
        ddpm_epochs=args.ddpm_epochs,
        es_sigma_values=args.es_sigma_values,
        es_lr_values=args.es_lr_values,
        ppo_kl_weight_values=args.ppo_kl_values,
        ppo_clip_values=args.ppo_clip_values,
        ppo_lr_values=args.ppo_lr_values,
        output_dir=args.output_dir,
        use_wandb=use_wandb,
        wandb_project=args.wandb_project,
        seed=args.seed,
        reuse_pretrained=args.reuse_pretrained,
    )
    
    # Set subsetting controls and retrain flag (now part of AblationConfig)
    config.only_dim = args.only_dim
    config.only_method = args.only_method if args.only_method != "BOTH" else None
    config.max_es_configs = args.max_es_configs
    config.max_ppo_configs = args.max_ppo_configs
    config.es_config_idx = args.es_config_idx
    config.ppo_config_idx = args.ppo_config_idx
    config.retrain_ddpm = args.retrain_ddpm
    
    # Run ablation study
    runner = AblationRunner(config)
    runner.run()


if __name__ == "__main__":
    main()

