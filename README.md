# Evolution Strategies vs PPO for Coupled Diffusion Models

> **Comprehensive Ablation Study**: Systematic comparison of gradient-free (ES) vs gradient-based (PPO) optimization for training conditional diffusion models across dimensions 1D-30D.

## 🎯 Key Findings

- **ES wins in low dimensions (1D-10D)** with 4/6 overall victories
- **PPO dominates high dimensions (20D-30D)** where ES catastrophically fails  
- **Critical transition at ~15D** marks the performance crossover point
- **ES KL divergence explodes** from 0.07 (10D) to 1,152,910 (30D)
- **PPO maintains trainability** even in 30D with proper regularization

## 📊 Results Summary

| Dimension | Best Method | ES KL | PPO KL | ES Correlation | PPO Correlation |
|-----------|-------------|-------|--------|----------------|-----------------|
| 1D        | ES          | 0.0002| 0.0002 | 0.9813         | 0.9953          |
| 2D        | ES          | 0.0008| 0.0017 | 0.9896         | 0.9842          |
| 5D        | ES          | 0.0133| 0.0364 | 0.9841         | 0.9838          |
| 10D       | ES          | 0.0704| 0.1125 | 0.9533         | 0.9678          |
| 20D       | PPO         | 42.78 | 5.57   | 0.6617         | 0.7898          |
| 30D       | PPO         | 1,152,910 | 142.11 | 0.4206    | 0.5619          |

## 📖 Documentation

- **[📄 Full Technical Blog Post](ablation_results/run_20251211_215609/TECHNICAL_BLOG_POST.md)** - Complete analysis with equations, plots, and discussion
- **[📊 Overall Summary](ablation_results/run_20251211_215609/OVERALL_SUMMARY.txt)** - Quick results overview
- **[📈 All Results (JSON)](ablation_results/run_20251211_215609/all_results.json)** - Complete numerical data

## 🖼️ Visualizations

All plots available in `ablation_results/run_20251211_215609/plots/`:
- Dimension-specific ablation plots (1D-30D)
- Training curves for all configurations
- Overall comparison across dimensions

## 🔬 Experimental Setup

- **Dimensions tested**: 1, 2, 5, 10, 20, 30
- **ES configurations**: 16 (4 σ × 4 learning rates)
- **PPO configurations**: 64 (4 KL weights × 4 clip params × 4 learning rates)
- **Total experiments**: 480 configurations
- **Evaluation metrics**: KL divergence, mutual information, correlation, MAE, conditional entropy

## 🚀 Quick Start

### Reproduce Results

```bash
# Clone repository
git clone https://github.com/adnankarim/es-vs-ppo-diffusion.git
cd es-vs-ppo-diffusion

# Install dependencies
pip install torch numpy matplotlib seaborn wandb

# Run ablation study
python run_ablation_study.py --dimensions 1 2 5 10 20 30
```

### View Results

- **GitHub Pages**: https://adnankarim.github.io/es-vs-ppo-diffusion/
- **Local**: Open `ablation_results/run_20251211_215609/TECHNICAL_BLOG_POST.md` in any markdown viewer

## 📁 Project Structure

```
.
├── run_ablation_study.py              # Main experiment script
├── ablation_results/
│   └── run_20251211_215609/          # Experiment results
│       ├── TECHNICAL_BLOG_POST.md    # Full technical analysis
│       ├── all_results.json          # Complete numerical results
│       ├── OVERALL_SUMMARY.txt       # Summary statistics
│       ├── plots/                    # All visualizations
│       └── logs/                     # Dimension-wise summaries
├── README.md                          # This file
└── _config.yml                        # GitHub Pages config
```

## 🔑 Key Insights

1. **Dimension-dependent performance**: ES excels in low dimensions but fails catastrophically beyond 15D
2. **Hyperparameter trends**: ES needs lower exploration noise (σ) as dimension grows; PPO needs higher KL regularization
3. **Information-theoretic analysis**: Mutual information preservation is the critical challenge in high dimensions
4. **Warmup is essential**: Both methods require gradient-based warmup before method-specific fine-tuning

## 📚 Citation

If you use this work, please cite:

```bibtex
@misc{es_vs_ppo_diffusion_2024,
  title={Evolution Strategies vs PPO for Coupled Diffusion Models: A Comprehensive Ablation Study},
  author={Research Team},
  year={2024},
  howpublished={\url{https://github.com/adnankarim/es-vs-ppo-diffusion}}
}
```

## 📝 License

This project is open source and available for research purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

---

**Experiment ID**: `run_20251211_215609`  
**Date**: December 13, 2024  
**Total Runtime**: ~18 hours on CUDA GPU

