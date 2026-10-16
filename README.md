# Flow Matching

A learning-by-doing exploration of flow matching, progressing from 2D toy problems through MNIST experiments. Next steps: scale up to larger datasets (CIFAR) and fine-tune or run inference with pre-trained models.

## What is flow matching?

Flow matching is a simulation-free generative modelling framework. A velocity field $v_\theta(x_t, t)$ is trained to transport samples from a simple source $p_0$ (e.g. Gaussian) to the data distribution $p_1$. At training time, pairs $(x_0, x_1)$ define a straight-path interpolant $x_t = (1-t)x_0 + tx_1$, and the model is regressed onto the conditional vector field $\dot{x}_t = x_1 - x_0$. Sampling is just ODE integration from $t=0$ to $t=1$.

## Notebooks

### 2D
| Notebook | Description |
|---|---|
| [simple_2d.ipynb](notebooks/simple_2d.ipynb) | Flow matching on a 2D moons dataset. Covers the core training and sampling loop. |

### MNIST

| Notebook | Description |
|---|---|
| [mnist_unconditioned_fm.ipynb](notebooks/mnist/mnist_unconditioned_fm.ipynb) | Baseline unconditioned flow matching on MNIST. |
| [mnist_noise_variance.ipynb](notebooks/mnist/mnist_noise_variance.ipynb) | Effect of source noise variance σ on training loss and sample quality. |
| [mnist_modes.ipynb](notebooks/mnist/mnist_modes.ipynb) | Using multimodal source distributions to identify modes. |
| [mnist_ot_coupling.ipynb](notebooks/mnist/mnist_ot_coupling.ipynb) | Optimal transport coupling to reduce path length and improve training. |
| [mnist_cond_base.ipynb](notebooks/mnist/mnist_cond_base.ipynb) | Conditioning by injecting class information into the source distribution. |
| [mnist_cond_embeddings.ipynb](notebooks/mnist/mnist_cond_embeddings.ipynb) | Conditional flow matching with learned class embeddings. |
| [mnist_cond_image_embeddings.ipynb](notebooks/mnist/mnist_cond_image_embeddings.ipynb) | Conditioning on image-domain embeddings. |
| [mnist_cond_cfg.ipynb](notebooks/mnist/mnist_cond_cfg.ipynb) | Classifier-free guidance (CFG) for conditional generation. |

## Setup

Dependencies are declared in `pyproject.toml`. Requires Python 3.12+.

**pip**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

**uv**
```bash
uv sync
```

## Next steps

- **Scale up** — train on CIFAR-10 or larger datasets; fine-tune or run inference with a pre-trained model.
- **Discrete flow matching** — extend the framework to discrete state spaces (text tokens, graphs) using MDLM or similar formulations.
