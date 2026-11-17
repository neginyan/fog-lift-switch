# demo_synth.py -- a small synthetic "network" for checking the demo's live part on CPU (MIT License, (c) 2026 Taishi Namba)
# The data are a mixture of Gaussians; the "network" is its exact denoiser, so it disagrees with the Gaussian
# denoiser D_G once the fog lifts. This only checks that the demo's pipeline runs; it says nothing about EDM.
import numpy as np


def make(d=48, k=6, c=0.02, spread=1.0, seed=0, err=0.01, freq=1.0):
    rng = np.random.default_rng(seed)
    R = np.random.default_rng(seed + 1).standard_normal((d, d))
    means = rng.standard_normal((k, d)) * spread / np.sqrt(d) * 4
    pis = np.full(k, 1.0 / k)
    mu = means.mean(0)
    cov = c * np.eye(d) + (means - mu).T @ (means - mu) / k

    def net_fn(x, sigma):
        v = c + float(sigma) ** 2
        d2 = ((x[:, None, :] - means[None]) ** 2).sum(-1)
        logw = np.log(pis)[None] - 0.5 * d2 / v
        w = np.exp(logw - logw.max(1, keepdims=True))
        w /= w.sum(1, keepdims=True)
        post = means[None] + (c / v) * (x[:, None, :] - means[None])
        # a small, deterministic network error of fixed size, so that u has a floor in the fog (as for real networks)
        return (w[:, :, None] * post).sum(1) + err * np.sqrt(2) * np.sin(freq * x @ R + float(sigma))

    return mu, cov, net_fn
