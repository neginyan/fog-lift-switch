# make_demo_notebook.py -- writes notebooks/demo_colab.ipynb (MIT License, (c) 2026 Taishi Namba)
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

INTRO = r"""# When does the fog lift? — a few-minute demo

This notebook accompanies the paper **"When does the fog lift? A switching rule fixed on synthetic worlds, tested by pre-registration on pretrained image diffusion models"** (Taishi Namba) and the repository [neginyan/fog-lift-switch](https://github.com/neginyan/fog-lift-switch).

A diffusion sampler starts in a *fog* of noise where a closed-form Gaussian denoiser $D_G$ (built from the data mean and covariance) is almost exact. The signal $u=\mathrm{median}\,|D_G-D_\mathrm{net}|^2/d$, measured on 5% of the images, is low in the fog and rises when the fog lifts. Two rules read it while sampling: **U_min** (switch when $u$ exceeds 2.38 × its running minimum) and **U_env** (the rise must persist over a short window). Before the switch the sampler follows $D_G$; after it, the network.

**Part 1 — replay the records (CPU, seconds; no GPU needed).** Draws the recorded signal curves of the paper's runs (CIFAR-10 stage C2, AFHQv2 stages C3 and C6), re-applies both rules to every recorded trace and checks that they switch exactly where they switched on the GPU, and recomputes all 25 pre-registered decisions (HC1–HC25) from the records, to compare with Table 1 of the paper.

**Part 2 — a small live run (free Colab T4 GPU, a few minutes).** Loads the official EDM CIFAR-10 VP network, builds the Gaussian denoiser from the CIFAR-10 training images on the spot, and samples 256 images with 18 Heun steps three ways from the same starting noise: the network alone (M0), U_min and U_env. It shows the signal rising, where each rule switched, the images side by side, and the fraction of network evaluations saved. The CIFAR-10 training images are downloaded from Hugging Face (`uoft-cs/cifar10`), and only if that fails from torchvision's source at the University of Toronto. The notebook prints which source it used, the measured times, and an exact fingerprint of the images, which it compares with the one measured when this demo was prepared.

**Time.** Part 1 takes seconds on a CPU; Part 2 takes a few minutes on a Colab GPU (downloads included). If Hugging Face is unavailable, the fallback download from Toronto can take 20 minutes or more. A warning about unauthenticated requests to the Hugging Face Hub (HF_TOKEN) may appear; it can be ignored.

**What this demo does not do.** It computes **no FID**: 256 images cannot judge image quality. Quality was judged in the paper on 10,000–15,000 images per method under pre-registered criteria (Table 1; Part 1 recomputes those decisions). On CIFAR-10 only U_min was pre-registered (stage C2: HC7 supported, HC8 not supported on cost); U_env is shown here for illustration.

**Licences.** Our code: MIT. The EDM code and the pretrained network (NVIDIA) are under CC BY-NC-SA 4.0 and are downloaded from their official locations, not from our repository. CIFAR-10 is downloaded from Hugging Face (`uoft-cs/cifar10`), or, if that fails, by `torchvision` from the University of Toronto, under its own terms. The rule and denoiser code (`c1core.py`–`c3core.py`) is used exactly as in the paper's runs."""

UNZIP = r"""# Only before the repository is public: upload the fog-lift-switch_part0*.zip files to /content (Files pane),
# then remove the '#' at the start of the next line and run this cell. Afterwards the setup cell uses /content/fog-lift-switch.
# !cd /content && for f in fog-lift-switch_part0*.zip; do unzip -q -o "$f"; done"""

SETUP = r"""# Setup: use this repository's code unchanged (clone it on Colab; use the local copy when run from the repository)
import os, sys, subprocess
REPO = next((os.path.abspath(p) for p in ('.', '..', '/content/fog-lift-switch') if os.path.exists(os.path.join(p, 'c1core.py'))), None)
if REPO is None:
    subprocess.run(['git', 'clone', '-q', 'https://github.com/neginyan/fog-lift-switch.git', '/content/fog-lift-switch'], check=True)
    REPO = '/content/fog-lift-switch'
sys.path.insert(0, REPO)
import numpy as np
import matplotlib.pyplot as plt
import demo_core as DC, c1core as C1, c3core as C3
print('repository:', REPO, '| rule code version:', C3.C3_VERSION)"""

P1_MD = "## Part 1 — replay the records (CPU)"

P1_REPLAY = r"""# Re-apply both rules to every recorded signal trace and compare with the recorded switches
rows = DC.replay_all()
n_ok = sum(r['match'] for r in rows)
print('recorded runs replayed: %d   switches reproduced exactly: %d' % (len(rows), n_ok))
for key in [('CIFAR-10', 'vp'), ('CIFAR-10', 've'), ('AFHQv2', 'vp'), ('AFHQv2', 've')]:
    for rule in ('U_min', 'U_env'):
        sel = [r for r in rows if (r['data'], r['net']) == key and r['rule'] == rule]
        if sel:
            sig = sorted({round(r['replayed_sigma'], 2) for r in sel}, reverse=True)
            print('  %-8s %s  %-5s  %2d batches, switch sigma in %s' % (key[0], key[1].upper(), rule, len(sel), sig))
assert n_ok == len(rows), 'a replayed switch differs from the record'
for fig in DC.plot_records(plt):
    plt.show()"""

P1_HC = r"""# Recompute the 25 pre-registered decisions from the records and compare with Table 1 of the paper
decisions, new_only = DC.recompute_decisions()
print('%-5s  %-14s  %s' % ('HC', 'recomputed', 'paper Table 1'))
for h, v in decisions.items():
    print('%-5s  %-14s  %s' % (h, v, DC.PAPER_TABLE1[h]))
same = decisions == DC.PAPER_TABLE1
print('\nall 25 decisions identical to Table 1:', same, '| supported:', sum(v == 'supported' for v in decisions.values()), 'of 25')
print('replication judged on the two unseen seed sets alone:', new_only)
assert same"""

P2_MD = r"""## Part 2 — a small live run (GPU)

Choose **Runtime → Change runtime type → T4 GPU**. Without a GPU, Part 2 is skipped."""

P2_NET = r"""# Load the official EDM network (CIFAR-10 32x32, unconditional, VP) and set up the sampler
import time, pickle
try:
    import torch
    HAS_GPU = torch.cuda.is_available()
except ImportError:
    HAS_GPU = False
RUN_PART2 = HAS_GPU
N_IMAGES, STEPS, SEED = 256, 18, 0
MODEL_URL = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl'
if not RUN_PART2:
    print('No GPU found: Part 2 is skipped. Choose Runtime > Change runtime type > T4 GPU and run again.')
else:
    T = {}
    t0 = time.time()
    if not os.path.isdir('/content/edm'):
        subprocess.run(['git', 'clone', '-q', 'https://github.com/NVlabs/edm.git', '/content/edm'], check=True)
    sys.path.insert(0, '/content/edm')
    import dnnlib
    dev = torch.device('cuda')
    with dnnlib.util.open_url(MODEL_URL) as f:
        net = pickle.load(f)['ema'].to(dev).eval()
    T['EDM code + network download'] = time.time() - t0

    @torch.no_grad()
    def net_fn(x, sigma):
        s = torch.as_tensor(sigma, dtype=torch.float64, device=dev)
        return net(x.reshape(-1, 3, 32, 32), s).to(torch.float64).reshape(x.shape[0], -1)

    sigmas = C1.edm_sigmas(STEPS)   # sigma from 80 to 0.002, rho = 7 (EDM defaults): 35 network calls
    print('GPU:', torch.cuda.get_device_name(0), '| network loaded in %.0f s' % T['EDM code + network download'])"""

P2_MOM = r"""# Gaussian denoiser D_G from the CIFAR-10 training images (same scaling as EDM: pixel/127.5 - 1, CHW order)
if RUN_PART2:
    X, source, T['CIFAR-10 download'], failed = DC.load_cifar10_train('/content/cifar')   # Hugging Face first, torchvision if it fails
    gram = lambda A: (lambda a: (a.T @ a).cpu().numpy())(torch.as_tensor(A, dtype=torch.float64, device=dev))
    fp = DC.cifar10_fingerprint(X, gram=gram)                # exact and independent of the order of the images
    print('CIFAR-10 from %s in %.1f s | %d images | content fingerprint %s...' % (source, T['CIFAR-10 download'], fp['n'], fp['content_sha256'][:16]))
    print(DC.fingerprint_note(fp))                           # compared with the fingerprint measured when this demo was prepared
    t0 = time.time()
    R = DC.to_chw_rows(X)                                    # uint8 (50000, 3072)
    mu, cov, n = C1.moments_from_batches(R[i:i + 5000].astype(np.float64) / 127.5 - 1 for i in range(0, len(R), 5000))
    lam, U, n_neg = C1.eig_moments(cov)
    to_dev = lambda a: torch.as_tensor(a, dtype=torch.float64, device=dev)
    G = C1.GaussianModel(mu, lam, U, to_backend=to_dev)
    T['mean + covariance + eigendecomposition'] = time.time() - t0
    print('%d images, d = %d | moments and eigendecomposition in %.1f s'
          % (n, len(mu), T['mean + covariance + eigendecomposition']))"""

P2_RUN = r"""# Sample 256 images three ways from the same starting noise: network only (M0), U_min 2.38, U_env
if RUN_PART2:
    g = torch.Generator(device=dev).manual_seed(SEED)
    latents = torch.randn(N_IMAGES, 3072, generator=g, device=dev, dtype=torch.float64)
    t0 = time.time()
    with torch.no_grad():
        res = DC.run_live(net_fn, G, latents, sigmas, subset_seed=SEED, sync=torch.cuda.synchronize)
    T['sampling (M0, u curve, U_min, U_env)'] = time.time() - t0
    print('\n'.join(DC.summary_lines(res)))
    print('\nmeasured times (s):', {k: round(v, 1) for k, v in T.items()})"""

P2_PLOT = r"""# The signal rising, the switches, and the images from the same starting noise
if RUN_PART2:
    DC.plot_live(plt, res); plt.show()
    DC.plot_images(plt, res, (3, 32, 32), n=10); plt.show()
    for name in ('U_min', 'U_env'):
        d = (res[name]['x'] - res['M0']['x']).abs().mean().item() * 127.5
        print('%s: mean absolute pixel difference from M0 = %.1f (0-255 scale); the samples differ in detail, '
              'because the rule followed D_G in the fog' % (name, d))"""

OUTRO = r"""## What to take from this

- The signal $u$ is flat in the fog and rises as the fog lifts; the rules switch on that rise without looking at image quality.
- The live run shows the savings on one batch of 256 images. **Image quality is not judged here.** In the paper, under pre-registered criteria, both rules kept FID within 3% of M0 on two AFHQv2 networks in a main test and a replication (Table 1; all 17 AFHQv2 hypotheses supported), while on CIFAR-10 four of eight hypotheses were not supported (Part 1 recomputes all of them).
- To reproduce the paper's GPU runs, use `notebooks/C3_colab.ipynb` (main test) and `notebooks/C6_colab.ipynb` (replication); they need several hours on an L4."""


def code(src, tags):
    return {"cell_type": "code", "execution_count": None, "metadata": {"tags": tags}, "outputs": [],
            "source": src.splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def build():
    cells = [md(INTRO), code(UNZIP, ["unzip"]), code(SETUP, ["setup"]), md(P1_MD), code(P1_REPLAY, ["part1"]), code(P1_HC, ["part1"]),
             md(P2_MD), code(P2_NET, ["part2"]), code(P2_MOM, ["part2"]), code(P2_RUN, ["part2"]), code(P2_PLOT, ["part2"]),
             md(OUTRO)]
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    p = os.path.join(HERE, "demo_colab.ipynb")
    json.dump(nb, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return p


if __name__ == "__main__":
    print(build())
