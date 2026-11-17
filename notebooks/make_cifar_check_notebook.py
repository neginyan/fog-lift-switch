# make_cifar_check_notebook.py -- writes notebooks/cifar10_source_check.ipynb (MIT License, (c) 2026 Taishi Namba)
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

INTRO = r"""# CIFAR-10 source check: Hugging Face vs torchvision

The demo (`demo_colab.ipynb`) downloads the CIFAR-10 training images from Hugging Face (`uoft-cs/cifar10`) and falls back to torchvision's source (University of Toronto) only if that fails. This notebook checks, once, that both sources give **the same 50,000 images in the same layout** (CHW, as EDM uses), so that the Gaussian denoiser built from them is the same.

It compares an exact, order-independent fingerprint (the per-dimension pixel sums and the Gram matrix $\sum_i x_ix_i^\top$ as integers, which determine the mean and covariance), a byte hash in the given order (to see whether the order is also the same), and the mean and covariance themselves. Optionally it compares them with the moments stored in stage C1 on Google Drive (`C1_results/cifar10_moments.npz`), which were built from the torchvision source.

**Time:** Hugging Face usually takes well under a minute; the torchvision source can take 20 minutes or more when the Toronto server is slow. Any runtime works (a GPU only speeds up the fingerprint)."""

UNZIP = r"""# Only before the repository is public: upload the fog-lift-switch_part0*.zip files to /content (Files pane),
# then remove the '#' at the start of the next line and run this cell.
# !cd /content && for f in fog-lift-switch_part0*.zip; do unzip -q -o "$f"; done"""

SETUP = r"""# Setup
import os, sys, subprocess, json, time
REPO = next((os.path.abspath(p) for p in ('.', '..', '/content/fog-lift-switch') if os.path.exists(os.path.join(p, 'demo_core.py'))), None)
if REPO is None:
    subprocess.run(['git', 'clone', '-q', 'https://github.com/neginyan/fog-lift-switch.git', '/content/fog-lift-switch'], check=True)
    REPO = '/content/fog-lift-switch'
sys.path.insert(0, REPO)
import numpy as np
import demo_core as DC, c1core as C1
USE_DRIVE = False   # True: also compare with C1_results/cifar10_moments.npz on Google Drive (stage C1)
try:
    import torch
    dev = torch.device('cuda') if torch.cuda.is_available() else None
except ImportError:
    dev = None
gram = (lambda A: (lambda a: (a.T @ a).cpu().numpy())(torch.as_tensor(A, dtype=torch.float64, device=dev))) if dev else None
OUT = {}"""

HF = r"""# Source 1: Hugging Face (uoft-cs/cifar10, parquet)
X_hf, src, sec, failed = DC.load_cifar10_train(loaders=[('Hugging Face uoft-cs/cifar10 (parquet)', DC.cifar10_train_hf)])
OUT['hf'] = {'seconds': round(sec, 1), **DC.cifar10_fingerprint(X_hf, gram=gram)}
print(json.dumps(OUT['hf'], indent=1))"""

TV = r"""# Source 2: torchvision (www.cs.toronto.edu) -- can take 20 minutes or more
X_tv, src, sec, failed = DC.load_cifar10_train(loaders=[('torchvision CIFAR10 (www.cs.toronto.edu)', lambda: DC.cifar10_train_torchvision('/content/cifar'))])
OUT['torchvision'] = {'seconds': round(sec, 1), **DC.cifar10_fingerprint(X_tv, gram=gram)}
print(json.dumps(OUT['torchvision'], indent=1))"""

CMP = r"""# Compare: the same images in the same layout (exact), the same order (bytes), the same moments
def moments(X):
    R = DC.to_chw_rows(X)
    mu, cov, n = C1.moments_from_batches(R[i:i + 5000].astype(np.float64) / 127.5 - 1 for i in range(0, len(R), 5000))
    return mu, cov
mu1, cov1 = moments(X_hf)
mu2, cov2 = moments(X_tv)
OUT['same_images_and_layout (content fingerprint)'] = OUT['hf']['content_sha256'] == OUT['torchvision']['content_sha256']
OUT['same_order (byte hash)'] = OUT['hf']['byte_sha256_in_order'] == OUT['torchvision']['byte_sha256_in_order']
OUT['moments max abs diff (mean, cov)'] = [float(np.abs(mu1 - mu2).max()), float(np.abs(cov1 - cov2).max())]
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    z = np.load('/content/drive/MyDrive/C1_results/cifar10_moments.npz')
    OUT['vs C1 moments on Drive: mean max abs diff (hf, torchvision)'] = [float(np.abs(mu1 - z['mu']).max()), float(np.abs(mu2 - z['mu']).max())]
print(json.dumps(OUT, indent=1))
print('\nBring back this output. Expected: content fingerprints equal, moments differences at rounding level (< 1e-12).')"""


def code(src, tags):
    return {"cell_type": "code", "execution_count": None, "metadata": {"tags": tags}, "outputs": [],
            "source": src.splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def build():
    cells = [md(INTRO), code(UNZIP, ["unzip"]), code(SETUP, ["setup"]), code(HF, ["hf"]), code(TV, ["torchvision"]), code(CMP, ["compare"])]
    nb = {"cells": cells, "metadata": {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    p = os.path.join(HERE, "cifar10_source_check.ipynb")
    json.dump(nb, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return p


if __name__ == "__main__":
    print(build())
