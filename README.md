# When does the fog lift? — a switching rule fixed on synthetic worlds, tested by pre-registration on pretrained image diffusion models

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24301909.svg)](https://doi.org/10.5281/zenodo.24301909)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neginyan/fog-lift-switch/blob/main/notebooks/demo_colab.ipynb)
[![verify](https://github.com/neginyan/fog-lift-switch/actions/workflows/verify.yml/badge.svg)](https://github.com/neginyan/fog-lift-switch/actions/workflows/verify.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code, pre-registrations and run records for the paper **"When does the fog lift? A switching rule fixed on synthetic worlds, tested by pre-registration on pretrained image diffusion models"** (Taishi Namba, paper G3; preprint [doi:10.5281/zenodo.24301909](https://doi.org/10.5281/zenodo.24301909)).

## In one paragraph

A diffusion sampler begins in a "fog" of noise where the data are visible only through their mean and covariance, and a closed-form Gaussian denoiser `D_G` is almost exact. The sampler can follow `D_G` there and call the expensive network only after the fog lifts. We detect that moment **while sampling and without looking at image quality**: the signal is `u = median |D_G − D_net|² / d` on 5% of the batch, which is low in the fog and rises when it lifts. Two rules read it — `U_min` (switch when `u` exceeds 2.38 × its running minimum) and the flicker-robust `U_env` — with constants fixed on synthetic worlds only. Under pre-registration, on two EDM networks for AFHQv2 64×64 that nobody had inspected, **all 8 hypotheses of the main test and all 9 of a replication on two further seed sets were supported**; both rules kept FID within 3% of the full sampler while saving about 25% of network evaluations (18–22% of wall-clock time with the implementation used for the runs; 24–25% with `fast_gauss.py`, which computes the Gaussian phase only on the 5% subset; measured on one batch per network), and on the VE network the price of not looking was 1.2–1.6% of the full cost against the best switching time chosen with FID on a fine grid. On CIFAR-10 the transfer partly failed (4 of 8 hypotheses not supported), and the paper reports this, the small quality cost on the VE network and the other limits with the same weight.

## What is here

| Path | Contents |
|---|---|
| `c1core.py`, `c2core.py`, `c3core.py` | Gaussian denoiser, teleportation, EDM Heun sampler, the rules `U_min` and `U_env`, cost accounting, FID helpers, judgments (version C3-1.0, exactly as run) |
| `c6core.py` | Replication with new seed sets and the finer VE grid (version C6-1.0, exactly as run) |
| `notebooks/demo_colab.ipynb` | **A few-minute demo** (the Colab badge above): Part 1 replays the recorded runs and recomputes HC1–HC25 on CPU in seconds; Part 2 samples 256 CIFAR-10 images on a free T4 with the network alone, `U_min` and `U_env`, and shows the signal, the switches, the images and the savings (no FID) |
| `demo_core.py` | Helpers for the demo (uses the rule code unchanged); `notebooks/make_demo_notebook.py` writes the notebook |
| `notebooks/cifar10_source_check.ipynb` | A one-time check that the two CIFAR-10 sources of the demo (Hugging Face `uoft-cs/cifar10` and torchvision) give the same 50,000 images in the same layout (exact fingerprint) and the same moments; written by `notebooks/make_cifar_check_notebook.py` |
| `fast_gauss.py` | An equivalent implementation of the Gaussian phase: since `D_G` is linear, the Heun steps before the switch are multiplied out per eigen-direction, so only the 5% subset is computed at each call and the whole batch once, at the switch. Same switches and, up to rounding, the same samples as `c3core.run_signal_rule` (checked by `verify`) |
| `notebooks/timing_colab.ipynb` | Measures the wall-clock time of the Gaussian phase, current vs `fast_gauss`, on the recorded AFHQv2 batch 0 (L4, about 20 minutes; needs the `C3_results/` folder of the main test); written by `notebooks/make_timing_notebook.py` |
| `records/C10_timing/timing_C10.json` | The wall-clock measurement of `fast_gauss` (Table 7 of the paper): `notebooks/timing_colab.ipynb` run once by the author on Google Colab (NVIDIA L4) on seed set 0, batch 0 of both AFHQv2 networks; checked by `verify` |
| `analysis/uenv_window.py` | The `U_env` window on the 18- and 40-step schedules (Table 6 of the paper) and the state of both rules at their switches on all 30 AFHQv2 batches |
| `notebooks/C1_colab.ipynb` … `C6_colab.ipynb` | The notebooks of the paper's runs: stages C1, C2, C3 (main test) and C6 (replication). Comments and messages are in Japanese. |
| `preregistration/` | Pre-registration documents (Japanese) and their SHA-256 hashes, per stage |
| `rules/` | The rule-definition documents handed from the synthetic-world lab (B21: `U_min`; B22: `U_env`) |
| `records/` | Run records: per-batch JSON files with every recorded signal value, switch and cost; summaries with FIDs; logs |
| `analysis/` | Scripts that recompute every reported number from the records, and `make_paper_figures.py` (Figures 1–6) |
| `figures/` | The six figures of the paper |
| `verify/verify_all.py` | One CPU check of everything (below) |

No third-party code, weights or data are included. The notebooks load them from their official locations.

## Verify

```bash
pip install -r requirements.txt
python verify/verify_all.py        # about 30 s on a laptop CPU; no GPU, no network access
```

It recomputes, from `records/`, every pre-registered decision (HC1–HC25), every number in Tables 1–3 of the paper, and every switching time (by replaying both rules on the recorded signal traces, with an independent re-implementation of each rule). It also checks the SHA-256 of the rule code and of every pre-registration, that the code cells of the notebooks are the code in this repository, and exact properties of the Gaussian denoiser and teleportation, each with a power check (a deliberately wrong version must fail). It also runs Part 1 of the demo notebook on CPU and the demo's live pipeline with a small synthetic network. It recomputes the `U_env` window table from the schedules and checks that `fast_gauss` reproduces the switches, the signal and the samples of the implementation used for the runs, and recomputes the measured timings and the wall-clock ranges quoted in the paper. It runs on every push via GitHub Actions.

FID values are taken from the run records, because the Inception features were not stored.

## Try it

Open the demo with the Colab badge. Part 1 needs no GPU; for Part 2 choose *Runtime → Change runtime type → T4 GPU*. Part 1 takes seconds on a CPU; Part 2 takes a few minutes on a Colab GPU (downloads included). If Hugging Face is unavailable, the fallback download of CIFAR-10 from Toronto can take 20 minutes or more. A warning about unauthenticated requests to the Hugging Face Hub (HF_TOKEN) may appear; it can be ignored. The demo computes no FID: image quality was judged in the paper on 10,000–15,000 images per method under pre-registered criteria.

## Reproduce the GPU runs

The notebooks run on Google Colab (C1–C2: T4; C3, C6: L4). The main test (`notebooks/C3_colab.ipynb`) takes about 6–7 hours per network on an L4, and the replication (`notebooks/C6_colab.ipynb`) about 14 hours in total. Results go to Google Drive, finished batches are skipped on restart, and the signal curves are not displayed until the final cell, which aggregates only when every registered run is complete.

## Licences

- Our code: **MIT** (`LICENSE`).
- EDM code and pretrained networks (NVIDIA): CC BY-NC-SA 4.0 — loaded by the notebooks from the official repository and distribution site.
- AFHQv2 (NAVER Corp.): CC BY-NC 4.0 — downloaded by the notebooks with the official StarGAN v2 script; only its mean and covariance are used.
- CIFAR-10, the Inception-v3 feature extractor and FID reference statistics used by EDM's FID code: loaded from their official sources under their own terms.

## Related

- Paper G: *When the blur is slower than the flow: clipped and corrected samplers on the critically damped Langevin flow* — doi:10.5281/zenodo.23689913, code: [cld-corrected-sampler](https://github.com/neginyan/cld-corrected-sampler)
- Paper G2: *Learned scores on the critically damped Langevin flow: a corrected sampler in 281 steps, and a metric-dependent fragility of the probability flow* — doi:10.5281/zenodo.23899158, code: [cld-learned-score-samplers](https://github.com/neginyan/cld-learned-score-samplers)

## Acknowledgments

This work was carried out with AI assistance: Anthropic Claude, OpenAI ChatGPT, Google Gemini. The research direction and all decisions were made by the author.

## Citation

```bibtex
@misc{namba2026g3,
  author = {Namba, Taishi},
  title  = {When does the fog lift? A switching rule fixed on synthetic worlds, tested by pre-registration on pretrained image diffusion models},
  year   = {2026},
  note   = {Preprint},
  doi    = {10.5281/zenodo.24301909}
}
```

Contact: Taishi Namba, Independent Researcher, Atsugi, Kanagawa, Japan — neginyan@gmail.com
