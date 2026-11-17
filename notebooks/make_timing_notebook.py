# make_timing_notebook.py -- writes notebooks/timing_colab.ipynb (MIT License, (c) 2026 Taishi Namba)
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

INTRO = r"""# Timing the Gaussian phase: current implementation vs `fast_gauss` (AFHQv2, L4, about 20 minutes)

Before the switch, the rules follow the Gaussian denoiser $D_G$. The implementation used for the paper's runs (`c3core.run_signal_rule`) advances the whole batch with $D_G$ at every call (two $d\times d$ products per call, $d = 12{,}288$). Because $D_G$ is linear, each Heun step with $D_G$ multiplies every eigen-coordinate by a number, so `fast_gauss.run_signal_rule_fast` computes only the 5% subset at each call and reconstructs the whole batch once, at the switch. Both give the same switch and, up to floating-point rounding, the same images.

This notebook measures, for the AFHQv2 VP and VE networks, on **seed set 0, batch 0 (1,000 images)** — the batch recorded in stage C3:

1. the Gaussian phase (start to switch) of `U_min 2.38` with the current and the fast implementation, and of `U_env` with the fast one;
2. one complete fast `U_min` run, whose switch, signal trace and **images** are compared with the run recorded in C3;
3. wall-clock ratios to the recorded M0 time of the same batch (also on an L4).

**Needs** the Google Drive folder `C3_results/` of the main test (it holds the Gaussian moments `moments` file and the recorded batch 0). **Runtime:** L4 GPU, about 10 minutes per network. Licences: EDM code and networks CC BY-NC-SA 4.0 (loaded from the official locations); our code MIT."""

SETUP = r"""# (1) Settings, Drive, GPU, repository and EDM code
ROOT = '/content/drive/MyDrive/C3_results'   # the folder of the main test (read only)
NETS = ['vp', 've']
NET_CHUNK = 125
from google.colab import drive
drive.mount('/content/drive')
import os, sys, json, time, pickle, subprocess
import numpy as np, torch
assert torch.cuda.is_available(), 'No GPU: choose Runtime > Change runtime type > L4 GPU.'
dev = torch.device('cuda'); print('GPU:', torch.cuda.get_device_name(0))
REPO = '/content/fog-lift-switch'
if not os.path.exists(os.path.join(REPO, 'fast_gauss.py')):
    subprocess.run(['git', 'clone', '-q', 'https://github.com/neginyan/fog-lift-switch.git', REPO], check=True)
if not os.path.isdir('/content/edm'):
    subprocess.run(['git', 'clone', '-q', 'https://github.com/NVlabs/edm.git', '/content/edm'], check=True)
sys.path[:0] = [REPO, '/content/edm']
import dnnlib
import c1core as C1, c2core as C2, c3core as C3, fast_gauss as FG
P = C3.PREREG3
assert C3.C3_VERSION == 'C3-1.0'"""

MOM = r"""# (2) Gaussian moments of AFHQv2 from the main test (not rebuilt)
z = np.load(C3.moments_path(ROOT))
assert int(z['n']) == P['dataset_n_images'] == 15803
to_dev = lambda a: torch.as_tensor(a, dtype=torch.float64, device=dev)
G = C1.GaussianModel(z['mu'], z['lam'], z['U'], to_backend=to_dev)
print('moments loaded:', int(z['n']), 'images, d =', len(z['mu']))"""

RUN = r"""# (3) For each network: Gaussian phase (current vs fast), one full fast run, comparison with the C3 record
k = P['b22_constants']
make_env = lambda: C3.UEnvRule(k['DELTA'], k['C'], k['M_MIN'], k['window_tol'])
RESULTS = {}
for NET in NETS:
    with dnnlib.util.open_url(C3.net_url(NET)) as f:
        net = pickle.load(f)['ema'].to(dev).eval()
    srange = C2.sampler_range(net.sigma_min, net.sigma_max)
    sig = C3.sigmas3(P['steps'], srange)

    @torch.no_grad()
    def net_fn(x, sigma):
        out = torch.empty_like(x)
        s = torch.as_tensor(sigma, dtype=torch.float64, device=dev)
        for i in range(0, x.shape[0], NET_CHUNK):
            xb = x[i:i + NET_CHUNK].reshape(-1, 3, 64, 64)
            out[i:i + NET_CHUNK] = net(xb, s).to(torch.float64).reshape(xb.shape[0], -1)
        return out

    g = torch.Generator(device=dev).manual_seed(1000003 * 0 + 0)          # seed set 0, batch 0 (as in C3)
    x0 = torch.randn(1000, 12288, generator=g, device=dev, dtype=torch.float64) * float(sig[0])
    rng = lambda: C1.subset_rng(0, P['steps'], 0)
    sync = torch.cuda.synchronize
    rec_dir = os.path.join(C3.net_dir(ROOT, NET), 'runs')
    rec_min = json.load(open(os.path.join(rec_dir, 'U_min_c2.38_N040_s0', 'b000.json')))
    rec_env = json.load(open(os.path.join(rec_dir, 'U_env_N040_s0', 'b000.json')))
    rec_m0 = json.load(open(os.path.join(rec_dir, 'M0_N040_s0', 'b000.json')))
    R = {'M0_recorded_seconds': rec_m0['seconds']}
    with torch.no_grad():
        net_fn(x0[:NET_CHUNK], 80.0); sync()                               # warm-up
        R['gauss_phase_current_U_min'] = FG.gaussian_phase(C3.run_signal_rule, C3.UMinAdapter(2.38), x0, sig, net_fn, G, rng(), sync=sync)[:2]
        R['gauss_phase_fast_U_min'] = FG.gaussian_phase(FG.run_signal_rule_fast, C3.UMinAdapter(2.38), x0, sig, net_fn, G, rng(), sync=sync)[:2]
        R['gauss_phase_fast_U_env'] = FG.gaussian_phase(FG.run_signal_rule_fast, make_env(), x0, sig, net_fn, G, rng(), stat_fn=C3.stat_u_finite, sync=sync)[:2]  # as installed for U_env in C3
        x_fast, info = FG.run_signal_rule_fast(C3.UMinAdapter(2.38), x0, sig, net_fn, G, rng(), sync=sync)
    R['fast_full_U_min_seconds'] = info['seconds']
    R['switch_call'] = {'U_min fast': info['switch_call'], 'U_min recorded': rec_min['switch_call'],
                        'U_env fast (eval index)': R['gauss_phase_fast_U_env'][1], 'U_env recorded (eval index)': len(rec_env['trace']) - 1}
    u_fast = np.array([t[3] for t in info['trace']]); u_rec = np.array([t[3] for t in rec_min['trace']])
    R['u_trace_max_rel_diff'] = float(np.max(np.abs(u_fast - u_rec) / u_rec)) if len(u_fast) == len(u_rec) else None
    img_fast = C1.to_uint8_images(x_fast, (3, 64, 64))
    rec_npz = os.path.join(rec_dir, 'U_min_c2.38_N040_s0', 'b000.npz')
    if os.path.exists(rec_npz):
        img_rec = np.load(rec_npz)['images']
        dif = np.abs(img_fast.astype(int) - img_rec.astype(int))
        R['images_vs_record'] = {'max_abs_diff_0_255': int(dif.max()), 'pixels_differing_fraction': float((dif > 0).mean())}
    cur_pre, fast_pre = R['gauss_phase_current_U_min'][0], R['gauss_phase_fast_U_min'][0]
    post = R['fast_full_U_min_seconds'] - fast_pre
    R['wall_ratio_to_M0'] = {'current (pre + post)': (cur_pre + post) / rec_m0['seconds'],
                             'fast (measured full run)': R['fast_full_U_min_seconds'] / rec_m0['seconds'],
                             'recorded C3 U_min run': rec_min['seconds'] / rec_m0['seconds']}
    R['cost_ratio'] = info['cost_ratio']
    RESULTS[NET] = R
    print(NET.upper(), json.dumps(R, indent=1, default=float))
    del net; torch.cuda.empty_cache()"""

REPORT = r"""# (4) Summary to bring back (copy this output, or the file timing_C10.json in ROOT)
json.dump(RESULTS, open(os.path.join(ROOT, 'timing_C10.json'), 'w'), indent=1, default=float)
for NET, R in RESULTS.items():
    print('%s: switch %s | u trace max rel diff %s | images %s' % (NET.upper(), R['switch_call'], R['u_trace_max_rel_diff'], R.get('images_vs_record')))
    print('     Gaussian phase U_min: current %.1f s, fast %.1f s | wall / M0: %s | network evaluations / M0 %.4f'
          % (R['gauss_phase_current_U_min'][0], R['gauss_phase_fast_U_min'][0],
             {k: round(v, 3) for k, v in R['wall_ratio_to_M0'].items()}, R['cost_ratio']))"""


def code(src, tags):
    return {"cell_type": "code", "execution_count": None, "metadata": {"tags": tags}, "outputs": [],
            "source": src.splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def build():
    cells = [md(INTRO), code(SETUP, ["setup"]), code(MOM, ["run"]), code(RUN, ["run"]), code(REPORT, ["run"])]
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    p = os.path.join(HERE, "timing_colab.ipynb")
    json.dump(nb, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return p


if __name__ == "__main__":
    print(build())
