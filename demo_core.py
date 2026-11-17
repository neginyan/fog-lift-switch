# demo_core.py -- helpers for notebooks/demo_colab.ipynb (MIT License, (c) 2026 Taishi Namba)
#
# The demo uses the rule and denoiser code of this repository unchanged (c1core.py, c2core.py, c3core.py).
# This file only (1) replays the recorded runs, (2) recomputes the pre-registered decisions from the records,
# and (3) runs one small live batch with a network supplied by the caller. It works with numpy arrays
# (CPU, used by verify/verify_all.py) and with torch tensors (GPU, used in the notebook).
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT, os.path.join(ROOT, "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)
import c1core as C1  # noqa: E402
import c2core as C2  # noqa: E402
import c3core as C3  # noqa: E402

REC = os.path.join(ROOT, "records")
SUP, NOT = "支持", "不支持"  # how the records spell "supported" / "not supported"
# Table 1 of the paper: outcome of every pre-registered hypothesis
PAPER_TABLE1 = {**{"HC%d" % i: "supported" for i in range(1, 26)},
                **{h: "not supported" for h in ("HC4", "HC5", "HC6", "HC8")}}


def _en(v):
    return {SUP: "supported", NOT: "not supported"}.get(v, v)


# ----------------------------------------------------------------------------------------------------------
# Part 1: replay the records
# ----------------------------------------------------------------------------------------------------------
def _batch_files(d):
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.startswith("b") and f.endswith(".json"))


def load_cifar_umin(net):
    """C2 (CIFAR-10, 18 steps): the recorded U_min 2.38 runs (traces end at the switch) and the longer 7.35 runs."""
    out = {}
    for c in ("2.38", "7.35"):
        d = os.path.join(REC, "C2_results", net, "runs", "U_min_c%s_N018_s0" % c)
        out[c] = [json.load(open(f, encoding="utf-8")) for f in _batch_files(d)]
    return out


def load_afhq(net, rule, seedset):
    """C3 (seed set 0) and C6 (seed sets 2, 3): AFHQv2, 40 steps. rule = 'U_env' or 'U_min'."""
    base = os.path.join(REC, "C3_results" if seedset == 0 else "C6_results", net, "runs")
    tag = ("U_env_N040_s%d" if rule == "U_env" else "U_min_c2.38_N040_s%d") % seedset
    return [json.load(open(f, encoding="utf-8")) for f in _batch_files(os.path.join(base, tag))]


def replay_umin(us, c_min=2.38):
    return C2.replay_umin(us, c_min)


def replay_uenv(sigmas, us):
    k = C3.PREREG3["b22_constants"]
    rule = C3.UEnvRule(k["DELTA"], k["C"], k["M_MIN"], k["window_tol"])
    return next((i for i, (s, u) in enumerate(zip(sigmas, us)) if rule.observe(s, u)), None)


def replay_all():
    """Replay both rules on every recorded trace; return rows and whether every switch matches the record."""
    rows = []
    for net in ("vp", "ve"):
        for j in load_cifar_umin(net)["2.38"]:
            u = [t[2] for t in j["trace"]]
            k = replay_umin(u)
            rows.append({"data": "CIFAR-10", "net": net, "seedset": 0, "batch": j["batch"], "rule": "U_min",
                         "recorded_sigma": j["switch_sigma"], "replayed_sigma": j["trace"][k][1] if k is not None else None,
                         "match": k == len(u) - 1 and j["trace"][k][0] == j["switch_call"]})
    for net in ("vp", "ve"):
        for s in (0, 2, 3):
            for rule in ("U_env", "U_min"):
                for j in load_afhq(net, rule, s):
                    sg = [t[1] for t in j["trace"]]
                    u = [t[3] for t in j["trace"]]
                    k = replay_uenv(sg, u) if rule == "U_env" else replay_umin(u)
                    rows.append({"data": "AFHQv2", "net": net, "seedset": s, "batch": j["batch"], "rule": rule,
                                 "recorded_sigma": j["switch_sigma"], "replayed_sigma": sg[k] if k is not None else None,
                                 "match": k == len(u) - 1 and j["trace"][k][0] == j["switch_call"]})
    return rows


def plot_records(plt):
    """Two figures: CIFAR-10 (C2) and AFHQv2 (C3 + C6), with the replayed switches."""
    figs = []
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for net, col in (("vp", "#1f5fa8"), ("ve", "#c4432b")):
        d = load_cifar_umin(net)
        for i, j in enumerate(d["7.35"]):
            ax.loglog([t[1] for t in j["trace"]], [t[2] for t in j["trace"]], ".-", color=col, alpha=0.3, lw=0.7,
                      label="CIFAR-10 %s (10 batches)" % net.upper() if i == 0 else None)
        for i, j in enumerate(d["2.38"]):
            u = [t[2] for t in j["trace"]]
            k = replay_umin(u)
            ax.plot(j["trace"][k][1], u[k], "x", color=col, ms=8, label="U_min 2.38 switch (replayed)" if i == 0 else None)
    ax.invert_xaxis()
    ax.set_xlabel("sigma (sampling runs left to right)")
    ax.set_ylabel("u = median |D_G - D_net|^2 / d")
    ax.set_title("CIFAR-10 (stage C2): the VE network spikes at sigma = 40.79 and U_min switches too early")
    ax.legend(fontsize=7)
    figs.append(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, net in zip(axes, ("vp", "ve")):
        for s, col in ((0, "#999999"), (2, "#4f8fcf"), (3, "#e08a3c")):
            env, umn = load_afhq(net, "U_env", s), load_afhq(net, "U_min", s)
            for i, (je, jm) in enumerate(zip(env, umn)):
                long = je if len(je["trace"]) >= len(jm["trace"]) else jm
                ax.loglog([t[1] for t in long["trace"]], [t[3] for t in long["trace"]], ".-", color=col, alpha=0.5, lw=0.6,
                          label="seed set %d" % s if i == 0 else None)
                ke = replay_uenv([t[1] for t in je["trace"]], [t[3] for t in je["trace"]])
                km = replay_umin([t[3] for t in jm["trace"]])
                ax.plot(jm["trace"][km][1], jm["trace"][km][3], "x", color="#c4432b", ms=6,
                        label="U_min switch" if (s == 0 and i == 0) else None)
                ax.plot(je["trace"][ke][1], je["trace"][ke][3], "+", color="#1f5fa8", ms=8, mew=1.3,
                        label="U_env switch" if (s == 0 and i == 0) else None)
        ax.invert_xaxis()
        ax.set_xlabel("sigma")
        ax.set_title("AFHQv2 64x64, %s network (C3 + C6, 15 batches)" % net.upper())
    axes[0].set_ylabel("u (5% subset median)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    figs.append(fig)
    return figs


def recompute_decisions():
    """Recompute HC1-HC25 from the records with the code of this repository. Returns {HC: 'supported'/'not supported'}."""
    import analyze_c2 as A2
    import analyze_c3 as A3
    import analyze_c6 as A6
    s1 = json.load(open(os.path.join(REC, "C1_results", "report", "summary_C1.json"), encoding="utf-8"))
    out = {h: _en(v) for h, v in C1.judge(s1["results"]).items() if h in ("HC1", "HC2", "HC3", "HC4")}
    out.update({h: _en(v) for h, v in A2.main()["judgement"].items()})
    o3 = A3.main()
    for net in ("vp", "ve"):
        out.update({h: _en(v) for h, v in o3[net]["judgement"].items()})
    o6 = A6.main()
    for net in ("vp", "ve"):
        out.update({h: _en(v) for h, v in o6["nets"][net]["hc"].items()})
    out["HC25"] = _en(o6["B"]["HC25"])
    new_only = {"%s %s (%s)" % (n.upper(), k.replace("pooled_new_only_", ""), c): _en(v[c])
                for n in ("vp", "ve") for k, v in o6["nets"][n]["recomputed"].items() if "new_only" in k for c in ("a", "b")}
    return dict(sorted(out.items(), key=lambda kv: int(kv[0][2:]))), new_only


# ----------------------------------------------------------------------------------------------------------
# Part 2: one small live batch
# ----------------------------------------------------------------------------------------------------------
def run_live(net_fn, G, latents, sigmas, subset_seed=0, sync=None):
    """Run M0 (network only), U_min 2.38 and U_env on the same starting noise, and the u curve along M0.

    latents: (n, d) standard normal (numpy or torch). Returns a dict with final samples, traces, costs and times.
    """
    def tick():
        if sync:
            sync()
        return time.time()

    x_init = latents * float(sigmas[0])
    res = {}
    t = tick()
    x_m0 = C1.heun_sample(x_init, sigmas, lambda x, s, i, c: net_fn(x, s))
    res["M0"] = {"x": x_m0, "cost": 1.0, "seconds": tick() - t}
    t = tick()
    res["u_along_M0"] = C1.pilot_curve(x_init, sigmas, net_fn, G)  # every call evaluated on every image (not a rule)
    res["u_along_M0_seconds"] = tick() - t
    t = tick()
    x, info = C3.run_signal_rule(C3.UMinAdapter(C3.PREREG3["umin_c_min"]), x_init, sigmas, net_fn, G,
                                 np.random.default_rng(subset_seed))
    res["U_min"] = {"x": x, "info": info, "cost": info["cost_ratio"], "seconds": tick() - t}
    C3.install_uenv()
    t = tick()
    x, info = C3.run_signal_rule(C3.B22["make_rule"](), x_init, sigmas, net_fn, G, np.random.default_rng(subset_seed),
                                 stat_fn=C3.B22["stat_fn"], frac=C3.B22["frac"], time_thin=C3.B22["time_thin"])
    res["U_env"] = {"x": x, "info": info, "cost": info["cost_ratio"], "seconds": tick() - t}
    return res


def summary_lines(res):
    lines = []
    for name in ("M0", "U_min", "U_env"):
        r = res[name]
        sw = "" if name == "M0" else "  switch at call %s (sigma = %.2f)" % (r["info"]["switch_call"], r["info"]["switch_sigma"])
        lines.append("%-6s network evaluations / M0 = %.3f  (saved %4.1f%%)  wall time %.1f s%s" % (
            name, r["cost"], 100 * (1 - r["cost"]), r["seconds"], sw))
    return lines


def plot_live(plt, res):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    p = res["u_along_M0"]
    ax.loglog([r["sigma"] for r in p], [r["dD2_per_dim_median"] for r in p], "-", color="#bbbbbb", lw=2,
              label="u along the network-only run (all images)")
    for name, col, mk in (("U_min", "#c4432b", "x"), ("U_env", "#1f5fa8", "+")):
        tr = res[name]["info"]["trace"]
        ax.loglog([t[1] for t in tr], [t[3] for t in tr], ".-", color=col, alpha=0.7, lw=0.8,
                  label="%s: u on the 5%% subset, until the switch" % name)
        ax.plot(tr[-1][1], tr[-1][3], mk, color=col, ms=10, mew=1.5)
    ax.invert_xaxis()
    ax.set_xlabel("sigma (sampling runs left to right)")
    ax.set_ylabel("u = median |D_G - D_net|^2 / d")
    ax.set_title("Live: the fog lifts, u rises, and the rules hand over to the network")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def image_grid(x, img_shape, n=8):
    a = C1.to_uint8_images(x[:n], img_shape)  # (n, C, H, W)
    return np.concatenate(list(a.transpose(0, 2, 3, 1)), axis=1)


def plot_images(plt, res, img_shape=(3, 32, 32), n=8):
    fig, axes = plt.subplots(3, 1, figsize=(n * 0.9, 3.3))
    for ax, name in zip(axes, ("M0", "U_min", "U_env")):
        ax.imshow(image_grid(res[name]["x"], img_shape, n))
        ax.set_axis_off()
        ax.set_title("%s  (network evaluations %.0f%% of M0)" % (name, 100 * res[name]["cost"]), fontsize=8)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------------------------------------
# CIFAR-10 training images for Part 2: a fast official source first, torchvision's source as the fallback
# ----------------------------------------------------------------------------------------------------------
CIFAR10_SHAPE = (50000, 32, 32, 3)
# First 16 hex digits of cifar10_fingerprint(...)["content_sha256"] for the Hugging Face copy, measured on Colab when this
# demo was prepared (the full value will be added after notebooks/cifar10_source_check.ipynb has been run).
CIFAR10_FINGERPRINT_PREFIX = "4713a5cd450ffa7c"


def fingerprint_note(fp):
    """Compare a fingerprint with the one measured when the demo was prepared; never stops the notebook."""
    if fp["content_sha256"].startswith(CIFAR10_FINGERPRINT_PREFIX):
        return "fingerprint %s... matches the fingerprint measured when this demo was prepared" % CIFAR10_FINGERPRINT_PREFIX
    return ("NOTE: fingerprint %s... differs from the one measured when this demo was prepared (%s...): the images "
            "or their layout differ from those used then; the demo continues, but its numbers may differ"
            % (fp["content_sha256"][:16], CIFAR10_FINGERPRINT_PREFIX))


def cifar10_train_hf():
    """CIFAR-10 training images from the Hugging Face dataset uoft-cs/cifar10 (the University of Toronto's account),
    read from its parquet files with huggingface_hub and pyarrow. Returns uint8 (50000, 32, 32, 3)."""
    import io
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    from PIL import Image
    files = sorted(f for f in HfApi().list_repo_files("uoft-cs/cifar10", repo_type="dataset")
                   if f.endswith(".parquet") and "train" in f.split("/")[-1])
    imgs = []
    for f in files:
        t = pq.read_table(hf_hub_download("uoft-cs/cifar10", f, repo_type="dataset"))
        col = next(c for c in t.column_names if c in ("img", "image"))
        imgs += [np.asarray(Image.open(io.BytesIO(d["bytes"])).convert("RGB"), dtype=np.uint8) for d in t.column(col).to_pylist()]
    return np.stack(imgs)


def cifar10_train_torchvision(root):
    """CIFAR-10 training images via torchvision (downloads from www.cs.toronto.edu). Returns uint8 (50000, 32, 32, 3)."""
    import torchvision
    return torchvision.datasets.CIFAR10(root, train=True, download=True).data


def load_cifar10_train(root="/content/cifar", loaders=None):
    """Try the sources in order; accept the first that returns uint8 images of shape (50000, 32, 32, 3).
    Returns (images, source name, seconds, [(source, error) for sources that failed])."""
    if loaders is None:
        loaders = [("Hugging Face uoft-cs/cifar10 (parquet)", cifar10_train_hf),
                   ("torchvision CIFAR10 (www.cs.toronto.edu)", lambda: cifar10_train_torchvision(root))]
    failed = []
    for name, fn in loaders:
        t0 = time.time()
        try:
            X = np.asarray(fn())
            if X.shape != CIFAR10_SHAPE or X.dtype != np.uint8:
                raise ValueError("unexpected array %s %s" % (X.shape, X.dtype))
            return X, name, time.time() - t0, failed
        except Exception as e:  # noqa: BLE001  (any failure: fall back to the next source)
            failed.append((name, repr(e)[:300]))
            print("CIFAR-10 source failed: %s (%s); trying the next one" % (name, failed[-1][1]))
    raise RuntimeError("no CIFAR-10 source worked: %s" % failed)


def to_chw_rows(X):
    """(n, H, W, C) uint8 -> (n, C*H*W), the layout EDM uses (channel, row, column)."""
    return X.transpose(0, 3, 1, 2).reshape(len(X), -1)


def cifar10_fingerprint(X, gram=None, chunk=5000):
    """An exact, order-independent fingerprint of a set of uint8 images in the CHW layout.

    content_sha256 hashes the per-dimension pixel sums and the Gram matrix sum_i x_i x_i^T, both as exact integers
    (the Gram matrix is computed in float64, which is exact here: every partial sum is an integer below 2^53).
    It does not depend on the order of the images but does depend on the layout (CHW vs HWC) and on every pixel value.
    The mean and covariance used by the Gaussian denoiser are functions of exactly these sums.
    byte_sha256_in_order hashes the raw bytes in their given order (it differs if only the order differs).
    gram: optional function A (k, d) float64 -> A^T A (e.g. on a GPU); numpy is used if None."""
    import hashlib
    R = to_chw_rows(X)
    n, d = R.shape
    s = R.sum(0, dtype=np.int64)
    G = np.zeros((d, d))
    for i in range(0, n, chunk):
        A = R[i:i + chunk].astype(np.float64)
        G += gram(A) if gram is not None else A.T @ A
    assert G.max() < 2 ** 53 and np.all(G == np.round(G))
    h = hashlib.sha256()
    h.update(np.int64(n).tobytes())
    h.update(s.astype("<i8").tobytes())
    h.update(np.round(G).astype("<i8").tobytes())
    return {"n": int(n), "d": int(d), "content_sha256": h.hexdigest(),
            "byte_sha256_in_order": hashlib.sha256(np.ascontiguousarray(R).tobytes()).hexdigest(),
            "pixel_sum": int(s.sum())}
