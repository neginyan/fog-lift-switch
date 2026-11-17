# make_paper_figures.py -- figures 1-6 of paper G3, drawn from the recorded results (MIT License, (c) 2026 Taishi Namba)
# Usage: python analysis/make_paper_figures.py   (runs analyze_c2/c3/c6 first if their outputs are missing)
# Writes figures/fig1_signal.pdf ... figures/fig6_fid_vs_n.pdf (and .png) at the repository root.
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
OUT = os.environ.get("G3_FIG_DIR", os.path.join(ROOT, "figures"))
REC = os.path.join(ROOT, "records")

C_VP, C_VE = "#1f5fa8", "#c4432b"
C_ENV, C_MIN = "#1f5fa8", "#c4432b"
C_GRID, C_TOL = "#555555", "#888888"


def _load(name, module):
    p = os.path.join(HERE, "out", name, "%s_recomputed.json" % name)
    if not os.path.exists(p):
        __import__(module).main()
    return json.load(open(p, encoding="utf-8"))


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 7.5,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight",
                         "pdf.fonttype": 42})
    return plt


def _save(fig, name):
    from matplotlib.ticker import NullFormatter
    for ax in fig.axes:
        if ax.get_xscale() == "log":
            ax.xaxis.set_minor_formatter(NullFormatter())
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "%s.%s" % (name, ext)), dpi=200)


def _sigma_ticks(ax, ticks):
    ax.set_xticks(ticks)
    ax.set_xticklabels([("%g" % t) for t in ticks])


def fig1(plt, c2):
    """The signal: u along M0 trajectories (CIFAR-10 pilot, 50 steps, 256 points)."""
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for net, col in (("VP", C_VP), ("VE", C_VE)):
        p = c2["pilot"][net]
        ax.loglog(p["sigma"], p["u"], "o-", ms=2.5, lw=1.1, color=col, label="CIFAR-10 %s network" % net)
    ax.axvspan(10, 80, color="#dddddd", alpha=0.5, lw=0)
    ax.invert_xaxis()
    ax.set_xlabel(r"noise level $\sigma$ (sampling runs from left to right)")
    ax.set_ylabel(r"$u=\mathrm{median}\,|D_G-D_\mathrm{net}|^2/d$")
    ax.text(0.03, 0.06, "fog: $D_G$ is nearly exact,\n$u$ shows the network's own error\n(smooth for VP, flickering for VE)",
            transform=ax.transAxes, fontsize=7.5, va="bottom")
    ax.text(0.97, 0.94, "fog lifts: $u$ rises,\nalmost the same curve\nfor both networks", transform=ax.transAxes,
            fontsize=7.5, va="top", ha="right")
    ax.legend(loc="center left", frameon=False)
    _save(fig, "fig1_signal")
    plt.close(fig)


def fig2(plt):
    """Pipeline of the study (synthetic worlds -> CIFAR-10 -> AFHQv2), with what failed and what was changed."""
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(17, 45)
    ax.axis("off")
    boxes = [
        (1, 24, 22, 20, "Synthetic worlds\n(lab B: B18-B22)",
         "D (in-sample), W, W2, W3\nB21: statistic u, rule U_min\nB22: flicker-robust U_env\n(D only; fails in W3)"),
        (26, 24, 22, 20, "CIFAR-10 32x32\n(C1, C2; 18 steps)",
         "C1: normalized gap, VP\nHC1-3 supported, HC4 not\nC2: U_min 2.38\nVP HC7 yes, HC8 no\nVE HC5, HC6 no (flicker)"),
        (51, 24, 22, 20, "AFHQv2 64x64\n(C3; 40 steps)",
         "two unseen networks\nVP and VE\nU_env and U_min 2.38\nHC9-HC16: all 8\nsupported"),
        (76, 24, 23, 20, "AFHQv2 replication\n(C6; seed sets 2, 3)",
         "15,000 images per method\nfiner grid for VE\nHC17-HC25: all 9\nsupported\n(new seeds alone too)"),
    ]
    for x, y, w, h, title, body in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fc="#f4f4f4", ec="#333333", lw=0.8))
        ax.text(x + w / 2, y + h - 1.2, title, ha="center", va="top", fontsize=7.6, weight="bold")
        ax.text(x + w / 2, y + h - 8.2, body, ha="center", va="top", fontsize=6.6)
    for x in (23, 48, 73):
        ax.annotate("", xy=(x + 3, 34), xytext=(x, 34), arrowprops=dict(arrowstyle="->", lw=1.1))
    notes = [
        (12, "constants fixed on\nsynthetic worlds only\n(U_env: D only;\n2.38: four worlds)"),
        (37, "failure traced to a\nnetwork-error spike at\nsigma = 40.79 (VE)\n-> motivates U_env"),
        (62, "pre-registered;\nnobody had seen the\nrule-side u curves"),
        (87.5, "seed set 0 was seen\nbefore registration;\nseeds 2, 3 were not"),
    ]
    for x, t in notes:
        ax.text(x, 21, t, ha="center", va="top", fontsize=6.4, color="#444444")
    _save(fig, "fig2_pipeline")
    plt.close(fig)


def fig3(plt, c2):
    """The C2 failure: the CIFAR-10 VE network flickers at sigma = 40.79 and U_min 2.38 switches too early."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    for net, col in (("vp", C_VP), ("ve", C_VE)):
        for j, r in enumerate(c2["per_batch"]["%s_7.35" % net]):
            ax.loglog(r["sigma"], r["u"], ".-", ms=2, lw=0.6, color=col, alpha=0.35,
                      label=("CIFAR-10 %s, 10 batches" % net.upper()) if j == 0 else None)
        for j, r in enumerate(c2["per_batch"]["%s_2.38" % net]):
            ax.plot([r["switch_sigma"]], [r["switch_u"]], "x", color=col, ms=6,
                    label=("U_min 2.38 switch (%s)" % net.upper()) if j == 0 else None)
    ax.axvline(40.786, color=C_VE, ls="--", lw=0.8, alpha=0.6)
    ax.axvline(8.40, color="#2a8a3a", lw=1.5, alpha=0.5, label="best teleport (grid)")
    ax.invert_xaxis()
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$u$ (5% subset median)")
    ax.set_title("(a) u on 18-step runs")
    ax.legend(frameon=False, fontsize=6.3)
    ax = axes[1]
    for net, col in (("vp", C_VP), ("ve", C_VE)):
        for j, r in enumerate(c2["per_batch"]["%s_7.35" % net]):
            u = np.asarray(r["u"])
            ratio = u[1:] / np.minimum.accumulate(u)[:-1]
            ax.semilogx(r["sigma"][1:], ratio, ".-", ms=2, lw=0.6, color=col, alpha=0.4,
                        label=("%s" % net.upper()) if j == 0 else None)
    ax.axhline(2.38, color="k", ls="--", lw=0.8, label="2.38")
    ax.invert_xaxis()
    ax.set_ylim(0, 8)
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$u_c/\min(u_0,\dots,u_{c-1})$")
    ax.set_title("(b) U_min statistic: the VE spike crosses 2.38")
    for a in axes:
        _sigma_ticks(a, [80, 40, 20, 10, 5, 2, 1])
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "fig3_cifar_ve_failure")
    plt.close(fig)


def fig4(plt, c6):
    """AFHQv2: u traces for three seed sets (15 batches per network) and the switches of the two rules."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    cols = {0: "#999999", 2: "#4f8fcf", 3: "#e08a3c"}
    kb = {"vp": 9.7232, "ve": 19.9456}
    for ax, net in zip(axes, ("vp", "ve")):
        tr = c6["nets"][net]["seed_traces"]
        for s in ("0", "2", "3"):
            for j, p in enumerate(tr[s]):
                ax.loglog(p["sigma"], p["u"], ".-", ms=2, lw=0.6, color=cols[int(s)], alpha=0.55,
                          label=("seed set %s" % s) if j == 0 else None)
        for s in ("0", "2", "3"):
            for j, p in enumerate(tr[s]):
                i_min, i_env = p["switch_call_min"], p["switch_call_env"]  # trace index = call (T1: every call evaluated)
                ax.plot(p["sigma"][i_min], p["u"][i_min], "x", color=C_MIN, ms=5,
                        label="U_min 2.38 switch" if (s == "0" and j == 0) else None)
                ax.plot(p["sigma"][i_env], p["u"][i_env], "+", color=C_ENV, ms=6, mew=1.2,
                        label="U_env switch" if (s == "0" and j == 0) else None)
        ax.axvline(kb[net], color="#2a8a3a", lw=1.5, alpha=0.5, label="best coarse-grid teleport")
        ax.axvline(27.84, color="#777777", ls=":", lw=0.8)
        ax.invert_xaxis()
        ax.set_xlabel(r"$\sigma$")
        ax.set_title("(%s) AFHQv2 64x64, %s network" % ("a" if net == "vp" else "b", net.upper()))
        _sigma_ticks(ax, [80, 60, 40, 30, 20, 15, 10])
    axes[0].set_ylabel(r"$u$ (5% subset median)")
    axes[0].legend(frameon=False, fontsize=6.3, loc="upper left")
    fig.tight_layout()
    _save(fig, "fig4_afhq_traces")
    plt.close(fig)


def fig5(plt, c6):
    """FID against cost on seed set 0 (5,000 images): coarse and fine teleport grids and the two rules."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for ax, net in zip(axes, ("vp", "ve")):
        s = json.load(open(os.path.join(REC, "C3_results", net, "report", "summary_C3.json"), encoding="utf-8"))["results"]
        g = sorted((r["cost_ratio_mean"], r["fid"], r["k"]) for r in s.values() if r["method"] == "WV_tele")
        g = [x for x in g if x[1] < 4.6]
        ax.plot([a for a, b, k in g], [b for a, b, k in g], "s-", color=C_GRID, ms=4, lw=1, label="teleport grid (coarse)")
        for a, b, k in g:
            ax.annotate("k=%d" % k, (a, b), textcoords="offset points", xytext=(3, 3), fontsize=6)
        m0 = s["M0_N040_s0"]["fid"]
        if net == "ve":
            B = c6["B"]["grid"]
            fx = sorted((v["cost"], v["fid"], int(k)) for k, v in B.items())
            ax.plot([a for a, b, k in fx], [b for a, b, k in fx], "o--", color="#2a8a3a", ms=4, lw=1,
                    label="fine grid (C6, k=10-12)")
            for a, b, k in fx:
                ax.annotate("k=%d" % k, (a, b), textcoords="offset points", xytext=(3, -9), fontsize=6, color="#2a8a3a")
        ax.plot([s["U_env_N040_s0"]["cost_ratio_mean"]], [s["U_env_N040_s0"]["fid"]], "*", color=C_ENV, ms=11, label="U_env")
        ax.plot([s["U_min_c2.38_N040_s0"]["cost_ratio_mean"]], [s["U_min_c2.38_N040_s0"]["fid"]], "*", color=C_MIN, ms=11,
                mfc="none", mew=1.2, label="U_min 2.38")
        if net == "vp":
            k13 = [a for a, b, k in g if k == 13][0]
            ax.axvline(k13 + 0.10, color="#2a8a3a", ls="--", lw=0.8, label="grid best + 0.10 (cost limit)")
        ax.axhline(1.03 * m0, color=C_TOL, ls=":", lw=1, label=r"$1.03\times$FID(M0)")
        ax.axhline(m0, color="k", lw=0.6, alpha=0.4)
        ax.text(0.80, m0, "  M0 (cost 1)", fontsize=6, va="bottom", color="#333333")
        ax.set_xlim(0.64, 0.82)
        ax.set_ylim(m0 - 0.12, 1.03 * m0 + 0.16)
        ax.set_xlabel("network evaluations / M0")
        ax.set_title("(%s) %s network, seed set 0" % ("a" if net == "vp" else "b", net.upper()))
    axes[0].set_ylabel("FID (5,000 images)")
    axes[1].legend(frameon=False, fontsize=6.3, loc="upper right")
    axes[0].legend(frameon=False, fontsize=6.3, loc="upper left")
    fig.tight_layout()
    _save(fig, "fig5_fid_cost")
    plt.close(fig)


def fig6(plt, c6):
    """FID against 1/N: the relative 3% tolerance tightens with N while the rule-M0 gap stays."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)
    x = np.array([1 / 5000, 1 / 10000, 1 / 15000])
    xx = np.linspace(0, 1 / 4500, 50)
    for ax, net in zip(axes, ("vp", "ve")):
        e = c6["nets"][net]["fid_vs_n"]
        for name, col, mk in (("M0", "k", "o"), ("U_env", C_ENV, "s"), ("U_min", C_MIN, "^")):
            y = [e[name]["fid_5k_mean"], e[name]["fid_10k"], e[name]["fid_15k"]]
            ax.plot(x * 1e4, y, mk, color=col, ms=4.5, label=name)
            ax.plot(xx * 1e4, e[name]["a_inf"] + e[name]["b"] * xx, "-", color=col, lw=0.7, alpha=0.6)
        ax.plot(xx * 1e4, 1.03 * (e["M0"]["a_inf"] + e["M0"]["b"] * xx), ":", color=C_TOL, lw=1.1,
                label=r"$1.03\times$FID(M0) (fit)")
        ax.set_xlim(0, 2.2)
        ax.set_xlabel(r"$10^4/N_{\mathrm{img}}$ ($N_{\mathrm{img}}$ = number of images)")
        ax.set_title("(%s) %s network" % ("a" if net == "vp" else "b", net.upper()))
    axes[0].set_ylabel("FID")
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout()
    _save(fig, "fig6_fid_vs_n")
    plt.close(fig)


def main():
    plt = _style()
    c2 = _load("c2", "analyze_c2")
    c6 = _load("c6", "analyze_c6")
    fig1(plt, c2)
    fig2(plt)
    fig3(plt, c2)
    fig4(plt, c6)
    fig5(plt, c6)
    fig6(plt, c6)
    return sorted(f for f in os.listdir(OUT) if f.startswith("fig"))


if __name__ == "__main__":
    print("\n".join(main()))
