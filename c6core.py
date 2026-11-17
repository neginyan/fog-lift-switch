# c6core.py -- 実験場C 段階C6：本番（C3）の結果を強める追加の試験（MIT License, (c) 2026 Taishi Namba）
#   A：種の組 2・3 を足す（二つの網、M0・WV_tele(k_best)・U_min 2.38・U_env）
#   B：VE 型の格子を細かくする（WV_tele の k = 10・11・12、種の組 0）
# 規則と走らせ方は c3core.py（版 C3-1.0、C3 の本番で Colab が走らせたもの）を一字も変えずに import して使う。
# C3 の結果（C3_results/）は読むだけで、書き換えない。データと積率も C3 のものを読むだけ。

import json
import math
import os

import numpy as np

import c1core as C1
import c3core as C3

C6_VERSION = "C6-1.0"

PREREG6 = {
    "base_code": "c3core.py C3-1.0 (unchanged)",
    "nets": ["vp", "ve"],
    "steps": 40,
    "n_images_per_seedset": 5000,
    "batch": 1000,
    "seedset_c3_main": 0,
    "seedset_c3_noise": 1,
    "seedsets_new": [2, 3],
    "seed_rule": "latents of batch b in seed set s: torch.Generator(cuda).manual_seed(1000003*s + b), b = 0..4; "
                 "seed sets 0..3 never share a seed because b < 1000003",
    "k_best_c3": {"vp": 13, "ve": 9},
    "grid_rule_A": "k_best is fixed from C3 (seed set 0) and not searched again; WV_tele(k_best) is run on each new seed set "
                   "so that the grid comparison uses FIDs from the same seed set",
    "jobs_A_per_seedset": ["M0", "WV_tele(k_best)", "U_min 2.38", "U_env"],
    "jobs_B": {"net": "ve", "seedset": 0, "k": [10, 11, 12]},
    "fid_rel_tol": 0.03,
    "cost_max": 0.85,
    "grid_fid_rel": 0.03,
    "grid_cost_margin": 0.10,
    "pooled_main": [0, 2, 3],
    "pooled_new_only": [2, 3],
    "hc_pooled_main": {"vp": {"U_env": ["HC17", "HC18"], "U_min": ["HC21", "HC22"]},
                       "ve": {"U_env": ["HC19", "HC20"], "U_min": ["HC23", "HC24"]}},
    "hc_B": "HC25",
}

NB = 5
SIG = C3.sigmas3()


def tag(method, s, k=0):
    if method == "U_min":
        return C3.tag3({"method": "U_min", "c_min": 2.38, "steps": 40, "seedset": s})
    if method == "U_env":
        return C3.tag3({"method": "B22", "steps": 40, "seedset": s})
    return C3.tag3({"method": method, "k": k, "steps": 40, "seedset": s})


def jobs_A(net, P=PREREG6):
    kb = P["k_best_c3"][net]
    J = []
    for s in P["seedsets_new"]:
        J += [{"method": "M0", "k": 0, "steps": 40, "seedset": s},
              {"method": "WV_tele", "k": kb, "steps": 40, "seedset": s},
              {"method": "U_min", "k": 0, "c_min": 2.38, "steps": 40, "seedset": s},
              {"method": "B22", "k": 0, "steps": 40, "seedset": s}]
    return J


def jobs_B(P=PREREG6):
    b = P["jobs_B"]
    return [{"method": "WV_tele", "k": k, "steps": 40, "seedset": b["seedset"]} for k in b["k"]]


def cost_of(method, k=0):
    if method == "WV_tele":
        return (2 * (40 - k) - 1) / 79
    return 1.0


def estimate_hours(net, sec_m0_batch, P=PREREG6, rule_cost=(0.75, 0.75), sec_rule_batch=None):
    """L4 の実測（M0 1 束の秒）から、A と B の時間を見積もる。規則は C3 の実測（1 束 508〜517 秒）を使う。"""
    kb = P["k_best_c3"][net]
    rule = sec_rule_batch if sec_rule_batch else 515.0
    per_seed = NB * (sec_m0_batch + cost_of("WV_tele", kb) * sec_m0_batch + 2 * rule) / 3600
    A = per_seed * len(P["seedsets_new"])
    B = NB * sum(cost_of("WV_tele", k) * sec_m0_batch for k in P["jobs_B"]["k"]) / 3600 if net == "ve" else 0.0
    return {"A_hours": A, "B_hours": B}


# ---------------------------------------------------------------------------
# 特徴量と FID（種の組 0 は C3 の結果から読み、2・3 は C6 の結果から読む）
# ---------------------------------------------------------------------------
def feats_path(root3, root6, net, method, s, k=0, n=5000):
    base = root3 if s in (0, 1) and (method != "WV_tele" or k in (9, 13, 16, 20, 22, 25)) else root6
    return os.path.join(C3.net_dir(base, net), "runs", tag(method, s, k), "feats_%d.npy" % n)


def load_feats(root3, root6, net, method, seedsets, k=0):
    fs = []
    for s in seedsets:
        p = feats_path(root3, root6, net, method, s, k)
        if not os.path.exists(p):
            return None
        fs.append(np.load(p))
    return np.concatenate(fs, 0)


def fid_pooled(root3, root6, net, method, seedsets, ref, k=0):
    f = load_feats(root3, root6, net, method, seedsets, k)
    if f is None:
        return None
    mu, sg = C1.feat_stats(f)
    return C1.fid_from_stats(mu, sg, ref[0], ref[1])


def cost_pooled(root3, root6, net, method, seedsets, k=0):
    cs = []
    for s in seedsets:
        base = root3 if s in (0, 1) else root6
        d = os.path.join(C3.net_dir(base, net), "runs", tag(method, s, k))
        for b in range(NB):
            p = os.path.join(d, "b%03d.json" % b)
            if not os.path.exists(p):
                return None
            with open(p, encoding="utf-8") as f:
                cs.append(json.load(f)["cost_ratio"])
    return float(np.mean(cs))


def judge_one(fid_rule, cost_rule, fid_m0, fid_grid, cost_grid, P=PREREG6):
    """C3 と同じ形の二つの判定。"""
    if None in (fid_rule, cost_rule, fid_m0):
        return {"a": "未判定", "b": "未判定"}
    a = "支持" if (fid_rule <= (1 + P["fid_rel_tol"]) * fid_m0 and cost_rule <= P["cost_max"]) else "不支持"
    if None in (fid_grid, cost_grid):
        b = "未判定"
    else:
        b = "支持" if (fid_rule <= (1 + P["grid_fid_rel"]) * fid_grid and cost_rule <= cost_grid + P["grid_cost_margin"]) else "不支持"
    return {"a": a, "b": b, "fid_rule": fid_rule, "cost_rule": cost_rule, "fid_M0": fid_m0,
            "fid_tol": (1 + P["fid_rel_tol"]) * fid_m0, "fid_grid": fid_grid, "cost_grid": cost_grid,
            "margin_fid_to_tol": (1 + P["fid_rel_tol"]) * fid_m0 - fid_rule,
            "margin_cost_to_grid": None if cost_grid is None else cost_grid + P["grid_cost_margin"] - cost_rule}


def judge6(root3, root6, ref, P=PREREG6):
    out = {"version": C6_VERSION, "nets": {}}
    for net in P["nets"]:
        kb = P["k_best_c3"][net]
        o = {"per_seedset": {}, "pooled_main": {}, "pooled_new_only": {}}
        for s in [0] + P["seedsets_new"]:
            fm0 = fid_pooled(root3, root6, net, "M0", [s], ref)
            fg = fid_pooled(root3, root6, net, "WV_tele", [s], ref, kb)
            cg = cost_of("WV_tele", kb)
            o["per_seedset"][s] = {r: judge_one(fid_pooled(root3, root6, net, r, [s], ref),
                                                cost_pooled(root3, root6, net, r, [s]), fm0, fg, cg, P)
                                   for r in ("U_env", "U_min")}
        for key in ("pooled_main", "pooled_new_only"):
            ss = P[key]
            fm0 = fid_pooled(root3, root6, net, "M0", ss, ref)
            fg = fid_pooled(root3, root6, net, "WV_tele", ss, ref, kb)
            o[key] = {r: judge_one(fid_pooled(root3, root6, net, r, ss, ref), cost_pooled(root3, root6, net, r, ss),
                                   fm0, fg, cost_of("WV_tele", kb), P) for r in ("U_env", "U_min")}
            o[key]["n_images"] = 5000 * len(ss)
        hc = P["hc_pooled_main"][net]
        for r in ("U_env", "U_min"):
            o[hc[r][0]] = o["pooled_main"][r]["a"]
            o[hc[r][1]] = o["pooled_main"][r]["b"]
            per = [o["per_seedset"][s][r] for s in [0] + P["seedsets_new"]]
            o["all_three_%s_a" % r] = ("支持" if all(x["a"] == "支持" for x in per)
                                      else ("未判定" if any(x["a"] == "未判定" for x in per) else "不支持"))
            o["all_three_%s_b" % r] = ("支持" if all(x["b"] == "支持" for x in per)
                                      else ("未判定" if any(x["b"] == "未判定" for x in per) else "不支持"))
        # 種の揺れ（M0 の種の組 0〜3 の FID の幅）
        fm = [fid_pooled(root3, root6, net, "M0", [s], ref) for s in (0, 1, 2, 3)]
        o["fid_M0_by_seedset"] = fm
        if all(x is not None for x in fm):
            o["fid_M0_spread"] = max(fm) - min(fm)
        out["nets"][net] = o
    # B：VE 型の細かい格子（種の組 0）
    fm0 = fid_pooled(root3, root6, "ve", "M0", [0], ref)
    bb = {}
    for k in P["jobs_B"]["k"]:
        f = fid_pooled(root3, root6, "ve", "WV_tele", [0], ref, k)
        bb[k] = {"fid": f, "cost": cost_of("WV_tele", k), "sigma": float(SIG[k]),
                 "within": None if (f is None or fm0 is None) else bool(f <= (1 + P["fid_rel_tol"]) * fm0)}
    if any(v["within"] is None for v in bb.values()):
        hb = "未判定"
    else:
        hb = "支持" if any(v["within"] for v in bb.values()) else "不支持"
    safe = [k for k, v in bb.items() if v["within"]]
    out["B"] = {"grid": bb, "fid_M0_s0": fm0, "tol": None if fm0 is None else (1 + P["fid_rel_tol"]) * fm0,
                P["hc_B"]: hb, "k_best_fine": max(safe) if safe else None}
    return out


# ---------------------------------------------------------------------------
# そろったか（最後のセルは、そろうまで集計も図も出さない）
# ---------------------------------------------------------------------------
def all_done6(root3, root6, P=PREREG6, nets=None):
    nets = P["nets"] if nets is None else nets
    miss = []
    for net in nets:
        for j in jobs_A(net, P):
            m = {"B22": "U_env"}.get(j["method"], j["method"])
            p = feats_path(root3, root6, net, m, j["seedset"], j["k"])
            if not os.path.exists(p):
                miss.append("%s/%s" % (net, tag(m, j["seedset"], j["k"])))
    if "ve" in nets:
        for j in jobs_B(P):
            p = feats_path(root3, root6, "ve", "WV_tele", 0, j["k"])
            if not os.path.exists(p):
                miss.append("ve/%s" % tag("WV_tele", 0, j["k"]))
    return (not miss), miss


def figures6(root3, root6, J, out_dir, P=PREREG6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    files = []
    # 図 1：種の組ごとの u の列と切り替え（U_env・U_min）
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    cols = {0: "gray", 2: "C0", 3: "C1"}
    for ax, net in zip(axes, P["nets"]):
        for s in [0] + P["seedsets_new"]:
            base = root3 if s == 0 else root6
            for r, mk in (("U_min", "x"), ("U_env", "+")):
                d = os.path.join(C3.net_dir(base, net), "runs", tag(r, s))
                for b in range(NB):
                    p = os.path.join(d, "b%03d.json" % b)
                    if not os.path.exists(p):
                        continue
                    with open(p, encoding="utf-8") as f:
                        inf = json.load(f)
                    tr = np.array([[x[1], x[3]] for x in inf["trace"]], dtype=float)
                    if r == "U_min":
                        ax.loglog(tr[:, 0], tr[:, 1], ".-", color=cols[s], alpha=0.35, lw=0.8,
                                  label=("seed set %d" % s) if b == 0 else None)
                    ax.plot(tr[-1, 0], tr[-1, 1], mk, color="r" if r == "U_min" else "b", ms=8)
        ax.axvline(SIG[P["k_best_c3"][net]], color="C2", lw=2, alpha=0.5, label="k_best (C3)")
        ax.invert_xaxis()
        ax.set_xlabel("sigma")
        ax.set_title("%s: u traces; x = U_min switch, + = U_env switch" % net.upper())
        ax.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(out_dir, "figC6_1_u_by_seedset.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    files.append(p)
    # 図 2：種の組ごとの FID − FID(M0) と、合わせた値
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    for ax, net in zip(axes, P["nets"]):
        o = J["nets"][net]
        xs = [0] + P["seedsets_new"]
        for i, r in enumerate(("U_env", "U_min")):
            y = [o["per_seedset"][s][r].get("fid_rule", np.nan) - o["per_seedset"][s][r].get("fid_M0", np.nan) for s in xs]
            tol = [0.03 * o["per_seedset"][s][r].get("fid_M0", np.nan) for s in xs]
            ax.plot(np.arange(3) + 0.1 * i, y, "o", label=r)
            if i == 0:
                ax.plot(np.arange(3), tol, "k_", ms=20, label="tolerance (0.03 x FID(M0))")
            pm = o["pooled_main"][r]
            if pm.get("fid_rule") is not None:
                ax.plot(3.2 + 0.1 * i, pm["fid_rule"] - pm["fid_M0"], "s", label="%s pooled 0+2+3" % r)
        ax.set_xticks([0, 1, 2, 3.2])
        ax.set_xticklabels(["seed 0", "seed 2", "seed 3", "pooled"])
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_ylabel("FID(rule) - FID(M0)")
        ax.set_title(net.upper())
        ax.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(out_dir, "figC6_2_fid_by_seedset.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    files.append(p)
    return files


def report6(root3, root6, ref, extra=None, log=print, P=PREREG6):
    J = judge6(root3, root6, ref, P)
    figs = figures6(root3, root6, J, os.path.join(root6, "figures"), P)
    rep = os.path.join(root6, "report")
    os.makedirs(rep, exist_ok=True)
    summ = {"version": C6_VERSION, "prereg": P, "judgement": J, "figures": [os.path.basename(f) for f in figs],
            "extra": extra or {}}
    C1._atomic_json(os.path.join(rep, "summary_C6.json"), summ)
    for net in P["nets"]:
        hc = P["hc_pooled_main"][net]
        log("%s：%s" % (net, {k: J["nets"][net][k] for r in ("U_env", "U_min") for k in hc[r]}))
    log("B（%s）：%s" % (P["hc_B"], J["B"][P["hc_B"]]))
    return summ


def pack6(root6, name="C6_report", top="C6_results"):
    return C1.pack_report(root6, name=name, max_files=100, top=top)
