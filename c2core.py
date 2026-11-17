# c2core.py -- 実験場C 段階C2 の中身（MIT License, (c) 2026 Taishi Namba）
#
# 正規化しない合図 U_min を、学習済みの EDM（VP と VE の網）で試す。
#   u = 束の 5% の点での |D_G − D_net|² / d の中央値（呼び出しごと、時間の間引きなし = T1P5）
#   U_min（B21_rule_for_C.md §3 の定義どおり）：
#          k ≥ 1 で u_k ≥ C_MIN × min(u_0, …, u_{k−1}) となった最初の評価で、全部の点を網に切り替える
#          （1 回目の評価では切り替えない。後戻りしない。切り替える前は、全部の点を D_G で進める）
# c1core.py（C1 で Colab が走らせたものと一字一句同じ）を import し、手を入れない。

import json
import math
import os
import time

import numpy as np

import c1core as C1

C2_VERSION = "C2-1.1"

EDM_BASE = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/"

# ---------------------------------------------------------------------------
# 事前登録の定数（C2_事前登録.md の JSON と同じでなければならない。検証で照合する）
# ---------------------------------------------------------------------------
PREREG2 = {
    "c_min_main": 2.38,
    "c_min_sub": 7.35,
    "subset_frac": 0.05,
    "time_thin": 1,
    "steps": 18,
    "batch": 1000,
    "n_images": 10000,
    "seedset_main": 0,
    "seedset_noise": 1,
    "wv_sigma_targets": [40.0, 20.0, 10.0, 5.0, 2.5, 1.25, 0.6],
    "fid_rel_tol": 0.03,
    "cost_max": 0.85,
    "grid_fid_rel": 0.03,
    "grid_cost_margin": 0.10,
    "vp_model": "edm-cifar10-32x32-uncond-vp.pkl",
    "ve_model": "edm-cifar10-32x32-uncond-ve.pkl",
    "sampler_sigma_min": 0.002,
    "sampler_sigma_max": 80.0,
    "sampler_rho": 7.0,
    "replication_rel_tol": 1e-4,
    "rule_inequality": ">=",
    "pre_switch_update": "D_G for all points (B21)",
    "subset_count": "max(1, round(0.05 * n))",
}


# ---------------------------------------------------------------------------
# サンプラーの目盛り（EDM の edm_sampler と同じ決め方）
# ---------------------------------------------------------------------------
def sampler_range(net_sigma_min, net_sigma_max, P=PREREG2):
    """edm_sampler：sigma_min = max(0.002, net.sigma_min)、sigma_max = min(80, net.sigma_max)。"""
    return max(P["sampler_sigma_min"], float(net_sigma_min)), min(P["sampler_sigma_max"], float(net_sigma_max))


def sigmas_for(steps, srange=None, P=PREREG2):
    lo, hi = srange if srange is not None else (P["sampler_sigma_min"], P["sampler_sigma_max"])
    return C1.edm_sigmas(steps, sigma_min=lo, sigma_max=hi, rho=P["sampler_rho"])


# ---------------------------------------------------------------------------
# 規則 U_min
# ---------------------------------------------------------------------------
class UMinRule:
    """observe(u) が True を返したら、その評価から網に切り替える。

    B21_rule_for_C.md §3 の定義どおり（評価の番号 k = 0, 1, 2, …）。
    - k = 0（1 回目の評価）では切り替えない（最小値がまだない）。
    - k ≥ 1：u_k ≥ c_min × min(u_0, …, u_{k−1}) なら切り替え（等しい場合も切り替える）。
      （c_min ≥ 1 では、最小値に今の評価を含めても含めなくても同じ判定になる。）
    - 切り替えたら後戻りしない。
    """

    def __init__(self, c_min):
        self.c_min = float(c_min)
        self.min = None
        self.n_eval = 0
        self.switched = False
        self.min_at_switch = None

    def observe(self, u):
        if self.switched:
            return True
        u = float(u)
        self.n_eval += 1
        if self.min is not None and u >= self.c_min * self.min:
            self.switched = True
            self.min_at_switch = self.min
            return True
        self.min = u if self.min is None else min(self.min, u)
        return False


def replay_umin(us, c_min):
    """u の列に規則を当て、切り替えた評価の番号（0 始まり）を返す。切り替えなければ None。"""
    r = UMinRule(c_min)
    for i, u in enumerate(us):
        if r.observe(u):
            return i
    return None


def u_points(DG, Dn):
    """点ごとの |D_G − D_net|² / d（numpy）。"""
    d = int(np.prod(DG.shape[1:]))
    return C1.to_np(C1.rownorm2(DG - Dn)).astype(np.float64) / d


# ---------------------------------------------------------------------------
# 一つの束を U_min で走らせる（T1P5）
# ---------------------------------------------------------------------------
def run_umin(c_min, x_init, sigmas, net_fn, G, rng, frac=None, sync=None):
    frac = PREREG2["subset_frac"] if frac is None else frac
    n = int(x_init.shape[0])
    net = C1.CountingNet(net_fn)
    N = len(sigmas) - 1
    rule = UMinRule(c_min)
    m = max(1, int(round(frac * n)))  # B21：round(0.05 × 束の大きさ)、少なくとも 1 点
    trace = []
    state = {"switch_sigma": None, "switch_call": None}
    if sync:
        sync()
    t0 = time.time()

    def D_call(x, s, i, c):
        if rule.switched:
            return net(x, s)
        DG = G.denoise(x, s)
        sub = np.sort(rng.choice(n, size=m, replace=False))
        isub = C1._idx(sub, x)
        xs = x[isub]
        Dn_sub = net(xs, s)
        DGs = DG[isub]
        u = float(np.median(u_points(DGs, Dn_sub)))
        gap = float(np.median(C1.gap_stat_points(DGs, Dn_sub, xs)))
        prev_min = rule.min
        switch = rule.observe(u)
        trace.append([int(c), float(s), u, gap, prev_min])
        if not switch:
            # B21：切り替える前は、全部の点を D_G で進める（網を評価した 5% の点も）
            return DG
        out = C1._clone(DG)
        out[isub] = Dn_sub
        rest = np.setdiff1d(np.arange(n), sub)
        if len(rest) > 0:
            irest = C1._idx(rest, x)
            out[irest] = net(x[irest], s)
        state["switch_sigma"] = float(s)
        state["switch_call"] = int(c)
        return out

    x = C1.heun_sample(x_init, sigmas, D_call)
    if sync:
        sync()
    info = {"method": "U_min", "k": 0, "c_min": float(c_min), "n": n, "steps": N, "subset_m": m,
            "trace": trace, "trace_columns": ["call", "sigma", "u", "gap_stat", "min_before"],
            "switched": rule.switched,
            "switch_sigma": state["switch_sigma"] if rule.switched else 0.0,
            "switch_call": state["switch_call"], "min_at_switch": rule.min_at_switch,
            "seconds": time.time() - t0, "net_points": net.points}
    info["cost_units"] = net.points / n
    info["cost_ratio"] = info["cost_units"] / C1.nfe_m0(N)
    return x, info


# ---------------------------------------------------------------------------
# 仕事の名前と一覧
# ---------------------------------------------------------------------------
def tag(job):
    if job["method"] == "U_min":
        return "U_min_c%s_N%03d_s%d" % (("%g" % job["c_min"]), job["steps"], job["seedset"])
    return C1.method_tag(job)


def umin_jobs(cs=None, steps=None, seedset=None):
    cs = [PREREG2["c_min_main"], PREREG2["c_min_sub"]] if cs is None else cs
    steps = PREREG2["steps"] if steps is None else steps
    seedset = PREREG2["seedset_main"] if seedset is None else seedset
    return [{"method": "U_min", "k": 0, "c_min": c, "steps": steps, "seedset": seedset} for c in cs]


def ve_base_jobs(sigmas, steps=None):
    """規則に関わらない仕事（M0、WV_tele の格子、別の種の M0）。安い k から。"""
    steps = PREREG2["steps"] if steps is None else steps
    s0 = PREREG2["seedset_main"]
    ks = C1.wv_k_grid(sigmas, PREREG2["wv_sigma_targets"])
    jobs = [{"method": "M0", "k": 0, "steps": steps, "seedset": s0}]
    jobs += [{"method": "WV_tele", "k": k, "steps": steps, "seedset": s0} for k in sorted(ks, reverse=True)]
    jobs += [{"method": "M0", "k": 0, "steps": steps, "seedset": PREREG2["seedset_noise"]}]
    return jobs


def ve_rule_jobs(steps=None):
    """規則の仕事：主の U_min（2.38）、副の U_min（7.35）、S_gap（C1 の形）。"""
    steps = PREREG2["steps"] if steps is None else steps
    s0 = PREREG2["seedset_main"]
    jobs = umin_jobs(steps=steps)
    jobs.append({"method": "S_gap", "k": 0, "steps": steps, "seedset": s0})
    return jobs


# ---------------------------------------------------------------------------
# 走らせる（1 束 1 ファイル、再開できる）
# ---------------------------------------------------------------------------
def phase_runs2(out, jobs, net_fn, G, make_latents, srange=None, branch=None, n_images=None, batch=None,
                img_shape=(3, 32, 32), sync=None, log=print):
    n_images = PREREG2["n_images"] if n_images is None else n_images
    batch = PREREG2["batch"] if batch is None else batch
    nb = int(math.ceil(n_images / batch))
    for job in jobs:
        tg = tag(job)
        d = os.path.join(out, "runs", tg)
        os.makedirs(d, exist_ok=True)
        sig = sigmas_for(job["steps"], srange)
        for b in range(nb):
            fi = os.path.join(d, "b%03d.npz" % b)
            fj = os.path.join(d, "b%03d.json" % b)
            if os.path.exists(fi) and os.path.exists(fj):
                continue
            n = min(batch, n_images - b * batch)
            x0 = make_latents(job["seedset"], b, n) * float(sig[0])
            rng = C1.subset_rng(job["seedset"], job["steps"], b)
            if job["method"] == "U_min":
                x, info = run_umin(job["c_min"], x0, sig, net_fn, G, rng, sync=sync)
            else:
                if job["method"] == "S_gap" and branch is None:
                    raise RuntimeError("S_gap には枝（パイロットの判定）が要ります")
                x, info = C1.run_one(job["method"], job["k"], x0, sig, net_fn, G, rng=rng,
                                     branch=branch or "fog", sync=sync)
            info.update({"tag": tg, "batch": b, "seedset": job["seedset"], "version": C2_VERSION,
                         "sigma_range": [float(sig[-2]), float(sig[0])]})
            C1._atomic_npz(fi, images=C1.to_uint8_images(x, img_shape))
            C1._atomic_json(fj, info)
            msg = "%s b%03d  費用 %.3f × M0  %.1f 秒" % (tg, b, info["cost_ratio"], info["seconds"])
            if "switch_sigma" in info and job["method"] in ("U_min", "S_gap"):
                msg += "  切り替え σ = %.4g" % info["switch_sigma"]
            log(msg)
            with open(os.path.join(out, "log.txt"), "a", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def phase_pilot2(out, net_fn, G, make_latents, srange=None, log=print):
    """C1 と同じパイロット（50 歩、256 枚、種の組 9）。S_gap の枝を C1 の規則で決める。"""
    P = C1.PREREG
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, "pilot.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)["decision"]
    sig = sigmas_for(P["pilot_steps"], srange)
    x0 = make_latents(P["pilot_seedset"], 0, P["pilot_n"]) * float(sig[0])
    rows = C1.pilot_curve(x0, sig, net_fn, G)
    dec = C1.decide_branch(rows)
    C1._atomic_json(p, {"version": C2_VERSION, "steps": P["pilot_steps"], "n": P["pilot_n"],
                        "seedset": P["pilot_seedset"], "rows": rows, "decision": dec,
                        "sigma_range": [float(sig[-2]), float(sig[0])]})
    log("パイロット（S_gap の枝）：%s（%s）" % (dec["branch"], dec["reason"]))
    return dec


# ---------------------------------------------------------------------------
# 集計
# ---------------------------------------------------------------------------
def collect2(out, ref=None, n_images=None, batch=None):
    n_images = PREREG2["n_images"] if n_images is None else n_images
    batch = PREREG2["batch"] if batch is None else batch
    res = C1.collect(out, n_images=n_images, batch=batch, ref=ref)
    for tg, r in res.items():
        if r["method"] == "U_min":
            with open(os.path.join(out, "runs", tg, "b000.json"), encoding="utf-8") as f:
                r["c_min"] = json.load(f)["c_min"]
            r["switched_all"] = r["switch_sigmas"] and all(s > 0 for s in r["switch_sigmas"])
    return res


def judge_rule(res, rule_tag, steps=None, seedset=None, P=PREREG2, srange=None):
    """規則（U_min など）の二つの判定。
    a：FID ≤ 1.03 × FID(M0)、かつ 費用 ≤ 0.85 × M0
    b：WV_tele の格子で「画質を落とさない」最も安い k（なければ M0）と比べ、
       FID ≤ 1.03 × FID(WV_tele(k_best))、かつ 費用 ≤ 費用(k_best) + 0.10
    """
    steps = P["steps"] if steps is None else steps
    seedset = P["seedset_main"] if seedset is None else seedset
    sig = sigmas_for(steps, srange)
    m0 = res.get(C1.method_tag({"method": "M0", "k": 0, "steps": steps, "seedset": seedset}))
    r = res.get(rule_tag)
    out = {"rule": rule_tag}
    if not m0 or "fid" not in m0 or not r or "fid" not in r or not r.get("complete"):
        out["status"] = "未判定（M0 か規則の FID がまだありません）"
        return out
    tol = (1 + P["fid_rel_tol"]) * m0["fid"]
    out.update({"fid_M0": m0["fid"], "fid_tol": tol, "fid_rule": r["fid"], "cost_rule": r["cost_ratio_mean"],
                "switch_sigmas": r.get("switch_sigmas")})
    out["a_fid_ok"] = r["fid"] <= tol
    out["a_cost_ok"] = r["cost_ratio_mean"] <= P["cost_max"]
    out["a"] = "支持" if (out["a_fid_ok"] and out["a_cost_ok"]) else "不支持"
    grid = sorted((v["k"], v["fid"], v["cost_ratio_mean"]) for v in res.values()
                  if v["method"] == "WV_tele" and v["steps"] == steps and v["seedset"] == seedset
                  and "fid" in v and v.get("complete"))
    out["grid_WV_tele"] = grid
    n_grid = len(C1.wv_k_grid(sig, P["wv_sigma_targets"]))
    if len(grid) < n_grid:
        out["b"] = "未判定（WV_tele の格子が %d/%d 本）" % (len(grid), n_grid)
    else:
        safe = [g for g in grid if g[1] <= tol]
        kb, fb, cb = max(safe, key=lambda z: z[0]) if safe else (0, m0["fid"], 1.0)
        out["WV_tele_best"] = {"k": kb, "fid": fb, "cost": cb, "sigma": float(sig[kb])}
        ok = r["fid"] <= (1 + P["grid_fid_rel"]) * fb and r["cost_ratio_mean"] <= cb + P["grid_cost_margin"]
        out["b_fid_ok"] = r["fid"] <= (1 + P["grid_fid_rel"]) * fb
        out["b_cost_ok"] = r["cost_ratio_mean"] <= cb + P["grid_cost_margin"]
        out["b"] = "支持" if ok else "不支持"
    noise = res.get(C1.method_tag({"method": "M0", "k": 0, "steps": steps, "seedset": P["seedset_noise"]}))
    if noise and "fid" in noise:
        out["fid_M0_other_seeds"] = noise["fid"]
        out["fid_seed_spread"] = abs(noise["fid"] - m0["fid"])
        out["rule_minus_M0"] = r["fid"] - m0["fid"]
        out["within_seed_spread"] = abs(r["fid"] - m0["fid"]) <= abs(noise["fid"] - m0["fid"])
    out["status"] = "判定済み"
    return out


def judge_all(res_vp, res_ve, srange_vp=None, srange_ve=None, P=PREREG2):
    main = tag(umin_jobs(cs=[P["c_min_main"]])[0])
    sub = tag(umin_jobs(cs=[P["c_min_sub"]])[0])
    sgap = C1.method_tag({"method": "S_gap", "k": 0, "steps": P["steps"], "seedset": P["seedset_main"]})
    jve = judge_rule(res_ve, main, srange=srange_ve)
    jvp = judge_rule(res_vp, main, srange=srange_vp)
    J = {"HC5": jve.get("a", jve["status"]), "HC6": jve.get("b", jve["status"]),
         "HC7": jvp.get("a", jvp["status"]), "HC8": jvp.get("b", jvp["status"]),
         "detail_VE_main": jve, "detail_VP_main": jvp,
         "obs_VE_sub": judge_rule(res_ve, sub, srange=srange_ve),
         "obs_VP_sub": judge_rule(res_vp, sub, srange=srange_vp),
         "obs_VE_S_gap": judge_rule(res_ve, sgap, srange=srange_ve),
         "obs_VP_S_gap_C1": judge_rule(res_vp, sgap, srange=srange_vp)}
    return J


# ---------------------------------------------------------------------------
# 確かめ：C2 の U_min の最初の評価（呼び出し 0）が、C1 の S_gap の呼び出し 0 と同じか
# （出発点・5% の点・網・D_G が同じなら、呼び出し 0 の u は同じ。呼び出し 1 からは、
#   C1 の S_gap が 5% の点を網の出力で進め、U_min（B21）は D_G で進めるので、比べない）
# ---------------------------------------------------------------------------
def replication_vs_c1(c1_out, c2_out, steps=None, seedset=None, c_min=None):
    steps = PREREG2["steps"] if steps is None else steps
    seedset = PREREG2["seedset_main"] if seedset is None else seedset
    c_min = PREREG2["c_min_main"] if c_min is None else c_min
    d1 = os.path.join(c1_out, "runs", C1.method_tag({"method": "S_gap", "k": 0, "steps": steps, "seedset": seedset}))
    d2 = os.path.join(c2_out, "runs", tag({"method": "U_min", "c_min": c_min, "steps": steps, "seedset": seedset}))
    rows = []
    worst = 0.0
    n_cmp = 0
    if not (os.path.isdir(d1) and os.path.isdir(d2)):
        return {"status": "比べるものがありません", "rows": rows}
    for f in sorted(x for x in os.listdir(d2) if x.endswith(".json")):
        p1 = os.path.join(d1, f)
        if not os.path.exists(p1):
            continue
        with open(p1, encoding="utf-8") as g:
            a = json.load(g)
        with open(os.path.join(d2, f), encoding="utf-8") as g:
            b = json.load(g)
        lim = min(a["switch_call"] if a.get("switch_call") is not None else 10 ** 9,
                  b["switch_call"] if b.get("switch_call") is not None else 10 ** 9)
        ta = {int(r[0]): r for r in a["trace"] if len(r) >= 4}
        tb = {int(r[0]): r for r in b["trace"]}
        cmp_calls = [c for c in sorted(tb) if c < lim and c in ta and c == 0]
        rel = [abs(tb[c][2] - ta[c][3]) / max(abs(ta[c][3]), 1e-300) for c in cmp_calls]
        rel_gap = [abs(tb[c][3] - ta[c][2]) / max(abs(ta[c][2]), 1e-300) for c in cmp_calls]
        w = max(rel + rel_gap) if cmp_calls else float("nan")
        if cmp_calls:
            worst = max(worst, w)
            n_cmp += len(cmp_calls)
        rows.append({"batch": f, "calls_compared": cmp_calls, "max_rel_diff": w})
    ok = n_cmp > 0 and worst <= PREREG2["replication_rel_tol"]
    return {"status": "再現" if ok else ("比べる呼び出しなし" if n_cmp == 0 else "ずれあり"),
            "max_rel_diff": worst, "n_compared": n_cmp, "rows": rows}


# ---------------------------------------------------------------------------
# 図
# ---------------------------------------------------------------------------
def make_figures2(out_dir, outs, res_by, judged, P=PREREG2):
    """outs = {"VP": c2 の vp フォルダ, "VE": ve フォルダ}、res_by = {"VP": res, "VE": res}"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    files = []
    # 図 1：u の軌跡と切り替えの位置
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, name in zip(axes, ("VP", "VE")):
        o = outs.get(name)
        drawn = False
        for c, col in ((P["c_min_main"], "C3"), (P["c_min_sub"], "C0")):
            d = os.path.join(o, "runs", tag({"method": "U_min", "c_min": c, "steps": P["steps"],
                                               "seedset": P["seedset_main"]})) if o else None
            if not d or not os.path.isdir(d):
                continue
            for i, fj in enumerate(sorted(x for x in os.listdir(d) if x.endswith(".json"))):
                with open(os.path.join(d, fj), encoding="utf-8") as f:
                    inf = json.load(f)
                tr = np.array([r[:3] for r in inf["trace"]], dtype=float)
                if len(tr):
                    ax.loglog(tr[:, 1], tr[:, 2], ".-", color=col, alpha=0.35,
                              label=("u traces, C_MIN = %g" % c) if i == 0 else None)
                    drawn = True
                if inf["switch_sigma"] > 0:
                    ax.axvline(inf["switch_sigma"], color=col, alpha=0.25, ls="--")
        jd = judged.get("detail_%s_main" % name, {})
        if jd.get("WV_tele_best"):
            ax.axvline(jd["WV_tele_best"]["sigma"], color="C2", lw=2, label="sigma of best WV_tele")
        ax.set_title("%s: u = median |D_G - D_net|^2 / d (5%% subset); dashed = switch" % name)
        ax.set_xlabel("sigma")
        ax.invert_xaxis()
        if drawn:
            ax.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig1_u_traces_switch.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    files.append(p)
    # 図 2：FID と費用
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, name in zip(axes, ("VP", "VE")):
        res = res_by.get(name, {})
        g = sorted((r["cost_ratio_mean"], r["fid"]) for r in res.values()
                   if r["method"] == "WV_tele" and r["seedset"] == P["seedset_main"] and "fid" in r)
        if g:
            ax.plot([a for a, b in g], [b for a, b in g], "s-", color="gray", label="WV_tele(k)")
        for tg, r in res.items():
            if "fid" not in r or r["seedset"] != P["seedset_main"]:
                continue
            if r["method"] == "M0":
                ax.plot(r["cost_ratio_mean"], r["fid"], "ko", ms=9, label="M0")
            elif r["method"] == "S_gap":
                ax.plot(r["cost_ratio_mean"], r["fid"], "m^", ms=9, label="S_gap")
            elif r["method"] == "U_min":
                main = abs(r.get("c_min", 0) - P["c_min_main"]) < 1e-9
                ax.plot(r["cost_ratio_mean"], r["fid"], "r*" if main else "b*", ms=13,
                        label="U_min %g" % r.get("c_min", 0))
        jd = judged.get("detail_%s_main" % name, {})
        if jd.get("fid_tol"):
            ax.axhline(jd["fid_tol"], color="gray", ls=":", label="1.03 x FID(M0)")
        ax.axvline(P["cost_max"], color="gray", ls="--", lw=0.8)
        ax.set_xlabel("net evaluations / M0")
        ax.set_ylabel("FID (10k images)")
        ax.set_title(name)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig2_fid_vs_cost.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    files.append(p)
    return files


def phase_report2(root, c1_out, ref, srange_vp=None, srange_ve=None, extra=None, n_images=None, batch=None,
                  log=print):
    vp = os.path.join(root, "vp")
    ve = os.path.join(root, "ve")
    kw = {"n_images": n_images or PREREG2["n_images"], "batch": batch or PREREG2["batch"]}
    res_vp = {}
    if os.path.isdir(c1_out):
        res_vp.update(C1.collect(c1_out, ref=ref, **kw))
    if os.path.isdir(vp):
        res_vp.update(collect2(vp, ref=ref, **kw))
    res_ve = collect2(ve, ref=ref, **kw) if os.path.isdir(ve) else {}
    J = judge_all(res_vp, res_ve, srange_vp, srange_ve)
    rep = os.path.join(root, "report")
    J["replication_VP_vs_C1"] = replication_vs_c1(c1_out, vp)
    figs = make_figures2(os.path.join(root, "figures"), {"VP": vp, "VE": ve}, {"VP": res_vp, "VE": res_ve}, J)
    summ = {"version": C2_VERSION, "prereg": PREREG2, "results_VP": res_vp, "results_VE": res_ve,
            "judgement": J, "figures": [os.path.basename(f) for f in figs], "extra": extra or {}}
    os.makedirs(rep, exist_ok=True)
    C1._atomic_json(os.path.join(rep, "summary_C2.json"), summ)
    for k in ("HC5", "HC6", "HC7", "HC8"):
        log("%s：%s" % (k, J[k]))
    log("VP の再現（C1 の S_gap の軌跡と）：%s" % J["replication_VP_vs_C1"]["status"])
    return summ


def estimate_minutes(cost_units_list, sec_per_m0_batch=232.0, n_batches=10, factor=1.0):
    """網の評価（M0 = 1 の単位）の合計から、時間（分）を見積もる。"""
    return float(sum(cost_units_list)) * sec_per_m0_batch * n_batches * factor / 60.0


def pack_report2(root, name="C2_report", max_files=100, top="C2_results"):
    return C1.pack_report(root, name=name, max_files=max_files, top=top)
