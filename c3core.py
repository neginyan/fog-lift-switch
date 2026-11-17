# c3core.py -- 実験場C 段階C3 の骨組み（MIT License, (c) 2026 Taishi Namba）
#
# まだ誰も u の曲線を見ていない網（EDM の AFHQv2 64×64 の VP 型と VE 型。網ごとに別に判定）で、
#   M0、WV_tele の格子（大きい k から、画質を落とさない最初の k で止める）、
#   B21 の U_min（2.38、比較）、B22 の規則（差し込み口）
# を一度だけ走らせるための道具。c1core.py と c2core.py（C1・C2 で Colab が走らせたもの）は import し、手を入れない。
#
# B22 の規則は、定義書 `B22_rule_for_C.md` が届いてから install_b22() で差し込む。
# 差し込むまでは、B22 の仕事を走らせようとすると止まる。

import json
import math
import os
import time

import numpy as np

import c1core as C1
import c2core as C2

C3_VERSION = "C3-1.0"

# ---------------------------------------------------------------------------
# 事前登録の定数（下書き。B22 の欄は定義書が届いてから書き入れる。確定のときに C3_事前登録.md と照合する）
# ---------------------------------------------------------------------------
PREREG3 = {
    "models": {"vp": "edm-afhqv2-64x64-uncond-vp.pkl", "ve": "edm-afhqv2-64x64-uncond-ve.pkl"},
    "hc_names": {"vp": {"U_env": ["HC9", "HC10"], "U_min": ["HC13", "HC14"]},
                 "ve": {"U_env": ["HC11", "HC12"], "U_min": ["HC15", "HC16"]}},
    "gpu": "NVIDIA L4 (both nets)",
    "dataset_n_images": 15803,
    "results_layout": "C3_results/{vp,ve}/ (net-specific); data and moments shared in C3_results/",
    "fid_ref": "afhqv2-64x64.npz",
    "dataset": "afhqv2-64x64 (StarGAN v2 download.sh afhq-v2-dataset → EDM dataset_tool.py --resolution=64x64)",
    "img_shape": [3, 64, 64],
    "steps": 40,
    "sampler_sigma_min": 0.002,
    "sampler_sigma_max": 80.0,
    "sampler_rho": 7.0,
    "batch": 1000,
    "n_images": 5000,
    "seedset_main": 0,
    "seedset_noise": 1,
    "wv_sigma_targets": [40.0, 20.0, 10.0, 5.0, 2.5, 1.25, 0.6],
    "wv_grid_order": "descending k, stop at the first k with FID <= 1.03 x FID(M0)",
    "fid_rel_tol": 0.03,
    "cost_max": 0.85,
    "grid_fid_rel": 0.03,
    "grid_cost_margin": 0.10,
    "umin_c_min": 2.38,
    "subset_frac": 0.05,
    "time_thin": 1,
    "b22_rule": "U_env (B22_rule_for_C.md section 3): W_k = {i <= k : ln sigma_i <= ln sigma_k + DELTA}, extended to the last M_MIN evaluations; "
                "X_i = max(u over W_i); P_k = min(X_0..X_{a_k - 1}); switch at the first k with a_k >= 1 and min(u over W_k) >= C x P_k; "
                "u = median of u_i over finite points; T1P5; before the switch all points advance with D_G",
    "b22_constants": {"DELTA": 0.3, "C": 1.5150000000000001, "M_MIN": 3, "window_tol": 1e-12,
                      "source": "B22/results/rule_constants_B22.json (C = 1.01 x 1.5)"},
    "uenv_window_sizes_40": [1, 2, 3, 4, 5, 5, 6, 5, 6, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    "uenv_first_decidable_eval_40": 5,
}


def srange3(P=PREREG3):
    return (P["sampler_sigma_min"], P["sampler_sigma_max"])


def sigmas3(steps=None, srange=None, P=PREREG3):
    steps = P["steps"] if steps is None else steps
    lo, hi = srange if srange is not None else srange3(P)
    return C1.edm_sigmas(steps, sigma_min=lo, sigma_max=hi, rho=P["sampler_rho"])


def nfe(steps):
    return C1.nfe_m0(steps)


# ---------------------------------------------------------------------------
# 一般の「合図で切り替える」走らせ方（U_min も B22 もここを通る）
#   rule.observe(sigma, stat) -> True なら、その呼び出しから全部の点を網へ（後戻りしない）
#   stat_fn(DG_sub, Dnet_sub, x_sub, sigma) -> float（既定は B21 の u：|D_G − D_net|²/d の中央値）
#   time_thin = T：切り替える前は、呼び出しの番号が T の倍数のときだけ評価（B21 §4 と同じ数え方）
#   pre_switch = "DG"：切り替える前は全部の点を D_G で進める（B21）
# ---------------------------------------------------------------------------
def stat_u(DGs, Dns, xs, sigma):
    return float(np.median(C2.u_points(DGs, Dns)))


def stat_u_finite(DGs, Dns, xs, sigma):
    """B22 §2：網の出力が有限でない点は除いた u_i の中央値（有限な点がなければ +inf）。"""
    Dn = C1.to_np(Dns)
    fin = np.isfinite(Dn.reshape(Dn.shape[0], -1)).all(1)
    if not fin.any():
        return float("inf")
    if fin.all():
        return float(np.median(C2.u_points(DGs, Dns)))
    idx = C1._idx(np.where(fin)[0], DGs)
    return float(np.median(C2.u_points(DGs[idx], Dns[idx])))


class UEnvRule:
    """B22 の U_env（B22_rule_for_C.md §3 の定義どおり。オンライン）。

    評価 k（0 始まり）の ln σ_k は増えない列。
    1. 窓 W_k = { i ≤ k : ln σ_i ≤ ln σ_k + Δ }（B22 と同じく 1e-12 の余裕で含める）。3 個より少なければ直近 3 個に広げる。
       a_k = 窓の最初の番号。
    2. 上の包絡 X_i = max(u_j : j ∈ W_i)。
    3. 基準 P_k = min(X_0, …, X_{a_k − 1})。
    4. a_k ≥ 1 で min(u_j : j ∈ W_k) ≥ C × P_k となった最初の k で切り替える（後戻りしない）。
    """

    def __init__(self, delta, C, m_min=3, tol=1e-12):
        self.delta, self.C, self.m_min, self.tol = float(delta), float(C), int(m_min), float(tol)
        self.lns, self.neg, self.us, self.X, self.Pmin = [], [], [], [], []
        self.switched = False
        self.k_switch = None
        self.last = None

    def window_start(self, j):
        # B22（common_b22.window_start）と同じ計算：−ln σ の増える列で、−ln σ_k − Δ − tol 以上の最初の番号
        import bisect
        a = bisect.bisect_left(self.neg, self.neg[j] - self.delta - self.tol, 0, j + 1)
        return min(a, max(j - self.m_min + 1, 0))

    def observe(self, sigma, u):
        if self.switched:
            return True
        self.lns.append(math.log(float(sigma)))
        self.neg.append(-self.lns[-1])
        self.us.append(float(u))
        j = len(self.us) - 1
        if j >= 1 and self.lns[j] > self.lns[j - 1] + 1e-15:
            raise ValueError("U_env：ln σ は増えない列でなければなりません")
        a = self.window_start(j)
        w = self.us[a:j + 1]
        self.X.append(max(w))
        self.Pmin.append(self.X[-1] if not self.Pmin else min(self.Pmin[-1], self.X[-1]))
        lo = min(w)
        base = self.Pmin[a - 1] if a >= 1 else None
        self.last = {"k": j, "a": a, "window_n": j - a + 1, "min_w": lo, "P": base}
        if a >= 1 and lo >= self.C * base:
            self.switched = True
            self.k_switch = j
            return True
        return False

    def state(self):
        return {"n_eval": len(self.us), "k_switch": self.k_switch, "last": self.last}


def uenv_window_sizes(sigmas_eval, delta, m_min=3, tol=1e-12):
    """u を使わず、評価の σ の列だけから、各評価の窓の大きさと a_k を出す。"""
    r = UEnvRule(delta, 1.0, m_min, tol)
    out = []
    for s in sigmas_eval:
        r.lns.append(math.log(float(s)))
        r.neg.append(-r.lns[-1])
        j = len(r.lns) - 1
        a = r.window_start(j)
        out.append({"k": j, "sigma": float(s), "a": a, "window_n": j - a + 1, "decidable": a >= 1})
    return out


def eval_sigmas(sigmas, time_thin=1):
    """T1 で評価する呼び出しの σ（Heun：σ_0, σ_1, σ_1, σ_2, σ_2, …）。"""
    return [s for c, (_, s) in enumerate(C1.call_sigmas(sigmas)) if c % int(time_thin) == 0]


def install_uenv(P=PREREG3):
    """B22 の U_env を、事前登録の定数で差し込む。"""
    k = P["b22_constants"]
    install_b22("U_env", lambda: UEnvRule(k["DELTA"], k["C"], k["M_MIN"], k["window_tol"]),
                {"DELTA": k["DELTA"], "C": k["C"], "M_MIN": k["M_MIN"]}, stat_fn=stat_u_finite, time_thin=1,
                frac=P["subset_frac"], source="B22_rule_for_C.md")


class UMinAdapter:
    """B21 の U_min（c2core.UMinRule）を、observe(sigma, stat) の形にする。"""

    def __init__(self, c_min):
        self.r = C2.UMinRule(c_min)

    @property
    def switched(self):
        return self.r.switched

    def observe(self, sigma, stat):
        return self.r.observe(stat)

    def state(self):
        return {"min": self.r.min, "n_eval": self.r.n_eval, "min_at_switch": self.r.min_at_switch}


def run_signal_rule(rule, x_init, sigmas, net_fn, G, rng, stat_fn=stat_u, frac=None, time_thin=1,
                    pre_switch="DG", sync=None):
    frac = PREREG3["subset_frac"] if frac is None else frac
    assert pre_switch == "DG", "この骨組みは B21 の形（切り替える前は全部の点を D_G）だけを用意しています"
    n = int(x_init.shape[0])
    net = C1.CountingNet(net_fn)
    N = len(sigmas) - 1
    m = max(1, int(round(frac * n)))
    trace = []
    st = {"switch_sigma": None, "switch_call": None, "switched": False}
    if sync:
        sync()
    t0 = time.time()

    def D_call(x, s, i, c):
        if st["switched"]:
            return net(x, s)
        DG = G.denoise(x, s)
        if c % int(time_thin) != 0:
            return DG
        sub = np.sort(rng.choice(n, size=m, replace=False))
        isub = C1._idx(sub, x)
        xs = x[isub]
        Dn_sub = net(xs, s)
        DGs = DG[isub]
        val = float(stat_fn(DGs, Dn_sub, xs, float(s)))
        u = float(np.median(C2.u_points(DGs, Dn_sub)))
        sw = bool(rule.observe(float(s), val))
        trace.append([int(c), float(s), val, u])
        if not sw:
            return DG
        out = C1._clone(DG)
        out[isub] = Dn_sub
        rest = np.setdiff1d(np.arange(n), sub)
        if len(rest) > 0:
            irest = C1._idx(rest, x)
            out[irest] = net(x[irest], s)
        st.update(switched=True, switch_sigma=float(s), switch_call=int(c))
        return out

    x = C1.heun_sample(x_init, sigmas, D_call)
    if sync:
        sync()
    info = {"n": n, "steps": N, "subset_m": m, "time_thin": int(time_thin), "trace": trace,
            "trace_columns": ["call", "sigma", "stat", "u"], "switched": st["switched"],
            "switch_sigma": st["switch_sigma"] if st["switched"] else 0.0, "switch_call": st["switch_call"],
            "seconds": time.time() - t0, "net_points": net.points}
    if hasattr(rule, "state"):
        info["rule_state"] = rule.state()
    info["cost_units"] = net.points / n
    info["cost_ratio"] = info["cost_units"] / C1.nfe_m0(N)
    return x, info


# ---------------------------------------------------------------------------
# B22 の差し込み口
# ---------------------------------------------------------------------------
B22 = {"installed": False, "name": None, "make_rule": None, "stat_fn": stat_u, "time_thin": 1,
       "frac": None, "constants": None, "source": None}


def install_b22(name, make_rule, constants, stat_fn=None, time_thin=1, frac=None, source="B22_rule_for_C.md"):
    """定義書どおりの規則を差し込む。make_rule() は observe(sigma, stat) を持つ新しい規則を返す。"""
    B22.update(installed=True, name=name, make_rule=make_rule, constants=constants,
               stat_fn=stat_fn or stat_u, time_thin=int(time_thin), frac=frac, source=source)


def b22_status():
    return {k: (v if k not in ("make_rule", "stat_fn") else (v is not None)) for k, v in B22.items()}


# ---------------------------------------------------------------------------
# 仕事
# ---------------------------------------------------------------------------
def tag3(job):
    if job["method"] == "U_min":
        return C2.tag(job)
    if job["method"] == "B22":
        return "U_env_N%03d_s%d" % (job["steps"], job["seedset"])
    return C1.method_tag(job)


def wv_desc_ks(sigmas, P=PREREG3):
    return sorted(C1.wv_k_grid(sigmas, P["wv_sigma_targets"]), reverse=True)


def run_job3(job, x0, sig, net_fn, G, rng, sync=None):
    if job["method"] == "U_min":
        x, info = run_signal_rule(UMinAdapter(job["c_min"]), x0, sig, net_fn, G, rng, sync=sync)
        info.update({"method": "U_min", "k": 0, "c_min": job["c_min"]})
        return x, info
    if job["method"] == "B22":
        if not B22["installed"]:
            raise RuntimeError("B22 の規則がまだ差し込まれていません（定義書が届いてから install_b22）")
        x, info = run_signal_rule(B22["make_rule"](), x0, sig, net_fn, G, rng, stat_fn=B22["stat_fn"],
                                  frac=B22["frac"], time_thin=B22["time_thin"], sync=sync)
        info.update({"method": "B22", "k": 0, "b22_name": B22["name"], "b22_constants": B22["constants"]})
        return x, info
    return C1.run_one(job["method"], job["k"], x0, sig, net_fn, G, rng=rng, sync=sync)


def phase_runs3(out, jobs, net_fn, G, make_latents, srange=None, n_images=None, batch=None, img_shape=None,
                sync=None, log=print, show_switch=False):
    """1 束 1 ファイル。切り替えの σ は log.txt にだけ書き、画面には出さない（show_switch=False）。"""
    n_images = PREREG3["n_images"] if n_images is None else n_images
    batch = PREREG3["batch"] if batch is None else batch
    img_shape = tuple(PREREG3["img_shape"]) if img_shape is None else img_shape
    nb = int(math.ceil(n_images / batch))
    for job in jobs:
        tg = tag3(job)
        d = os.path.join(out, "runs", tg)
        os.makedirs(d, exist_ok=True)
        sig = sigmas3(job["steps"], srange)
        for b in range(nb):
            fi = os.path.join(d, "b%03d.npz" % b)
            fj = os.path.join(d, "b%03d.json" % b)
            if os.path.exists(fi) and os.path.exists(fj):
                continue
            n = min(batch, n_images - b * batch)
            x0 = make_latents(job["seedset"], b, n) * float(sig[0])
            rng = C1.subset_rng(job["seedset"], job["steps"], b)
            x, info = run_job3(job, x0, sig, net_fn, G, rng, sync=sync)
            info.update({"tag": tg, "batch": b, "seedset": job["seedset"], "version": C3_VERSION,
                         "sigma_range": [float(sig[-2]), float(sig[0])]})
            C1._atomic_npz(fi, images=C1.to_uint8_images(x, img_shape))
            C1._atomic_json(fj, info)
            msg = "%s b%03d  費用 %.3f × M0  %.1f 秒" % (tg, b, info["cost_ratio"], info["seconds"])
            full = msg + ("  切り替え σ = %.4g" % info["switch_sigma"] if "switch_sigma" in info else "")
            log(full if show_switch else msg)
            with open(os.path.join(out, "log.txt"), "a", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + full + "\n")


def fid_of(out, tg, ref, n_images=None):
    n_images = PREREG3["n_images"] if n_images is None else n_images
    fp = os.path.join(out, "runs", tg, "feats_%d.npy" % n_images)
    if not os.path.exists(fp):
        return None
    mu, sg = C1.feat_stats(np.load(fp))
    return C1.fid_from_stats(mu, sg, ref[0], ref[1])


def phase_grid_adaptive(out, net_fn, G, make_latents, feat_fn, ref, srange=None, n_images=None, batch=None,
                        img_shape=None, sync=None, log=print, P=PREREG3):
    """WV_tele を大きい k（安い方）から走らせ、FID ≤ 1.03 × FID(M0) になった最初の k で止める。
    M0（種の組 0）の FID が先に要る。返り値：走らせた k の一覧と、止まった k（なければ None）。"""
    n_images = P["n_images"] if n_images is None else n_images
    batch = P["batch"] if batch is None else batch
    steps, s0 = P["steps"], P["seedset_main"]
    m0 = tag3({"method": "M0", "k": 0, "steps": steps, "seedset": s0})
    f0 = fid_of(out, m0, ref, n_images)
    if f0 is None:
        raise RuntimeError("先に M0 の FID が要ります")
    tol = (1 + P["fid_rel_tol"]) * f0
    sig = sigmas3(steps, srange)
    done, stop = [], None
    for k in wv_desc_ks(sig, P):
        job = {"method": "WV_tele", "k": k, "steps": steps, "seedset": s0}
        phase_runs3(out, [job], net_fn, G, make_latents, srange=srange, n_images=n_images, batch=batch,
                    img_shape=img_shape, sync=sync, log=log)
        C1.phase_features(out, [tag3(job)], feat_fn, n_images=n_images, batch=batch, log=log)
        f = fid_of(out, tag3(job), ref, n_images)
        done.append(k)
        log("WV_tele k=%d（σ = %.3g）：FID %.4f（基準 %.4f）" % (k, sig[k], f, tol))
        if f <= tol:
            stop = k
            break
    C1._atomic_json(os.path.join(out, "grid_state.json"), {"done": done, "stop": stop, "tol": tol, "fid_M0": f0})
    return done, stop


# ---------------------------------------------------------------------------
# 判定（C2 と同じ形。格子は「大きい k から止めるまで」）
# ---------------------------------------------------------------------------
def grid_best(res, sig, steps, seedset, tol, f0, P=PREREG3):
    """走らせた格子から k_best（画質を落とさない最大の k）を決める。
    大きい k から順に、すべて走っていて、最初に基準内になった k が k_best。
    全部の k を走らせて基準内がなければ k_best = 0（M0）。それ以外は未判定。"""
    ks = wv_desc_ks(sig, P)
    have = {v["k"]: v for v in res.values() if v["method"] == "WV_tele" and v["steps"] == steps
            and v["seedset"] == seedset and "fid" in v and v.get("complete")}
    for k in ks:
        if k not in have:
            return None, "未判定（WV_tele k=%d の FID がまだありません）" % k
        if have[k]["fid"] <= tol:
            return {"k": k, "fid": have[k]["fid"], "cost": have[k]["cost_ratio_mean"], "sigma": float(sig[k])}, None
    return {"k": 0, "fid": f0, "cost": 1.0, "sigma": float(sig[0])}, None


def judge3(res, rule_tag, P=PREREG3, srange=None):
    steps, s0 = P["steps"], P["seedset_main"]
    sig = sigmas3(steps, srange)
    m0 = res.get(tag3({"method": "M0", "k": 0, "steps": steps, "seedset": s0}))
    r = res.get(rule_tag)
    out = {"rule": rule_tag}
    if not m0 or "fid" not in m0 or not r or "fid" not in r or not r.get("complete"):
        out["status"] = "未判定（M0 か規則の FID がまだありません）"
        out["a"] = out["b"] = out["status"]
        return out
    tol = (1 + P["fid_rel_tol"]) * m0["fid"]
    out.update({"fid_M0": m0["fid"], "fid_tol": tol, "fid_rule": r["fid"], "cost_rule": r["cost_ratio_mean"],
                "switch_sigmas": r.get("switch_sigmas")})
    out["a"] = "支持" if (r["fid"] <= tol and r["cost_ratio_mean"] <= P["cost_max"]) else "不支持"
    best, why = grid_best(res, sig, steps, s0, tol, m0["fid"], P)
    if best is None:
        out["b"] = why
    else:
        out["WV_tele_best"] = best
        ok = r["fid"] <= (1 + P["grid_fid_rel"]) * best["fid"] and r["cost_ratio_mean"] <= best["cost"] + P["grid_cost_margin"]
        out["b"] = "支持" if ok else "不支持"
    noise = res.get(tag3({"method": "M0", "k": 0, "steps": steps, "seedset": P["seedset_noise"]}))
    if noise and "fid" in noise:
        out["fid_M0_other_seeds"] = noise["fid"]
        out["fid_seed_spread"] = abs(noise["fid"] - m0["fid"])
        out["within_seed_spread"] = abs(r["fid"] - m0["fid"]) <= out["fid_seed_spread"]
    out["status"] = "判定済み"
    return out


def collect3(out, ref=None, n_images=None, batch=None, P=PREREG3):
    n_images = P["n_images"] if n_images is None else n_images
    batch = P["batch"] if batch is None else batch
    res = C1.collect(out, n_images=n_images, batch=batch, ref=ref)
    for tg, r in res.items():
        if r["method"] in ("U_min", "B22"):
            with open(os.path.join(out, "runs", tg, "b000.json"), encoding="utf-8") as f:
                j = json.load(f)
            r["c_min"] = j.get("c_min")
            r["b22_name"] = j.get("b22_name")
    return res


def judge_all3(res, srange=None, P=PREREG3, net=None):
    """二つの規則（U_env = B22、U_min 2.38 = B21）を、どちらも判定する。net を渡すと HC 番号を付ける。"""
    umin = tag3({"method": "U_min", "c_min": P["umin_c_min"], "steps": P["steps"], "seedset": P["seedset_main"]})
    b22 = tag3({"method": "B22", "steps": P["steps"], "seedset": P["seedset_main"]})
    jb = judge3(res, b22, P, srange)
    ju = judge3(res, umin, P, srange)
    # HC 番号は、B22 の定義書が届いて事前登録を確定するときに決める（下書きでは仮の名前）
    J = {"U_env_a": jb["a"], "U_env_b": jb["b"], "detail_B22": jb,
         "U_min_a": ju["a"], "U_min_b": ju["b"], "detail_U_min": ju}
    if net is not None:
        ea, eb = P["hc_names"][net]["U_env"]
        ma, mb = P["hc_names"][net]["U_min"]
        J.update({"net": net, ea: jb["a"], eb: jb["b"], ma: ju["a"], mb: ju["b"]})
    return J


def report3(out, ref, srange=None, n_images=None, batch=None, extra=None, log=print, P=PREREG3, net=None):
    res = collect3(out, ref=ref, n_images=n_images, batch=batch, P=P)
    J = judge_all3(res, srange, P, net=net)
    rep = os.path.join(out, "report")
    os.makedirs(rep, exist_ok=True)
    figs = figures3(out, res, J, P)
    summ = {"version": C3_VERSION, "prereg": P, "b22": b22_status(), "results": res, "judgement": J,
            "figures": [os.path.basename(f) for f in figs], "extra": extra or {}}
    C1._atomic_json(os.path.join(rep, "summary_C3.json"), summ)
    keys = (P["hc_names"][net]["U_env"] + P["hc_names"][net]["U_min"]) if net else ["U_env_a", "U_env_b", "U_min_a", "U_min_b"]
    for k in keys:
        log("%s%s：%s" % ("[%s] " % net if net else "", k, J[k]))
    return summ


def figures3(out, res, J, P=PREREG3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = os.path.join(out, "figures")
    os.makedirs(fd, exist_ok=True)
    files = []
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for tg, col in ((tag3({"method": "U_min", "c_min": P["umin_c_min"], "steps": P["steps"], "seedset": 0}), "C3"),
                    (tag3({"method": "B22", "steps": P["steps"], "seedset": 0}), "C0")):
        d = os.path.join(out, "runs", tg)
        if not os.path.isdir(d):
            continue
        for i, fj in enumerate(sorted(x for x in os.listdir(d) if x.endswith(".json"))):
            with open(os.path.join(d, fj), encoding="utf-8") as f:
                inf = json.load(f)
            tr = np.array([[r[1], r[3]] for r in inf["trace"]], dtype=float)
            if len(tr):
                ax.loglog(tr[:, 0], tr[:, 1], ".-", color=col, alpha=0.35, label=tg if i == 0 else None)
            if inf.get("switch_sigma", 0) > 0:
                ax.axvline(inf["switch_sigma"], color=col, ls="--", alpha=0.25)
    best = J["detail_B22"].get("WV_tele_best") or J["detail_U_min"].get("WV_tele_best")
    if best:
        ax.axvline(best["sigma"], color="C2", lw=2, label="sigma of best WV_tele")
    ax.invert_xaxis()
    ax.set_xlabel("sigma")
    ax.set_title("u = median |D_G - D_net|^2 / d (5%% subset); dashed = switch")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(fd, "fig1_u_traces_switch.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    files.append(p)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    g = sorted((r["cost_ratio_mean"], r["fid"]) for r in res.values() if r["method"] == "WV_tele" and "fid" in r)
    if g:
        ax.plot([a for a, b in g], [b for a, b in g], "s-", color="gray", label="WV_tele(k) run")
    for tg, r in res.items():
        if "fid" not in r or r["seedset"] != P["seedset_main"]:
            continue
        mk = {"M0": "ko", "U_min": "r*", "B22": "b*"}.get(r["method"])
        if mk:
            ax.plot(r["cost_ratio_mean"], r["fid"], mk, ms=11, label=tg)
    jd = J["detail_B22"] if "fid_tol" in J["detail_B22"] else J["detail_U_min"]
    if jd.get("fid_tol"):
        ax.axhline(jd["fid_tol"], color="gray", ls=":", label="1.03 x FID(M0)")
    ax.axvline(P["cost_max"], color="gray", ls="--", lw=0.8)
    ax.set_xlabel("net evaluations / M0")
    ax.set_ylabel("FID (%d images)" % P["n_images"])
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(fd, "fig2_fid_vs_cost.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    files.append(p)
    return files


def estimate_hours(ratio_vs_cifar=1.8, steps=40, n_images=5000, units=6.0, cifar_sec_per_1000_m0=232.0):
    """CIFAR の M0（18 歩、35 回）の実測から、64×64・steps 歩での M0 1 本の時間と、全体の時間を見積もる。"""
    per_img_nfe = cifar_sec_per_1000_m0 / 1000 / 35
    m0_hours = per_img_nfe * ratio_vs_cifar * C1.nfe_m0(steps) * n_images / 3600
    return {"M0_hours": m0_hours, "total_hours": m0_hours * units}


def pack_report3(out, name="C3_report", max_files=100, top="C3_results"):
    return C1.pack_report(out, name=name, max_files=max_files, top=top)


# ---------------------------------------------------------------------------
# 網ごとのフォルダ（C3b）
#   C3_results/                      データと積率（二つの網で共通）：afhqv2_64_moments.npz、fid_real_5k.json
#   C3_results/vp/  C3_results/ve/   網ごとの結果：runs/、log.txt、net_info.json、grid_state.json、report/、figures/
# C3（版 C3-0.1）のノートブックは、VP 型の結果を C3_results/ の直下に書いていた。
# migrate_legacy() は、それを C3_results/vp/ へ名前の付け替え（os.replace）だけで移す。計算し直さない。
# ---------------------------------------------------------------------------
NETS = ("vp", "ve")
SHARED_FILES = ("afhqv2_64_moments.npz", "fid_real_5k.json")
LEGACY_NET_ITEMS = ("runs", "log.txt", "net_info.json", "grid_state.json", "report", "figures")


def net_dir(root, net):
    assert net in NETS, "NET は 'vp' か 've'"
    return os.path.join(root, net)


def moments_path(root):
    """s_G の積率は、網によらず一つ（データだけで決まる）。"""
    return os.path.join(root, SHARED_FILES[0])


def net_url(net, base="https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/", P=PREREG3):
    return base + P["models"][net]


def migrate_legacy(root, log=print):
    """C3_results/ の直下にある VP 型の結果（C3 の版）を、C3_results/vp/ へ移す。
    何度呼んでも同じ（済んだものは飛ばす）。同じ名前が両方にあれば、上書きせずに止まる。"""
    vp = net_dir(root, "vp")
    moved, conflicts = [], []
    for it in LEGACY_NET_ITEMS:
        src, dst = os.path.join(root, it), os.path.join(vp, it)
        if it == "report" and os.path.exists(os.path.join(src, "summary_C3_both.json")):
            continue  # C4 の直し：直下の report/ に二つの網をまとめた集計があれば、それは C3b 以後の集計（VP 型の結果ではない）
        if os.path.exists(src):
            if os.path.exists(dst):
                conflicts.append(it)
                continue
            os.makedirs(vp, exist_ok=True)
            os.replace(src, dst)
            moved.append(it)
    if conflicts:
        raise RuntimeError("C3_results/ の直下と vp/ の両方に %s があります。上書きせずに止まりました。本筋に相談してください。" % conflicts)
    if moved:
        log("C3 の版の VP 型の結果を vp/ へ移しました：%s" % moved)
        with open(os.path.join(vp, "log.txt"), "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + "MIGRATED from C3_results/ (C3-0.1 layout): %s\n" % moved)
    return moved


def report_both(root, ref, srange=None, n_images=None, batch=None, extras=None, log=print, P=PREREG3):
    """二つの網を、それぞれ別に判定して並べる（どちらかが支持なら良い、という読み方はしない）。"""
    both = {"version": C3_VERSION, "nets": {}}
    for net in NETS:
        d = net_dir(root, net)
        if not os.path.isdir(os.path.join(d, "runs")):
            both["nets"][net] = {"status": "結果がまだありません"}
            continue
        summ = report3(d, ref, srange=srange, n_images=n_images, batch=batch,
                       extra=(extras or {}).get(net), log=log, P=P, net=net)
        J = summ["judgement"]
        both["nets"][net] = {k: J[k] for k in P["hc_names"][net]["U_env"] + P["hc_names"][net]["U_min"]}
    rep = os.path.join(root, "report")
    os.makedirs(rep, exist_ok=True)
    C1._atomic_json(os.path.join(rep, "summary_C3_both.json"), both)
    return both


def all_done(root, P=PREREG3):
    """二つの網の本番の計算（M0・格子・U_min・U_env・別の種の M0 と、その FID の特徴量）が全部そろったか。
    そろうまでは、最後のセルは集計も図も出さない（u の曲線を誰も見ないため）。"""
    steps, s0, s1, n = P["steps"], P["seedset_main"], P["seedset_noise"], P["n_images"]
    need_jobs = [{"method": "M0", "k": 0, "steps": steps, "seedset": s0},
                 {"method": "M0", "k": 0, "steps": steps, "seedset": s1},
                 {"method": "U_min", "k": 0, "c_min": P["umin_c_min"], "steps": steps, "seedset": s0},
                 {"method": "B22", "k": 0, "steps": steps, "seedset": s0}]
    missing = []
    for net in NETS:
        d = net_dir(root, net)
        for j in need_jobs:
            if not os.path.exists(os.path.join(d, "runs", tag3(j), "feats_%d.npy" % n)):
                missing.append("%s/%s" % (net, tag3(j)))
        gs = os.path.join(d, "grid_state.json")
        if not os.path.exists(gs):
            missing.append("%s/grid_state.json" % net)
        else:
            with open(gs, encoding="utf-8") as f:
                g = json.load(f)
            ks = wv_desc_ks(sigmas3(steps, None, P), P)
            if g.get("stop") is None and list(g.get("done", [])) != ks:
                missing.append("%s/格子（途中）" % net)
    return (not missing), missing
