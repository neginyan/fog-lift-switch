# c1core.py -- 実験場C 段階C1 の中身（MIT License, (c) 2026 Taishi Namba）
#
# 霧が晴れる時刻の合図 S_gap を、学習済みの拡散モデル（EDM の形：D(x, sigma) を返す網）で試すための道具。
# 同じコードを、Colab（torch, GPU）と作業場の CPU 確かめ（numpy）の両方で使う。
#   - 状態 x は (n, d) の配列（torch.Tensor でも numpy.ndarray でもよい）
#   - 網は net_fn(x, sigma) -> D（x と同じ型・形）。sigma は python の float。
# 他者のコード・重み（EDM, Inception）はここには入っていない。ノートブックが公式の場所から読む。

import json
import math
import os
import time

import numpy as np

C1_VERSION = "C1-1.0"

# ---------------------------------------------------------------------------
# 事前登録の定数（C1_事前登録.md の JSON と同じでなければならない。検証で照合する）
# ---------------------------------------------------------------------------
PREREG = {
    # 規則（実験場 B から変えない）
    "factor": 4.0,              # 基準値の 4 倍を超えたら切り替え
    "fog_sigma": 10.0,          # 基準値を作る霧の濃い所（sigma >= 10）
    "subset_frac": 0.05,        # 合図は束の 5% の点だけで作る
    # σ>=10 が霧の濃い所に当たらない場合の代わり（最初の 3 回の呼び出しの中央値）
    "fallback_first_calls": 3,
    # 「σ>=10 が霧の濃い所に当たるか」の判定（パイロット、M0 の軌道、50 歩）
    "pilot_steps": 50,
    "pilot_n": 256,
    "pilot_seedset": 9,
    "min_fog_calls": 3,
    "flat_band_low": [10.0, 20.0],
    "flat_band_high": [40.0, 80.0],
    "flat_ratio_max": 2.0,
    # 本番
    "main_steps": 18,
    "batch": 1000,
    "n_images_main": 10000,
    "seedset_main": 0,
    "seedset_noise": 1,
    "wv_sigma_targets": [40.0, 20.0, 10.0, 5.0, 2.5, 1.25, 0.6],
    # 判定
    "fid_rel_tol": 0.03,        # 画質を落とさない = FID <= 1.03 * FID(M0)
    "hc1_batch_frac": 0.8,      # HC1：束の 8 割以上
    "hc3_cost_max": 0.85,       # HC3：網の評価 <= 0.85 * M0
    "hc4_fid_rel": 0.03,        # HC4：FID <= 1.03 * FID(WV_tele 最良)
    "hc4_cost_margin": 0.10,    # HC4：費用 <= 費用(WV_tele 最良) + 0.10
}

# EDM の標準の目盛り
EDM_SIGMA_MIN = 0.002
EDM_SIGMA_MAX = 80.0
EDM_RHO = 7.0


# ---------------------------------------------------------------------------
# numpy / torch の小さな橋
# ---------------------------------------------------------------------------
def _is_torch(a):
    return type(a).__module__.startswith("torch")


def to_np(a):
    if _is_torch(a):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def _clone(a):
    return a.clone() if _is_torch(a) else a.copy()


def _idx(idx, like):
    if _is_torch(like):
        import torch
        return torch.as_tensor(np.asarray(idx), device=like.device, dtype=torch.long)
    return np.asarray(idx)


def rownorm2(a):
    """各行の二乗ノルム（長さ n の配列）。"""
    return (a * a).reshape(a.shape[0], -1).sum(-1)


# ---------------------------------------------------------------------------
# ガウスのスコア s_G（データの平均と共分散だけ）
# ---------------------------------------------------------------------------
def moments_from_batches(batches):
    """batches: (m, d) の numpy 配列の列。float64 で平均と共分散（1/N で割る）を返す。"""
    n = 0
    s1 = None
    s2 = None
    for b in batches:
        b = np.asarray(b, dtype=np.float64)
        if s1 is None:
            s1 = np.zeros(b.shape[1])
            s2 = np.zeros((b.shape[1], b.shape[1]))
        n += b.shape[0]
        s1 += b.sum(0)
        s2 += b.T @ b
    mu = s1 / n
    cov = s2 / n - np.outer(mu, mu)
    cov = 0.5 * (cov + cov.T)
    return mu, cov, n


def eig_moments(cov):
    lam, U = np.linalg.eigh(np.asarray(cov, dtype=np.float64))
    n_neg = int((lam < 0).sum())
    lam = np.clip(lam, 0.0, None)
    return lam, U, n_neg


class GaussianModel:
    """D_G(x, s) = mu + Sigma (Sigma + s^2 I)^{-1} (x - mu)、固有値分解で計算する。

    mu, lam, U は「状態と同じ型（numpy か torch）」で持つ。to_backend で変換する。
    """

    def __init__(self, mu, lam, U, to_backend=None):
        f = to_backend if to_backend is not None else (lambda a: np.asarray(a, dtype=np.float64))
        self.mu = f(np.asarray(mu, dtype=np.float64))
        self.lam = f(np.asarray(lam, dtype=np.float64))
        self.U = f(np.asarray(U, dtype=np.float64))
        self.d = int(np.asarray(mu).shape[0])

    @classmethod
    def from_cov(cls, mu, cov, to_backend=None):
        lam, U, _ = eig_moments(cov)
        return cls(mu, lam, U, to_backend)

    def denoise(self, x, sigma):
        c = (x - self.mu) @ self.U
        c = c * (self.lam / (self.lam + float(sigma) ** 2))
        return self.mu + c @ self.U.T

    def score(self, x, sigma):
        return (self.denoise(x, sigma) - x) / float(sigma) ** 2

    def teleport(self, x, s_from, s_to):
        """ガウスの確率流 ODE を s_from から s_to まで厳密に解く（Wang & Vastola の analytical teleportation）。"""
        c = (x - self.mu) @ self.U
        c = c * ((self.lam + float(s_to) ** 2) / (self.lam + float(s_from) ** 2)) ** 0.5
        return self.mu + c @ self.U.T


# ---------------------------------------------------------------------------
# EDM の目盛りと Heun 型の決定論的サンプラー（Karras et al. 2022, Alg. 2, S_churn = 0）
# ---------------------------------------------------------------------------
def edm_sigmas(num_steps, sigma_min=EDM_SIGMA_MIN, sigma_max=EDM_SIGMA_MAX, rho=EDM_RHO):
    i = np.arange(num_steps, dtype=np.float64)
    t = (sigma_max ** (1 / rho) + i / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return np.append(t, 0.0)


def nfe_m0(num_steps):
    return 2 * num_steps - 1


def call_sigmas(sigmas):
    """Heun の各呼び出しの (歩, sigma) の列。"""
    out = []
    N = len(sigmas) - 1
    for i in range(N):
        out.append((i, float(sigmas[i])))
        if i < N - 1:
            out.append((i, float(sigmas[i + 1])))
    return out


def heun_sample(x0, sigmas, D_call, start_step=0):
    """D_call(x, sigma, step, call_index) -> D。start_step から歩き始める（x0 は sigmas[start_step] の点）。"""
    x = x0
    N = len(sigmas) - 1
    ncall = 0
    for i in range(start_step, N):
        tc = float(sigmas[i])
        tn = float(sigmas[i + 1])
        d = (x - D_call(x, tc, i, ncall)) / tc
        ncall += 1
        xn = x + (tn - tc) * d
        if i < N - 1:
            d2 = (xn - D_call(xn, tn, i, ncall)) / tn
            ncall += 1
            xn = x + (tn - tc) * (0.5 * d + 0.5 * d2)
        x = xn
    return x


class CountingNet:
    """網の呼び出しを点の数で数える包み。"""

    def __init__(self, net_fn):
        self.net_fn = net_fn
        self.points = 0

    def __call__(self, x, sigma):
        self.points += int(x.shape[0])
        return self.net_fn(x, sigma)


# ---------------------------------------------------------------------------
# 合図 S_gap の規則（定数は PREREG）
# ---------------------------------------------------------------------------
class GapRule:
    """各呼び出しの統計量 s（束の点での中央値）を受け取り、切り替えるかを返す。

    mode = "fog"  : sigma >= fog_sigma の呼び出しの s の中央値を基準値にする。
                    sigma < fog_sigma になった最初の呼び出しで基準値を固める。以後、s > factor * 基準値 で切り替え。
    mode = "first": 最初の fallback_first_calls 回の呼び出しの s の中央値を基準値にする。
    切り替えたら後戻りしない。
    """

    def __init__(self, mode="fog", factor=None, fog_sigma=None, n_first=None):
        assert mode in ("fog", "first")
        self.mode = mode
        self.factor = PREREG["factor"] if factor is None else factor
        self.fog_sigma = PREREG["fog_sigma"] if fog_sigma is None else fog_sigma
        self.n_first = PREREG["fallback_first_calls"] if n_first is None else n_first
        self.base_vals = []
        self.baseline = None
        self.switched = False
        self.ncalls = 0

    def in_baseline_phase(self, sigma):
        if self.baseline is not None:
            return False
        if self.mode == "fog":
            return sigma >= self.fog_sigma
        return self.ncalls < self.n_first

    def observe(self, sigma, s):
        """True を返したら、この呼び出しから網に切り替える。"""
        if self.switched:
            self.ncalls += 1
            return True
        if self.in_baseline_phase(sigma):
            self.base_vals.append(float(s))
            self.ncalls += 1
            return False
        if self.baseline is None:
            if len(self.base_vals) == 0:
                raise RuntimeError("基準値を作る呼び出しが 1 つもありません（mode=%s）" % self.mode)
            self.baseline = float(np.median(self.base_vals))
        self.ncalls += 1
        if float(s) > self.factor * self.baseline:
            self.switched = True
            return True
        return False


def gap_stat_points(DG, Dn, x):
    """|s_G - s_net|^2 / |s_net|^2 = |D_G - D_net|^2 / |D_net - x|^2（sigma は消える）。点ごとの値（numpy）。"""
    num = to_np(rownorm2(DG - Dn)).astype(np.float64)
    den = to_np(rownorm2(Dn - x)).astype(np.float64)
    return num / np.maximum(den, 1e-300)


# ---------------------------------------------------------------------------
# 方法
# ---------------------------------------------------------------------------
def method_tag(m):
    if m["method"] in ("M0", "S_gap", "Gall"):
        base = m["method"]
    else:
        base = "%s_k%02d" % (m["method"], m["k"])
    return "%s_N%03d_s%d" % (base, m["steps"], m["seedset"])


def run_one(method, k, x_init, sigmas, net_fn, G, rng=None, branch="fog", rule_kw=None, sync=None):
    """一つの束を走らせる。x_init は sigma_max = sigmas[0] の点（= latents * sigmas[0]）。

    返り値：(x_final, info)。info["cost_units"] は「束全体で網 1 回」を単位にした網の評価の回数。
    """
    n = int(x_init.shape[0])
    net = CountingNet(net_fn)
    N = len(sigmas) - 1
    info = {"method": method, "k": k, "n": n, "steps": N, "branch": None}
    if sync:
        sync()
    t0 = time.time()

    if method == "M0":
        x = heun_sample(x_init, sigmas, lambda x, s, i, c: net(x, s))

    elif method == "Gall":
        x = heun_sample(x_init, sigmas, lambda x, s, i, c: G.denoise(x, s))

    elif method == "WV_step":
        # 最初の k 歩（その歩の補正の呼び出しも含む）を D_G で進み、あとは網
        x = heun_sample(x_init, sigmas, lambda x, s, i, c: G.denoise(x, s) if i < k else net(x, s))
        info["switch_sigma"] = float(sigmas[k])

    elif method == "WV_tele":
        # sigma_max から sigmas[k] までガウスで厳密に飛ばし、そこから網で Heun
        xk = G.teleport(x_init, float(sigmas[0]), float(sigmas[k])) if k > 0 else x_init
        x = heun_sample(xk, sigmas, lambda x, s, i, c: net(x, s), start_step=k)
        info["switch_sigma"] = float(sigmas[k])

    elif method == "S_gap":
        if rng is None:
            rng = np.random.default_rng(0)
        kw = dict(rule_kw or {})
        frac = kw.pop("subset_frac", PREREG["subset_frac"])
        rule = GapRule(mode=branch, **kw)
        m = max(1, int(math.ceil(frac * n)))
        trace = []
        state = {"switch_sigma": None, "switch_call": None}

        def D_call(x, s, i, c):
            if rule.switched:
                rule.observe(s, float("nan"))
                return net(x, s)
            DG = G.denoise(x, s)
            sub = np.sort(rng.choice(n, size=m, replace=False))
            isub = _idx(sub, x)
            xs = x[isub]
            Dn_sub = net(xs, s)
            st = float(np.median(gap_stat_points(DG[isub], Dn_sub, xs)))
            dd = float(np.median(to_np(rownorm2(DG[isub] - Dn_sub)).astype(np.float64))) / G.d
            switch = rule.observe(s, st)
            trace.append([c, float(s), st, dd])  # 呼び出し, σ, 合図の統計量, 中央値 |D_G − D_net|²/d（観察用）
            out = _clone(DG)
            out[isub] = Dn_sub
            if switch:
                rest = np.setdiff1d(np.arange(n), sub)
                if len(rest) > 0:
                    irest = _idx(rest, x)
                    out[irest] = net(x[irest], s)
                state["switch_sigma"] = float(s)
                state["switch_call"] = int(c)
            return out

        x = heun_sample(x_init, sigmas, D_call)
        info.update({
            "branch": branch, "subset_m": m, "trace": trace, "baseline": rule.baseline,
            "switched": rule.switched,
            "switch_sigma": state["switch_sigma"] if rule.switched else 0.0,
            "switch_call": state["switch_call"],
        })
    else:
        raise ValueError(method)

    if sync:
        sync()
    info["seconds"] = time.time() - t0
    info["net_points"] = net.points
    info["cost_units"] = net.points / n
    info["cost_ratio"] = info["cost_units"] / nfe_m0(N)
    return x, info


# ---------------------------------------------------------------------------
# パイロット：M0 の軌道に沿って、各呼び出しで全点の統計量と |D_G - D_net|^2 を測る
# ---------------------------------------------------------------------------
def pilot_curve(x_init, sigmas, net_fn, G):
    rows = []

    def D_call(x, s, i, c):
        Dn = net_fn(x, s)
        DG = G.denoise(x, s)
        st = gap_stat_points(DG, Dn, x)
        dd = to_np(rownorm2(DG - Dn)).astype(np.float64) / G.d
        rows.append({"call": int(c), "step": int(i), "sigma": float(s),
                     "stat_median": float(np.median(st)),
                     "stat_q25": float(np.quantile(st, 0.25)), "stat_q75": float(np.quantile(st, 0.75)),
                     "dD2_per_dim_median": float(np.median(dd)), "dD2_per_dim_mean": float(np.mean(dd))})
        return Dn

    heun_sample(x_init, sigmas, D_call)
    return rows


def decide_branch(rows, P=PREREG):
    """事前登録の規則で、σ>=10 が霧の濃い所に当たるかを決める。"""
    s = np.array([r["sigma"] for r in rows])
    v = np.array([r["stat_median"] for r in rows])
    fog = s >= P["fog_sigma"]
    res = {"n_fog_calls": int(fog.sum()), "sigma_max": float(s.max())}
    lo = (s >= P["flat_band_low"][0]) & (s <= P["flat_band_low"][1])
    hi = (s >= P["flat_band_high"][0]) & (s <= P["flat_band_high"][1])
    res["n_low"] = int(lo.sum())
    res["n_high"] = int(hi.sum())
    ok_range = res["sigma_max"] > P["fog_sigma"] and res["n_fog_calls"] >= P["min_fog_calls"]
    if lo.sum() == 0 or hi.sum() == 0 or not ok_range:
        res.update({"ratio_low_over_high": None, "flat": False, "branch": "first",
                    "reason": "sigma の範囲が足りない"})
        return res
    ratio = float(np.median(v[lo]) / np.median(v[hi]))
    flat = (1.0 / P["flat_ratio_max"]) <= ratio <= P["flat_ratio_max"]
    res.update({"ratio_low_over_high": ratio, "flat": bool(flat),
                "branch": "fog" if flat else "first",
                "reason": "平ら" if flat else "平らでない（基準値を最初の %d 回で作る）" % P["fallback_first_calls"]})
    return res


# ---------------------------------------------------------------------------
# 書き出し（1 束 1 ファイル、途中で切れても再開できる）
# ---------------------------------------------------------------------------
def _atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _atomic_npz(path, **arrs):
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)


def to_uint8_images(x, img_shape):
    a = to_np(x).astype(np.float64).reshape((-1,) + tuple(img_shape))
    return np.clip(a * 127.5 + 128.0, 0, 255).astype(np.uint8)


def wv_k_grid(sigmas, targets=None):
    targets = PREREG["wv_sigma_targets"] if targets is None else targets
    t = np.asarray(sigmas[1:-1])  # k = 1..N-1
    ks = set()
    for tg in targets:
        ks.add(int(np.argmin(np.abs(np.log(t) - math.log(tg)))) + 1)
    return sorted(ks)


def main_jobs(steps=None, seedset=None, ks=None, with_noise=True):
    steps = PREREG["main_steps"] if steps is None else steps
    seedset = PREREG["seedset_main"] if seedset is None else seedset
    ks = wv_k_grid(edm_sigmas(steps)) if ks is None else ks
    jobs = [{"method": "M0", "k": 0, "steps": steps, "seedset": seedset},
            {"method": "S_gap", "k": 0, "steps": steps, "seedset": seedset}]
    for k in ks:
        jobs.append({"method": "WV_tele", "k": k, "steps": steps, "seedset": seedset})
    for k in ks:
        jobs.append({"method": "WV_step", "k": k, "steps": steps, "seedset": seedset})
    if with_noise:
        jobs.append({"method": "M0", "k": 0, "steps": steps, "seedset": PREREG["seedset_noise"]})
    return jobs


def subset_rng(seedset, steps, b):
    return np.random.default_rng([20261112, int(seedset), int(steps), int(b)])


def phase_pilot(out, net_fn, G, make_latents, n=None, steps=None, seedset=None, log=print):
    n = PREREG["pilot_n"] if n is None else n
    steps = PREREG["pilot_steps"] if steps is None else steps
    seedset = PREREG["pilot_seedset"] if seedset is None else seedset
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, "pilot.json")
    if os.path.exists(p):
        log("パイロットは済んでいます：" + p)
        with open(p, encoding="utf-8") as f:
            return json.load(f)["decision"]
    sig = edm_sigmas(steps)
    x0 = make_latents(seedset, 0, n) * float(sig[0])
    rows = pilot_curve(x0, sig, net_fn, G)
    dec = decide_branch(rows)
    _atomic_json(p, {"version": C1_VERSION, "steps": steps, "n": n, "seedset": seedset,
                     "rows": rows, "decision": dec, "prereg": PREREG})
    log("パイロットの判定：%s（%s）" % (dec["branch"], dec["reason"]))
    return dec


def load_branch(out):
    with open(os.path.join(out, "pilot.json"), encoding="utf-8") as f:
        return json.load(f)["decision"]["branch"]


def phase_runs(out, jobs, net_fn, G, make_latents, n_images=None, batch=None, img_shape=(3, 32, 32),
               branch=None, sync=None, log=print):
    n_images = PREREG["n_images_main"] if n_images is None else n_images
    batch = PREREG["batch"] if batch is None else batch
    if branch is None:
        branch = load_branch(out)
    n_batches = int(math.ceil(n_images / batch))
    for job in jobs:
        tag = method_tag(job)
        d = os.path.join(out, "runs", tag)
        os.makedirs(d, exist_ok=True)
        sig = edm_sigmas(job["steps"])
        for b in range(n_batches):
            fi = os.path.join(d, "b%03d.npz" % b)
            fj = os.path.join(d, "b%03d.json" % b)
            if os.path.exists(fi) and os.path.exists(fj):
                continue
            nb = min(batch, n_images - b * batch)
            x0 = make_latents(job["seedset"], b, nb) * float(sig[0])
            x, info = run_one(job["method"], job["k"], x0, sig, net_fn, G,
                              rng=subset_rng(job["seedset"], job["steps"], b), branch=branch, sync=sync)
            info.update({"tag": tag, "batch": b, "seedset": job["seedset"], "version": C1_VERSION})
            _atomic_npz(fi, images=to_uint8_images(x, img_shape))
            _atomic_json(fj, info)
            msg = "%s b%03d  費用 %.3f × M0  %.1f 秒" % (tag, b, info["cost_ratio"], info["seconds"])
            if job["method"] == "S_gap":
                msg += "  切り替え σ = %.4g" % info["switch_sigma"]
            log(msg)
            with open(os.path.join(out, "log.txt"), "a", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------
def fid_from_stats(mu, sigma, mu_ref, sigma_ref):
    """EDM の fid.py と同じ式。"""
    import scipy.linalg
    m = np.square(mu - mu_ref).sum()
    try:
        s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    except TypeError:  # 新しい scipy には disp がない
        s = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref))
    return float(np.real(m + np.trace(sigma + sigma_ref - s * 2)))


def feat_stats(feats):
    feats = np.asarray(feats, dtype=np.float64)
    return feats.mean(0), np.cov(feats, rowvar=False)


def run_complete(out, tag, n_images=None, batch=None):
    n_images = PREREG["n_images_main"] if n_images is None else n_images
    batch = PREREG["batch"] if batch is None else batch
    d = os.path.join(out, "runs", tag)
    nb = int(math.ceil(n_images / batch))
    return all(os.path.exists(os.path.join(d, "b%03d.npz" % b)) for b in range(nb))


def phase_features(out, tags, feat_fn, n_images=None, batch=None, log=print):
    """feat_fn(uint8 の NCHW numpy) -> (m, F) numpy。方法ごとに feats.npy を書く。"""
    n_images = PREREG["n_images_main"] if n_images is None else n_images
    batch = PREREG["batch"] if batch is None else batch
    nb = int(math.ceil(n_images / batch))
    for tag in tags:
        d = os.path.join(out, "runs", tag)
        fp = os.path.join(d, "feats_%d.npy" % n_images)
        if os.path.exists(fp) or not run_complete(out, tag, n_images, batch):
            continue
        fs = []
        for b in range(nb):
            with np.load(os.path.join(d, "b%03d.npz" % b)) as z:
                fs.append(np.asarray(feat_fn(z["images"]), dtype=np.float64))
        np.save(fp + ".tmp.npy", np.concatenate(fs, 0))
        os.replace(fp + ".tmp.npy", fp)
        log("特徴量：%s" % tag)


# ---------------------------------------------------------------------------
# 集計と判定
# ---------------------------------------------------------------------------
def collect(out, n_images=None, batch=None, ref=None):
    n_images = PREREG["n_images_main"] if n_images is None else n_images
    batch = PREREG["batch"] if batch is None else batch
    nb = int(math.ceil(n_images / batch))
    res = {}
    rd = os.path.join(out, "runs")
    if not os.path.isdir(rd):
        return res
    for tag in sorted(os.listdir(rd)):
        d = os.path.join(rd, tag)
        infos = []
        for b in range(nb):
            fj = os.path.join(d, "b%03d.json" % b)
            if os.path.exists(fj):
                with open(fj, encoding="utf-8") as f:
                    infos.append(json.load(f))
        if not infos:
            continue
        r = {"method": infos[0]["method"], "k": infos[0]["k"], "steps": infos[0]["steps"],
             "seedset": infos[0]["seedset"], "n_batches_done": len(infos), "complete": len(infos) == nb,
             "cost_ratio_mean": float(np.mean([i["cost_ratio"] for i in infos])),
             "seconds_total": float(np.sum([i["seconds"] for i in infos]))}
        if "switch_sigma" in infos[0]:
            sw = [float(i["switch_sigma"]) for i in infos]
            r["switch_sigmas"] = sw
            r["switch_sigma_median"] = float(np.median(sw))
        if r["method"] == "S_gap":
            r["baselines"] = [i["baseline"] for i in infos]
            r["branch"] = infos[0]["branch"]
        fp = os.path.join(d, "feats_%d.npy" % n_images)
        if ref is not None and os.path.exists(fp):
            mu, sg = feat_stats(np.load(fp))
            r["fid"] = fid_from_stats(mu, sg, ref[0], ref[1])
        res[tag] = r
    return res


def judge(res, steps=None, seedset=None, P=PREREG):
    """HC1〜HC4 を事前登録の基準で判定する。足りないものがあれば「未判定」。"""
    steps = P["main_steps"] if steps is None else steps
    seedset = P["seedset_main"] if seedset is None else seedset
    sig = edm_sigmas(steps)
    T = lambda m, k=0: method_tag({"method": m, "k": k, "steps": steps, "seedset": seedset})
    out = {"steps": steps, "seedset": seedset}
    m0 = res.get(T("M0"))
    sg = res.get(T("S_gap"))
    if not m0 or "fid" not in m0 or not sg or "fid" not in sg:
        out["status"] = "未判定（M0 か S_gap の FID がまだありません）"
        return out
    tol = (1 + P["fid_rel_tol"]) * m0["fid"]
    out["fid_M0"] = m0["fid"]
    out["fid_tol"] = tol
    out["fid_S_gap"] = sg["fid"]
    out["cost_S_gap"] = sg["cost_ratio_mean"]
    out["switch_sigmas_S_gap"] = sg["switch_sigmas"]

    def grid(meth):
        g = []
        for tag, r in res.items():
            if r["method"] == meth and r["steps"] == steps and r["seedset"] == seedset and "fid" in r:
                g.append((r["k"], r["fid"], r["cost_ratio_mean"]))
        return sorted(g)

    gs = grid("WV_step")
    gt = grid("WV_tele")
    out["grid_WV_step"] = gs
    out["grid_WV_tele"] = gt

    # HC1：WV_step の格子で、小さい k から続けて「画質を落とさない」最大の k
    kstar = 0
    for k, f, c in gs:
        if f <= tol:
            kstar = k
        else:
            break
    out["k_star_WV_step"] = kstar
    if not gs:
        out["HC1"] = "未判定（WV_step の格子がありません）"
    elif kstar == 0:
        out["sigma_safe"] = None
        out["HC1"] = "不支持（WV_step で画質を落とさず飛ばせる k がない）"
    else:
        ss = float(sig[kstar])
        out["sigma_safe"] = ss
        sw = np.asarray(sg["switch_sigmas"])
        frac = float(np.mean((sw >= ss * (1 - 1e-9)) & (sw > 0)))
        out["HC1_frac"] = frac
        out["HC1"] = "支持" if frac >= P["hc1_batch_frac"] else "不支持"
    # HC2
    out["HC2"] = "支持" if sg["fid"] <= tol else "不支持"
    # HC3
    out["HC3"] = "支持" if sg["cost_ratio_mean"] <= P["hc3_cost_max"] else "不支持"
    # HC4：WV_tele の格子で画質を落とさない k のうち最も安い（最大の k）
    safe_t = [(k, f, c) for (k, f, c) in gt if f <= tol]
    if not gt:
        out["HC4"] = "未判定（WV_tele の格子がありません）"
    else:
        if safe_t:
            kb, fb, cb = max(safe_t, key=lambda z: z[0])
        else:
            kb, fb, cb = 0, m0["fid"], 1.0
        out["WV_tele_best"] = {"k": kb, "fid": fb, "cost": cb, "sigma": float(sig[kb])}
        ok = sg["fid"] <= (1 + P["hc4_fid_rel"]) * fb and sg["cost_ratio_mean"] <= cb + P["hc4_cost_margin"]
        out["HC4"] = "支持" if ok else "不支持"
    noise = res.get(method_tag({"method": "M0", "k": 0, "steps": steps, "seedset": P["seedset_noise"]}))
    if noise and "fid" in noise:
        out["fid_M0_other_seeds"] = noise["fid"]
    out["status"] = "判定済み"
    return out


# ---------------------------------------------------------------------------
# 図
# ---------------------------------------------------------------------------
def make_figures(out, res, judged, steps=None, seedset=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = PREREG["main_steps"] if steps is None else steps
    seedset = PREREG["seedset_main"] if seedset is None else seedset
    fd = os.path.join(out, "figures")
    os.makedirs(fd, exist_ok=True)
    files = []
    # 図 1：パイロットの |D_G - D_net|^2 と合図の統計量、切り替えの位置
    pp = os.path.join(out, "pilot.json")
    if os.path.exists(pp):
        with open(pp, encoding="utf-8") as f:
            pil = json.load(f)
        rows = pil["rows"]
        s = np.array([r["sigma"] for r in rows])
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].loglog(s, [r["dD2_per_dim_median"] for r in rows], "o-", ms=3, label="median |D_G - D_net|^2 / d")
        ax[1].loglog(s, [r["stat_median"] for r in rows], "o-", ms=3, label="S_gap stat (median)")
        sgr = res.get(method_tag({"method": "S_gap", "k": 0, "steps": steps, "seedset": seedset}))
        for a in ax:
            a.axvline(PREREG["fog_sigma"], color="gray", ls=":", label="sigma = 10")
            if sgr:
                for i, sw in enumerate(sgr["switch_sigmas"]):
                    if sw > 0:
                        a.axvline(sw, color="C3", alpha=0.3, label="S_gap switch" if i == 0 else None)
            if judged.get("sigma_safe"):
                a.axvline(judged["sigma_safe"], color="C2", ls="--", label="sigma_safe (WV_step)")
            a.invert_xaxis()
            a.set_xlabel("sigma")
            a.legend(fontsize=7)
        ax[1].set_title("branch: %s" % pil["decision"]["branch"])
        ax[0].set_title("pilot (M0 trajectories, %d steps)" % pil["steps"])
        fig.tight_layout()
        p = os.path.join(fd, "fig1_gaussian_gap_and_switch.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        files.append(p)
    # 図 2：FID と費用
    fig, ax = plt.subplots(figsize=(6, 4.5))
    any_pt = False
    for meth, mk in (("WV_tele", "s-"), ("WV_step", "^-")):
        g = sorted([(r["cost_ratio_mean"], r["fid"]) for r in res.values()
                    if r["method"] == meth and r["steps"] == steps and r["seedset"] == seedset and "fid" in r])
        if g:
            ax.plot([a for a, b in g], [b for a, b in g], mk, label=meth)
            any_pt = True
    for meth, mk in (("M0", "ko"), ("S_gap", "r*")):
        r = res.get(method_tag({"method": meth, "k": 0, "steps": steps, "seedset": seedset}))
        if r and "fid" in r:
            ax.plot([r["cost_ratio_mean"]], [r["fid"]], mk, ms=11, label=meth)
            any_pt = True
    if judged.get("fid_tol"):
        ax.axhline(judged["fid_tol"], color="gray", ls=":", label="1.03 x FID(M0)")
    if any_pt:
        ax.set_xlabel("net evaluations / M0")
        ax.set_ylabel("FID (%d images)" % PREREG["n_images_main"])
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = os.path.join(fd, "fig2_fid_vs_cost.png")
        fig.savefig(p, dpi=130)
        files.append(p)
    plt.close(fig)
    # 図 3：S_gap の統計量の軌跡（束ごと）
    d = os.path.join(out, "runs", method_tag({"method": "S_gap", "k": 0, "steps": steps, "seedset": seedset}))
    if os.path.isdir(d):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for fj in sorted(x for x in os.listdir(d) if x.endswith(".json")):
            with open(os.path.join(d, fj), encoding="utf-8") as f:
                inf = json.load(f)
            tr = np.array(inf["trace"]) if inf["trace"] else np.zeros((0, 3))
            if len(tr):
                ax.loglog(tr[:, 1], tr[:, 2], ".-", alpha=0.5)
            if inf["baseline"]:
                ax.axhline(PREREG["factor"] * inf["baseline"], color="C3", alpha=0.2)
        ax.invert_xaxis()
        ax.set_xlabel("sigma")
        ax.set_ylabel("stat (median over 5% subset)")
        ax.set_title("S_gap traces; red = 4 x baseline")
        fig.tight_layout()
        p = os.path.join(fd, "fig3_sgap_traces.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        files.append(p)
    return files


def phase_report(out, ref, steps=None, seedset=None, n_images=None, batch=None, extra=None, log=print):
    res = collect(out, n_images=n_images, batch=batch, ref=ref)
    judged = judge(res, steps=steps, seedset=seedset)
    figs = make_figures(out, res, judged, steps=steps, seedset=seedset)
    rep = os.path.join(out, "report")
    os.makedirs(rep, exist_ok=True)
    summ = {"version": C1_VERSION, "prereg": PREREG, "results": res, "judgement": judged,
            "figures": [os.path.basename(f) for f in figs]}
    if extra:
        summ["extra"] = extra
    _atomic_json(os.path.join(rep, "summary_C1.json"), summ)
    for k in ("HC1", "HC2", "HC3", "HC4"):
        if k in judged:
            log("%s：%s" % (k, judged[k]))
    log("状態：%s" % judged.get("status"))
    return summ


# ---------------------------------------------------------------------------
# 持ち帰る結果を zip にする（画像の npz と特徴量は入れない。1 つ 100 ファイル以下、パスを保つ）
# ---------------------------------------------------------------------------
def pack_report(out, name="C1_report", max_files=100, top="C1_results"):
    import zipfile
    files = []
    for dp, dn, fn in os.walk(out):
        dn.sort()
        for f in sorted(fn):
            if f.endswith((".json", ".png", ".txt")) and ".tmp" not in f:
                files.append(os.path.join(dp, f))
    for old in os.listdir(out):
        if old.startswith(name + "_part") and old.endswith(".zip"):
            os.remove(os.path.join(out, old))
    zips = []
    for i in range(0, len(files), max_files):
        zp = os.path.join(out, "%s_part%02d.zip" % (name, i // max_files + 1))
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for f in files[i:i + max_files]:
                z.write(f, os.path.join(top, os.path.relpath(f, out)))
        zips.append(zp)
    return zips, len(files)
