# verify_all.py -- one CPU check for paper G3 (MIT License, (c) 2026 Taishi Namba)
#
# Recomputes, from the run records in records/, every pre-registered decision (HC1-HC25), every number printed in
# Tables 1-4 of the paper, and every switching time (by replaying the rules on the recorded signal traces).
# Also checks the SHA-256 of the rule code and of the pre-registrations, the code embedded in the notebooks,
# and a few exact properties of the Gaussian denoiser and of the rules (with "power" checks: deliberately wrong
# versions must fail).
#
# Usage:  python verify/verify_all.py            (needs numpy, scipy, matplotlib; no GPU, no network access)
# Exit code 0 if every check passes.
import hashlib
import json
import math
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
import c1core as C1  # noqa: E402
import c2core as C2  # noqa: E402
import c3core as C3  # noqa: E402
import c6core as C6  # noqa: E402

REC = os.path.join(ROOT, "records")
RESULTS = []
SUP, NOT = "支持", "不支持"  # "supported", "not supported" as written in the records

CODE_SHA = {  # versions run on Colab: C3-1.0 (c1-c3) and C6-1.0 (c6)
    "c1core.py": "b2b17f57854ab535dbcb93b2712b65d0f050cc3ff093d24fd49428141a1ca32e",
    "c2core.py": "6fb4c46b979666cd5ea594afceb7df100bbf54711a4fc45e0398f62be4a6421e",
    "c3core.py": "972e1b74e08c07109b78d3d12ad5d6bbede833a8c3c7804fc15f4209e0547336",
    "c6core.py": "958b7b276a351be19c407bbdbe579ca9d9791a5e7331ce76fa657d9c89d32a72",
}
PREREG_SHA = {  # also listed in each preregistration/*/prereg_sha256.txt and in Table 5 of the paper
    "C1": ["4f75cdca9305baacbe5b762e1e346f3a36332cd7108e768fae6ef5d480198ffb"],
    "C2": ["f2a269dc41c0cedbc8f4bfc4d18a6f9926273d7b080cb8c383c892b74f2233ba",
           "44b46c0358a08573b97b360a989a49cb6818c22dbcc3ac6590988139dacc527a"],
    "C3": ["2300b50b19386b8def6ad2e5c730ded5d4752c10a5e89b534e68db10fb374ca1",
           "2835e57ea53324109b3caa96e778e7f58cd294adbf0997adf0402fcfc789ce61"],
    "C6": ["57646881ccbcafb0ab3b8f2792099e6031ea5f41d62238de951800dd42776c5d"],
}


def check(name, ok, detail=""):
    RESULTS.append({"name": name, "ok": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name + (("  [" + detail + "]") if detail else ""))


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def r(x, n):
    return round(float(x), n)


# ---------------------------------------------------------------------------------------------------------------
# 1. Integrity: code, pre-registrations, notebooks
# ---------------------------------------------------------------------------------------------------------------
def t_integrity():
    for f, h in CODE_SHA.items():
        check("code: %s is the version run on Colab" % f, sha(os.path.join(ROOT, f)) == h, h[:12])
    for st, hs in PREREG_SHA.items():
        d = os.path.join(ROOT, "preregistration", st)
        listed = {}
        for line in open(os.path.join(d, "prereg_sha256.txt"), encoding="utf-8"):
            p = line.split()
            if len(p) >= 2:
                listed[p[1]] = p[0]
        got = {n: sha(os.path.join(d, n)) for n in listed}
        check("pre-registration %s: files unchanged since registration (%d file(s))" % (st, len(listed)),
              all(got[n] == listed[n] for n in listed) and sorted(got.values()) == sorted(hs), ", ".join(h[:8] for h in hs))
    # notebooks: every '%%writefile .../cXcore.py' cell must equal the core file of the same name
    import glob
    bad, n = [], 0
    for nb in sorted(glob.glob(os.path.join(ROOT, "notebooks", "*.ipynb"))):
        cells = json.load(open(nb, encoding="utf-8"))["cells"]
        for c in cells:
            src = "".join(c.get("source", []))
            if src.startswith("%%writefile") and "core.py" in src.split("\n", 1)[0]:
                name = os.path.basename(src.split("\n", 1)[0].split()[-1])
                body = src.split("\n", 1)[1]
                core = open(os.path.join(ROOT, name), encoding="utf-8").read()
                n += 1
                if body.rstrip("\n") != core.rstrip("\n"):
                    bad.append((os.path.basename(nb), name))
    # C1/C2 notebooks embed older or identical cores: only the C3 and C6 notebooks are required to match exactly
    must = [b for b in bad if b[0] in ("C3_colab.ipynb", "C6_colab.ipynb")]
    check("notebooks: the code cells of the C3 and C6 notebooks are the rule code in this repository", not must and n > 0,
          "%d embedded core cells checked; differing (older stage notebooks): %s" % (n, sorted(set(bad)) or "none"))


# ---------------------------------------------------------------------------------------------------------------
# 2. Exact properties (no records needed)
# ---------------------------------------------------------------------------------------------------------------
def t_exact():
    from scipy.stats import multivariate_normal
    rng = np.random.default_rng(1)
    d = 6
    A = rng.standard_normal((d, d))
    cov = A @ A.T / d + 0.05 * np.eye(d)
    mu = rng.standard_normal(d)
    G = C1.GaussianModel.from_cov(mu, cov)
    worst = worst_wrong = 0.0
    for sigma in (0.05, 0.5, 3.0, 40.0):
        x = mu + rng.standard_normal((5, d)) * (1 + sigma)
        mvn = multivariate_normal(mu, cov + sigma ** 2 * np.eye(d))
        h = 1e-5 * (1 + sigma)
        num = np.zeros_like(x)
        for j in range(d):
            e = np.zeros(d)
            e[j] = h
            num[:, j] = (mvn.logpdf(x + e) - mvn.logpdf(x - e)) / (2 * h)
        worst = max(worst, np.abs(G.score(x, sigma) - num).max() / np.abs(num).max())
        wrong = -np.linalg.solve(cov, (x - mu).T).T  # sigma^2 I forgotten
        worst_wrong = max(worst_wrong, np.abs(wrong - num).max() / np.abs(num).max())
    check("Gaussian denoiser: (D_G - x)/sigma^2 is the exact score of N(mu, Sigma + sigma^2 I) (Tweedie)", worst < 1e-6,
          "max rel. error %.1e; wrong version %.1e" % (worst, worst_wrong))
    check("Gaussian denoiser: power check (forgetting sigma^2 I fails)", worst_wrong > 1e-2)
    # teleportation = exact solution of the Gaussian probability-flow ODE
    A = rng.standard_normal((5, 5))
    G = C1.GaussianModel.from_cov(rng.standard_normal(5), A @ A.T + 0.1 * np.eye(5))
    x0 = rng.standard_normal((4, 5)) * 80.0
    s0, s1, n = 80.0, 0.7, 4000
    us = np.linspace(math.log(s0), math.log(s1), n + 1)
    f = lambda u, x: x - G.denoise(x, math.exp(u))  # noqa: E731  dx/dln(sigma) = x - D
    x = x0.copy()
    for i in range(n):
        hh = us[i + 1] - us[i]
        k1 = f(us[i], x)
        k2 = f(us[i] + hh / 2, x + hh / 2 * k1)
        k3 = f(us[i] + hh / 2, x + hh / 2 * k2)
        k4 = f(us[i + 1], x + hh * k3)
        x = x + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    err = np.abs(G.teleport(x0, s0, s1) - x).max() / np.abs(x).max()
    c = (x0 - G.mu) @ G.U
    wrong = G.mu + (c * ((G.lam + s1 ** 2) / (G.lam + s0 ** 2))) @ G.U.T
    errw = np.abs(wrong - x).max() / np.abs(x).max()
    check("teleport: closed form equals an RK4 solution of the Gaussian probability-flow ODE", err < 1e-9 and errw > 1e-2,
          "%.1e (wrong exponent: %.1e)" % (err, errw))
    # schedule and call counts
    s40 = C1.edm_sigmas(40)
    check("EDM schedule: 40 steps -> 79 calls, sigma from 80 to 0.002; 18 steps -> 35 calls",
          C1.nfe_m0(40) == 79 and C1.nfe_m0(18) == 35 and abs(float(s40[0]) - 80) < 1e-9 and abs(float(s40[-2]) - 0.002) < 1e-12)


def umin_literal(us, c):
    for k in range(1, len(us)):
        if us[k] >= c * min(us[:k]):
            return k
    return None


def uenv_literal(sig, us, delta=0.3, C=1.515000000000000, m_min=3):
    """U_env written again from the definition in the paper (Sec. 3.1), independently of c3core."""
    ln = [math.log(s) for s in sig]
    X, start = [], []
    for k in range(len(us)):
        a = min(i for i in range(k + 1) if ln[i] <= ln[k] + delta + 1e-12)
        a = min(a, max(k - m_min + 1, 0))
        start.append(a)
        X.append(max(us[a:k + 1]))
        if a >= 1 and min(us[a:k + 1]) >= C * min(X[:a]):
            return k
    return None


def uenv_c3(sig, us):
    k = C3.PREREG3["b22_constants"]
    rule = C3.UEnvRule(k["DELTA"], k["C"], k["M_MIN"], k["window_tol"])
    return next((i for i, (a, b) in enumerate(zip(sig, us)) if rule.observe(a, b)), None)


def t_rules():
    k = C3.PREREG3["b22_constants"]
    check("rule constants: U_env Delta = 0.3, C = 1.515, M_MIN = 3; U_min C = 2.38",
          abs(k["DELTA"] - 0.3) < 1e-12 and abs(k["C"] - 1.515) < 1e-9 and k["M_MIN"] == 3 and C3.PREREG3["umin_c_min"] == 2.38,
          "C = %r" % k["C"])
    rng = np.random.default_rng(7)
    sig40 = [float(s) for s in C3.eval_sigmas(C1.edm_sigmas(40))]
    sig_fine = list(np.exp(np.linspace(np.log(80), np.log(0.01), 300)))
    diff_min = diff_env = 0
    pw_min = pw_env = 0
    for t in range(4000):
        sig = sig40 if t % 2 == 0 else sig_fine
        n = len(sig)
        # valleys with flicker, ties (values on a coarse lattice) and rises of several shapes
        base = np.exp(rng.normal(0, 0.4, n)) * np.r_[np.linspace(1, 0.4, n // 2), np.linspace(0.4, rng.uniform(1, 12), n - n // 2)]
        us = list(np.round(base * 8) / 8 + 1e-3)
        if t % 4 == 1:  # plant an exact tie u_k = 2.38 x running minimum
            kk = int(rng.integers(2, n))
            us[kk] = 2.38 * min(us[:kk])
        if umin_literal(us, 2.38) != C2.replay_umin(us, 2.38):
            diff_min += 1
        if umin_literal(us, 2.38) != next((kk for kk in range(1, n) if us[kk] > 2.38 * min(us[:kk])), None):
            pw_min += 1
        if uenv_literal(sig, us) != uenv_c3(sig, us):
            diff_env += 1
        if uenv_literal(sig, us) != uenv_literal(sig, us, m_min=1):
            pw_env += 1
    check("U_min: the code equals the definition on 4,000 random traces with ties", diff_min == 0,
          "power: the strict-inequality version differs on %d" % pw_min)
    check("U_env: the code equals an independent implementation of the definition on 4,000 random traces", diff_env == 0,
          "power: the version without M_MIN differs on %d" % pw_env)
    check("rules: power checks bite (wrong versions differ on some traces)", pw_min > 0 and pw_env > 0)
    # cost formula
    check("cost: (0.05 c + n - c)/n; e.g. switch at call 21 of 79 -> 0.7475",
          abs((0.05 * 21 + 79 - 21) / 79 - 0.74746835443) < 1e-10)


# ---------------------------------------------------------------------------------------------------------------
# 3. CIFAR-10 (C1, C2)
# ---------------------------------------------------------------------------------------------------------------
def t_cifar():
    s1 = json.load(open(os.path.join(REC, "C1_results", "report", "summary_C1.json"), encoding="utf-8"))
    j1 = C1.judge(s1["results"])
    ok = all(j1[h] == s1["judgement"][h] for h in ("HC1", "HC2", "HC3", "HC4"))
    exp = {"HC1": SUP, "HC2": SUP, "HC3": SUP, "HC4": NOT}
    check("C1: HC1-HC4 recomputed from the records = notebook = Table 1 (supported, supported, supported, not)",
          ok and all(j1[h] == exp[h] for h in exp))
    calls = [json.load(open(os.path.join(REC, "C1_results", "runs", "S_gap_N018_s0", "b%03d.json" % b)))["switch_call"] for b in range(10)]
    cost = np.mean([(0.05 * c + 35 - c) / 35 for c in calls])
    sw = sorted(round(x, 1) for x in j1["switch_sigmas_S_gap"])
    check("C1: Table 1 numbers (FID 4.0734, tol 4.2060, cost 0.8263, 10/10 at sigma >= 8.40, best teleport 0.657)",
          r(j1["fid_S_gap"], 4) == 4.0734 and r(j1["fid_tol"], 4) == 4.2060 and r(cost, 4) == 0.8263
          and abs(cost - j1["cost_S_gap"]) < 1e-12 and j1["HC1_frac"] == 1.0 and r(j1["sigma_safe"], 2) == 8.40
          and r(j1["WV_tele_best"]["cost"], 3) == 0.657 and sw.count(28.4) == 4 and sw.count(19.4) == 6,
          "switch calls %s" % calls)
    pil = json.load(open(os.path.join(REC, "C1_results", "pilot.json"), encoding="utf-8"))
    br = C1.decide_branch(pil["rows"])
    check("C1: the pilot chose the alternative baseline (ratio 42.3), as registered",
          br["branch"] == pil["decision"]["branch"] == "first" and r(br["ratio_low_over_high"], 1) == 42.3)
    import analyze_c2 as A2
    o = A2.main()
    check("C2: HC5-HC8 recomputed = notebook = Table 1 (not, not, supported, not)",
          all(o["same_as_summary"].values()) and o["judgement"] == {"HC5": NOT, "HC6": NOT, "HC7": SUP, "HC8": NOT})
    ve, vp = o["detail_VE_main"], o["detail_VP_main"]
    check("C2: Table 1 numbers (VE 4.0395 vs 4.1631, cost 0.9186; VP 4.0543 vs 4.2060, cost 0.772; grid limit 0.757)",
          r(ve["fid_rule"], 4) == 4.0395 and r(ve["fid_tol"], 4) == 4.1631 and r(ve["cost_rule"], 4) == 0.9186
          and r(vp["fid_rule"], 4) == 4.0543 and r(vp["cost_rule"], 3) == 0.772
          and r(ve["WV_tele_best"]["cost"] + 0.10, 3) == 0.757 and r(vp["WV_tele_best"]["cost"] + 0.10, 3) == 0.757)
    bad = 0
    for key in ("vp_2.38", "ve_2.38", "vp_7.35", "ve_7.35"):
        c = float(key.split("_")[1])
        for row in o["per_batch"][key]:
            k = C2.replay_umin(row["u"], c)
            if k != len(row["u"]) - 1 or umin_literal(row["u"], c) != k or abs(row["cost_from_call"] - row["cost"]) > 1e-12:
                bad += 1
    check("C2: replaying U_min on the 40 recorded traces gives the recorded switches and costs", bad == 0)
    sp = o["ve_spike"]
    ratios = [x for b in sp for x in b["ratio"]]
    nz = o["same_sigma_pair_ratio"]["ve"]
    check("C2: the spike - every VE batch switched at sigma = 40.79; both evaluations there 2.43-6.20 x the minimum; "
          "same-level pairs elsewhere at most 1.73 (median 1.22)",
          all(abs(s - 40.786) < 1e-3 for s in ve["switch_sigmas"]) and r(min(ratios), 2) == 2.43 and r(max(ratios), 2) == 6.20
          and r(nz["max"], 2) == 1.73 and r(nz["median"], 2) == 1.22 and len(ratios) == 20)
    pil = o["pilot"]
    check("C2: pilot neighbour ratios in the fog - VE up to 7.4, VP up to 1.24 (Sec. 2.2)",
          r(pil["VE"]["max_neighbor_ratio_sigma_ge_10"], 1) == 7.4 and r(pil["VP"]["max_neighbor_ratio_sigma_ge_10"], 2) == 1.24)
    post = o["posthoc_ve_2.38_without_spike"]
    sw = sorted(round(p["switch_sigma"], 2) for p in post)
    check("C2 (after the fact, not evidence): without the spike, 2.38 would switch at 8.40 (7) and 12.91 (3), cost ~0.71",
          sw.count(8.4) == 7 and sw.count(12.91) == 3 and r(np.mean([p["cost_if_evaluated"] for p in post]), 2) == 0.71)


# ---------------------------------------------------------------------------------------------------------------
# 4. AFHQv2 (C3 main test, C6 replication)
# ---------------------------------------------------------------------------------------------------------------
def t_afhq():
    import analyze_c3 as A3
    import analyze_c6 as A6
    o3 = A3.main()
    exp = {"vp": {"HC9": SUP, "HC10": SUP, "HC13": SUP, "HC14": SUP}, "ve": {"HC11": SUP, "HC12": SUP, "HC15": SUP, "HC16": SUP}}
    check("C3: HC9-HC16 recomputed = notebook = Table 1 (all 8 supported)",
          all(o3[n]["same_as_notebook"] and o3[n]["judgement"] == exp[n] for n in exp))
    t1 = {("vp", "U_env"): (3.7636, 0.7667), ("vp", "U_min"): (3.7828, 0.7475), ("ve", "U_env"): (3.9822, 0.7378), ("ve", "U_min"): (3.9601, 0.7378)}
    ok = all(r(o3[n]["rules"][m]["fid"], 4) == f and r(o3[n]["rules"][m]["cost"], 4) == c for (n, m), (f, c) in t1.items())
    ok &= r(o3["vp"]["tol"], 4) == 3.8864 and r(o3["ve"]["tol"], 4) == 4.0097
    ok &= r(o3["vp"]["grid_fid_upper"], 4) == 3.9623 and r(o3["ve"]["grid_fid_upper"], 4) == 4.0490
    ok &= r(o3["vp"]["grid_cost_upper"], 4) == 0.7709 and r(o3["ve"]["grid_cost_upper"], 4) == 0.8722
    check("C3: Table 1 numbers (FIDs, tolerances, costs, grid limits)", ok)
    check("C3: HC10 cost margin 0.0042; VE FID margins 0.027 / 0.050; second-seed spread 0.064 (VE)",
          r(o3["vp"]["rules"]["U_env"]["margin_cost_to_grid"], 4) == 0.0042 and r(o3["ve"]["rules"]["U_env"]["margin_fid_to_tol"], 3) == 0.027
          and r(o3["ve"]["rules"]["U_min"]["margin_fid_to_tol"], 3) == 0.050 and r(o3["ve"]["seed_spread"], 3) == 0.064)
    sav = {n: {m: 1 - o3[n]["rules"][m]["cost"] for m in ("U_env", "U_min")} for n in ("vp", "ve")}
    check("C3: savings 23.3-26.2%", r(min(v for d in sav.values() for v in d.values()) * 100, 1) == 23.3
          and r(max(v for d in sav.values() for v in d.values()) * 100, 1) == 26.2)
    delays = {n: [p["delay_evals"] for p in o3[n]["per_batch"]] for n in ("vp", "ve")}
    check("C3: U_env minus U_min switch = [-2,-2,0,-4,0] (VP) and [0,-2,-2,+2,+2] (VE) evaluations",
          delays["vp"] == [-2, -2, 0, -4, 0] and delays["ve"] == [0, -2, -2, 2, 2], str(delays))
    o6 = A6.main()
    for n in ("vp", "ve"):
        x = o6["nets"][n]
        check("C6 %s: pooled and new-seed-only judgments recomputed = notebook, all supported; supported on each seed set" % n.upper(),
              all(v["same"] and v["a"] == SUP and v["b"] == SUP for v in x["recomputed"].values())
              and all(v == SUP for v in x["hc"].values()) and all(v == SUP for v in x["all_three"].values())
              and all(x["per_seedset"][s][m][c] == SUP for s in x["per_seedset"] for m in ("U_env", "U_min") for c in ("a", "b")))
        check("C6 %s: costs recomputed from per-batch records = summary; seed set 0 = C3 records" % n.upper(),
              x["costs_recomputed_equal"] and x["seed0_matches_C3"])
        bad = []
        for s in A6.SEEDS:
            for rule in ("U_env", "U_min"):
                for j in A6.runs(n, rule, s):
                    sg = [t[1] for t in j["trace"]]
                    u = [t[3] for t in j["trace"]]
                    got = uenv_c3(sg, u) if rule == "U_env" else C2.replay_umin(u, 2.38)
                    lit = uenv_literal(sg, u) if rule == "U_env" else umin_literal(u, 2.38)
                    if got != len(u) - 1 or lit != got or j["trace"][got][0] != j["switch_call"] \
                            or abs((0.05 * j["switch_call"] + 79 - j["switch_call"]) / 79 - j["cost_ratio"]) > 1e-12:
                        bad.append((s, rule, j["batch"]))
        check("C6 %s: replaying both rules on all 30 recorded traces gives the recorded switches and costs" % n.upper(), not bad, str(bad))
        check("C6 %s: the two rules see identical signal values up to the earlier switch (15 batches)" % n.upper(),
              all(t["prefix_same"] for s in A6.SEEDS for t in x["seed_traces"][s]))
    vp, ve = o6["nets"]["vp"]["recomputed"], o6["nets"]["ve"]["recomputed"]
    T1 = {  # Table 1, C6 rows: (FID, tol a, cost, tol b)
        ("vp", "pooled_main_U_env"): (2.4957, 2.5621, 0.7523, 2.7175), ("vp", "pooled_main_U_min"): (2.5054, 2.5621, 0.7475, 2.7175),
        ("ve", "pooled_main_U_env"): (2.7393, 2.7525, 0.7330, 2.7692), ("ve", "pooled_main_U_min"): (2.7320, 2.7525, 0.7346, 2.7692),
        ("vp", "pooled_new_only_U_env"): (2.8215, 2.8932, 0.7451, None), ("vp", "pooled_new_only_U_min"): (2.8161, 2.8932, 0.7475, None),
        ("ve", "pooled_new_only_U_env"): (3.0602, 3.0974, 0.7306, None), ("ve", "pooled_new_only_U_min"): (3.0623, 3.0974, 0.7330, None)}
    ok = True
    for (nn, kk), (f, t, c, tb) in T1.items():
        d = vp if nn == "vp" else ve
        v = d[kk]
        ok &= r(v["fid_rule"], 4) == f and r(v["fid_tol"], 4) == t and r(v["cost"], 4) == c
        if tb is not None:
            ok &= r(v["fid_grid_upper"], 4) == tb and r(v["cost_grid_upper"], 4) == (0.7709 if d is vp else 0.8722)
    check("C6: Table 1 numbers (pooled 15k and new-seed-only 10k)", ok)
    # Table 2
    ok = True
    T2 = {("vp", "U_env"): ([-0.010, 0.032, -0.010], [0.123, 0.081, 0.122], (0.008, 0.066, 0.075), (0.013, 0.072, 0.084)),
          ("vp", "U_min"): ([0.010, 0.005, 0.004], [0.104, 0.107, 0.108], (0.018, 0.057, 0.075), (0.007, 0.077, 0.084)),
          ("ve", "U_env"): ([0.089, 0.056, 0.069], [0.027, 0.063, 0.048], (0.067, 0.013, 0.080), (0.053, 0.037, 0.090)),
          ("ve", "U_min"): ([0.067, 0.077, 0.044], [0.050, 0.043, 0.073], (0.060, 0.020, 0.080), (0.055, 0.035, 0.090))}
    for (n, m), (dif, mar, pool, new) in T2.items():
        x = o6["nets"][n]
        ok &= [r(x["per_seedset"][s][m]["diff"], 3) for s in (0, 2, 3)] == dif
        ok &= [r(x["per_seedset"][s][m]["margin_fid_to_tol"], 3) for s in (0, 2, 3)] == mar
        p, q = x["recomputed"]["pooled_main_" + m], x["recomputed"]["pooled_new_only_" + m]
        ok &= (r(p["diff_rule_minus_M0"], 3), r(p["margin_fid_to_tol"], 3), r(p["tol_width"], 3)) == pool
        ok &= (r(q["diff_rule_minus_M0"], 3), r(q["margin_fid_to_tol"], 3), r(q["tol_width"], 3)) == new
    ok &= r(o6["nets"]["vp"]["fid_M0_spread"], 3) == 0.072 and r(o6["nets"]["ve"]["fid_M0_spread"], 3) == 0.077
    check("C6: every number of Table 2 (paired differences, margins, widths, seed spreads 0.072 / 0.077)", ok)
    sav = [1 - vp["pooled_main_U_env"]["cost"], 1 - vp["pooled_main_U_min"]["cost"], 1 - ve["pooled_main_U_env"]["cost"], 1 - ve["pooled_main_U_min"]["cost"]]
    check("C6: pooled savings 24.8%, 25.3% (VP) and 26.7%, 26.5% (VE)", [r(s * 100, 1) for s in sav] == [24.8, 25.3, 26.7, 26.5])
    fog = [t["fog_max_ratio"] for s in A6.SEEDS for t in o6["nets"]["ve"]["seed_traces"][s]]
    at = {round(t["fog_at_sigma"], 2) for s in A6.SEEDS for t in o6["nets"]["ve"]["seed_traces"][s]}
    fvp = [t["fog_max_ratio"] for s in A6.SEEDS for t in o6["nets"]["vp"]["seed_traces"][s]]
    mx = max(t["max_ratio_before_umin_switch"] for n in ("vp", "ve") for s in A6.SEEDS for t in o6["nets"][n]["seed_traces"][s])
    check("flicker: VE fog maximum at sigma = 27.84 in all 15 batches, 1.67-2.28 (closest 0.096 below 2.38); VP 1.34-1.85; "
          "largest pre-switch ratio 2.371",
          at == {27.84} and r(min(fog), 2) == 1.67 and r(max(fog), 2) == 2.28 and r(2.38 - max(fog), 3) == 0.096
          and r(min(fvp), 2) == 1.34 and r(max(fvp), 2) == 1.85 and r(mx, 3) == 2.371)
    dl = [t["delay_evals"] for n in ("vp", "ve") for s in A6.SEEDS for t in o6["nets"][n]["seed_traces"][s]]
    check("C6: U_env minus U_min between -4 and +3 evaluations over 30 batches", min(dl) == -4 and max(dl) == 3)
    # Table 3 (price) and HC25
    B = o6["B"]
    check("HC25 / fine grid: k=10 3.945 and k=11 3.965 within 4.0097, k=12 4.045 outside; k_best = 11 at cost 0.7215",
          B["HC25"] == SUP and B["k_best_fine"] == 11 and B["recomputed_within"] == {"10": True, "11": True, "12": False}
          and [r(B["grid"][k]["fid"], 3) for k in ("10", "11", "12")] == [3.945, 3.965, 4.045]
          and r(B["tol"], 4) == 4.0097 and r(B["grid"]["11"]["cost"], 4) == 0.7215)
    vpc = 53 / 79  # VP coarse-grid best k = 13
    price = [r(B["price_vs_fine_grid_seed0"]["U_env"], 3), r(B["price_vs_fine_grid_pooled"]["U_env"], 3), r(B["price_vs_fine_grid_pooled"]["U_min"], 3),
             r(o3["ve"]["rules"]["U_env"]["cost"] - 0.7721518987341772, 3),
             r(o3["vp"]["rules"]["U_env"]["cost"] - vpc, 3), r(o3["vp"]["rules"]["U_min"]["cost"] - vpc, 3),
             r(vp["pooled_main_U_env"]["cost"] - vpc, 3), r(vp["pooled_main_U_min"]["cost"] - vpc, 3)]
    check("Table 3 (price of not looking): VE fine +0.016 (seed 0), +0.012/+0.013 (pooled); VE coarse -0.034; "
          "VP coarse +0.096/+0.077 (seed 0), +0.081/+0.077 (pooled)",
          price == [0.016, 0.012, 0.013, -0.034, 0.096, 0.077, 0.081, 0.077], str(price))
    # FID against 1/N (Sec. 6.1, observation)
    e, f = o6["nets"]["ve"]["fid_vs_n"], o6["nets"]["vp"]["fid_vs_n"]
    check("Sec. 6.1: a + b/N fits - VE a = 2.06, width 0.062, differences 0.055; VP a = 1.86, width 0.056, differences 0.014-0.019; "
          "largest residual 0.013",
          r(e["M0"]["a_inf"], 2) == 2.06 and r(e["tol_width_inf"], 3) == 0.062 and r(e["U_env"]["diff_inf"], 3) == 0.055
          and r(e["U_min"]["diff_inf"], 3) == 0.055 and r(f["M0"]["a_inf"], 2) == 1.86 and r(f["tol_width_inf"], 3) == 0.056
          and r(f["U_env"]["diff_inf"], 3) == 0.014 and r(f["U_min"]["diff_inf"], 3) == 0.019
          and r(max(d[k]["fit_resid_max"] for d in (e, f) for k in ("M0", "U_env", "U_min")), 3) == 0.013)


def t_figures():
    import make_paper_figures as F
    with tempfile.TemporaryDirectory() as td:
        F.OUT = td
        names = F.main()
    check("figures: the six paper figures are drawn from the records", len([x for x in names if x.endswith(".pdf")]) == 6)


def t_demo():
    """The demo notebook: Part 1 runs on CPU; it embeds no copy of the rule code; its live part runs end to end."""
    import matplotlib
    matplotlib.use("Agg")
    import importlib.util
    nbp = os.path.join(ROOT, "notebooks", "demo_colab.ipynb")
    nb = json.load(open(nbp, encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("mkdemo", os.path.join(ROOT, "notebooks", "make_demo_notebook.py"))
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    with tempfile.TemporaryDirectory() as td:
        mk.HERE = td
        fresh = json.load(open(mk.build(), encoding="utf-8"))
    check("demo: notebooks/demo_colab.ipynb is exactly what make_demo_notebook.py writes", fresh == nb)
    srcs = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    embedded = [s for s in srcs if s.startswith("%%writefile")]
    check("demo: the notebook embeds no copy of the rule code; it imports c1core-c3core and demo_core from the repository",
          not embedded and any("import demo_core as DC, c1core as C1, c3core as C3" in x for x in srcs))
    unz = [c for c in nb["cells"] if c["cell_type"] == "code" and "unzip" in c["metadata"].get("tags", [])]
    lines = "".join(unz[0]["source"]).splitlines() if len(unz) == 1 else []
    check("demo: a commented cell (before the setup) unzips the repository zips for use before the repository is public",
          len(unz) == 1 and nb["cells"].index(unz[0]) == 1 and all(x.startswith("#") for x in lines)
          and '# !cd /content && for f in fog-lift-switch_part0*.zip; do unzip -q -o "$f"; done' in lines)
    cwd = os.getcwd()
    os.chdir(os.path.join(ROOT, "notebooks"))
    ns, ok, err = {}, True, ""
    t0 = __import__("time").time()
    try:
        import matplotlib.pyplot as plt
        plt.show = lambda *a, **k: None
        for c in nb["cells"]:
            if c["cell_type"] == "code" and set(c["metadata"].get("tags", [])) & {"setup", "part1"}:
                exec("".join(c["source"]), ns)
        plt.close("all")
    except Exception as e:  # noqa: BLE001
        ok, err = False, repr(e)
    finally:
        os.chdir(cwd)
    check("demo Part 1 runs on CPU: 80 recorded switches reproduced, 25 decisions identical to Table 1",
          ok and sum(r["match"] for r in ns.get("rows", [])) == 80 and ns.get("same") is True,
          err or "%.1f s" % (__import__("time").time() - t0))
    for c in nb["cells"]:
        if c["cell_type"] == "code" and "part2" in c["metadata"].get("tags", []):
            compile("".join(c["source"]), "part2", "exec")
    import demo_core as DC
    sys.path.insert(0, HERE)
    import demo_synth as S
    mu, cov, net_fn = S.make()
    G = C1.GaussianModel.from_cov(mu, cov)
    res = DC.run_live(net_fn, G, np.random.default_rng(1).standard_normal((256, 48)), C1.edm_sigmas(18))
    tm, te = res["U_min"]["info"]["trace"], res["U_env"]["info"]["trace"]
    n = min(len(tm), len(te))
    check("demo Part 2 pipeline (synthetic network on CPU): both rules switch, save evaluations, and see identical u up to the earlier switch",
          res["U_min"]["info"]["switched"] and res["U_env"]["info"]["switched"] and res["U_min"]["cost"] < 1 and res["U_env"]["cost"] < 1
          and all(a[3] == b[3] for a, b in zip(tm[:n], te[:n])) and len(res["u_along_M0"]) == 35
          and DC.image_grid(res["M0"]["x"], (3, 4, 4), 8).shape == (4, 32, 3),
          "U_min %.3f, U_env %.3f of M0" % (res["U_min"]["cost"], res["U_env"]["cost"]))


def t_window():
    """Reviewer point: the U_env window on the two schedules, recomputed from the schedules (Table 6 of the paper)."""
    import uenv_window as W
    r18, r40 = W.window_table(18), W.window_table(40)
    s18, s40 = W.summarize(r18), W.summarize(r40)
    check("U_env window, 18 steps (CIFAR-10): first step 0.329 in ln sigma > Delta = 0.3; every window from k = 2 has 3 evaluations",
          r(s18["first_step_dln_sigma"], 3) == 0.329 and s18["counts_window_n_from_k2"] == {3: 33})
    check("U_env window, 40 steps (AFHQv2): first step 0.141 < Delta; windows from k = 2: 3 evals x48, 4 x24, 5 x3, 6 x2",
          r(s40["first_step_dln_sigma"], 3) == 0.141 and s40["counts_window_n_from_k2"] == {3: 48, 4: 24, 5: 3, 6: 2})
    band = [x for x in r40 if 9.0 < x["sigma"] < 21.0]
    check("40 steps, sigma 9.7-20: windows of 3 (first evaluation of a level) and 4 (second); span 0.169-0.187; "
          "smooth-rise factor C exp(2 span) = 2.12-2.20 (not 2.76)",
          {x["window_n"] for x in band} == {3, 4} and r(min(x["span_ln_sigma"] for x in band), 3) == 0.169
          and r(max(x["span_ln_sigma"] for x in band), 3) == 0.187 and r(min(x["smooth_rise_factor"] for x in band), 2) == 2.12
          and r(max(x["smooth_rise_factor"] for x in band), 2) == 2.20)
    S = W.switch_states()
    e = [x for x in S if x["delay_evals"] < 0]
    check("G2 (records, 30 AFHQv2 batches): every U_env switch is at the first evaluation of a level (window of 3)",
          len(S) == 30 and all(x["at_env"]["window_n"] == 3 and x["k_env"] % 2 == 1 for x in S))
    check("G2: U_env earlier in 14 batches, same in 7, later in 9; when earlier, u_k / running min was only 1.93-2.35 (< 2.38) "
          "while the whole window was 1.62-2.21 x running min, with P_k / running min 1.06-1.57 over all batches",
          (len(e), sum(x["delay_evals"] == 0 for x in S), sum(x["delay_evals"] > 0 for x in S)) == (14, 7, 9)
          and r(min(x["at_env"]["umin_ratio"] for x in e), 2) == 1.93 and r(max(x["at_env"]["umin_ratio"] for x in e), 2) == 2.35
          and r(min(x["at_env"]["min_window"] / x["at_env"]["running_min"] for x in e), 2) == 1.62
          and r(max(x["at_env"]["min_window"] / x["at_env"]["running_min"] for x in e), 2) == 2.21
          and r(min(x["at_env"]["P_over_running_min"] for x in S), 2) == 1.06 and r(max(x["at_env"]["P_over_running_min"] for x in S), 2) == 1.57)


def t_fast():
    """fast_gauss: same switches, same signal and same samples (up to rounding) as c3core.run_signal_rule."""
    import fast_gauss as FG
    sys.path.insert(0, HERE)
    import demo_synth as S
    k = C3.PREREG3["b22_constants"]
    mk = {"U_min": lambda: C3.UMinAdapter(2.38), "U_env": lambda: C3.UEnvRule(k["DELTA"], k["C"], k["M_MIN"], k["window_tol"])}
    n_runs = same = 0
    du = dx = 0.0
    prods = []
    for steps in (18, 40):
        for seed in range(4):
            mu, cov, net_fn = S.make(d=64, seed=seed, err=(0.005, 0.01, 0.03, 0.01)[seed])
            G = C1.GaussianModel.from_cov(mu, cov)
            sig = C1.edm_sigmas(steps)
            x0 = np.random.default_rng(100 + seed).standard_normal((200, 64)) * float(sig[0])
            for name, f in mk.items():
                xa, ia = C3.run_signal_rule(f(), x0, sig, net_fn, G, np.random.default_rng(seed))
                xb, ib = FG.run_signal_rule_fast(f(), x0, sig, net_fn, G, np.random.default_rng(seed))
                n_runs += 1
                same += ia["switch_call"] == ib["switch_call"] and ia["net_points"] == ib["net_points"] and len(ia["trace"]) == len(ib["trace"])
                ua, ub = np.array([t[3] for t in ia["trace"]]), np.array([t[3] for t in ib["trace"]])
                if len(ua) == len(ub):
                    du = max(du, float(np.max(np.abs(ua - ub) / np.abs(ua))))
                dx = max(dx, float(np.max(np.abs(xa - xb)) / np.max(np.abs(xa))))
                prods.append((2 * (ia["switch_call"] + 1), ib["gauss_full_batch_products"]))
    check("fast_gauss: identical switching call and network evaluations in %d runs (d = 64, batch 200, 18 and 40 steps, both rules)" % n_runs,
          same == n_runs)
    check("fast_gauss: u traces and final samples agree up to rounding", du < 1e-9 and dx < 1e-8, "max rel. diff u %.1e, samples %.1e" % (du, dx))
    check("fast_gauss: full-batch d x d products in the Gaussian phase drop from 2 per call to at most 4 in total",
          all(b <= 4 for a, b in prods), "current: %d-%d, fast: %d-%d" % (min(a for a, b in prods), max(a for a, b in prods),
                                                                         min(b for a, b in prods), max(b for a, b in prods)))
    # power check: a version whose corrector uses the predictor factor only (Euler) must differ
    src = open(os.path.join(ROOT, "fast_gauss.py"), encoding="utf-8").read()
    bad = src.replace("F = F + h * (0.5 * g(tc) * F + 0.5 * g(tn) * Fp)", "F = Fp")
    assert bad != src
    ns = {}
    exec(compile(bad, "fast_gauss_wrong", "exec"), ns)
    mu, cov, net_fn = S.make(d=64, seed=1)
    G = C1.GaussianModel.from_cov(mu, cov)
    sig = C1.edm_sigmas(40)
    x0 = np.random.default_rng(101).standard_normal((200, 64)) * 80.0
    never = lambda: C3.UMinAdapter(1e30)  # never switches: the whole run is the Gaussian phase
    xa, ia = C3.run_signal_rule(never(), x0, sig, net_fn, G, np.random.default_rng(1))
    xb, ib = FG.run_signal_rule_fast(never(), x0, sig, net_fn, G, np.random.default_rng(1))
    xw, iw = ns["run_signal_rule_fast"](never(), x0, sig, net_fn, G, np.random.default_rng(1))
    ok_never = not ia["switched"] and not ib["switched"] and float(np.max(np.abs(xa - xb)) / np.max(np.abs(xa))) < 1e-10
    pw = float(np.max(np.abs(xw - xa)) / np.max(np.abs(xa)))
    check("fast_gauss: a run that never switches (all Gaussian) agrees with c3core; power check: an Euler-only corrector differs",
          ok_never and pw > 1e-2, "wrong version: %.1e" % pw)
    import importlib.util
    spec = importlib.util.spec_from_file_location("mktime", os.path.join(ROOT, "notebooks", "make_timing_notebook.py"))
    mkt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mkt)
    nb = json.load(open(os.path.join(ROOT, "notebooks", "timing_colab.ipynb"), encoding="utf-8"))
    with tempfile.TemporaryDirectory() as td:
        mkt.HERE = td
        fresh = json.load(open(mkt.build(), encoding="utf-8"))
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), "timing", "exec")
    check("timing notebook: notebooks/timing_colab.ipynb is exactly what make_timing_notebook.py writes, and every cell parses", fresh == nb)


def t_timing():
    """The measured wall-clock time of fast_gauss (records/C10_timing, Table 7 of the paper) and the wall-clock ranges in the text."""
    J = json.load(open(os.path.join(REC, "C10_timing", "timing_C10.json"), encoding="utf-8"))
    ok = True
    for net in ("vp", "ve"):
        R = J[net]
        rd = os.path.join(REC, "C3_results", net, "runs")
        rm = json.load(open(os.path.join(rd, "U_min_c2.38_N040_s0", "b000.json"), encoding="utf-8"))
        re_ = json.load(open(os.path.join(rd, "U_env_N040_s0", "b000.json"), encoding="utf-8"))
        m0 = json.load(open(os.path.join(rd, "M0_N040_s0", "b000.json"), encoding="utf-8"))
        sw = R["switch_call"]
        ok &= sw["U_min fast"] == sw["U_min recorded"] == rm["switch_call"] == R["gauss_phase_fast_U_min"][1] == R["gauss_phase_current_U_min"][1]
        ok &= sw["U_env fast (eval index)"] == sw["U_env recorded (eval index)"] == len(re_["trace"]) - 1 == re_["switch_call"] == R["gauss_phase_fast_U_env"][1]
        ok &= abs(R["M0_recorded_seconds"] - m0["seconds"]) < 1e-6
        cur = (R["gauss_phase_current_U_min"][0] + R["fast_full_U_min_seconds"] - R["gauss_phase_fast_U_min"][0]) / m0["seconds"]
        fast = R["fast_full_U_min_seconds"] / m0["seconds"]
        rec = rm["seconds"] / m0["seconds"]
        w = R["wall_ratio_to_M0"]
        ok &= abs(cur - w["current (pre + post)"]) < 1e-12 and abs(fast - w["fast (measured full run)"]) < 1e-12 and abs(rec - w["recorded C3 U_min run"]) < 1e-12
        ok &= abs(R["cost_ratio"] - (0.05 * 21 + 79 - 21) / 79) < 1e-12 and abs(rm["cost_ratio"] - R["cost_ratio"]) < 1e-12
    vp, ve = J["vp"], J["ve"]
    table7 = [vp["switch_call"]["U_min fast"], ve["switch_call"]["U_min fast"], vp["switch_call"]["U_env fast (eval index)"], ve["switch_call"]["U_env fast (eval index)"],
              float("%.1e" % vp["u_trace_max_rel_diff"]), float("%.1e" % ve["u_trace_max_rel_diff"]),
              vp["images_vs_record"]["max_abs_diff_0_255"], ve["images_vs_record"]["max_abs_diff_0_255"],
              r(100 * vp["images_vs_record"]["pixels_differing_fraction"], 3), r(100 * ve["images_vs_record"]["pixels_differing_fraction"], 3),
              r(vp["gauss_phase_current_U_min"][0], 1), r(vp["gauss_phase_fast_U_min"][0], 1), r(ve["gauss_phase_current_U_min"][0], 1), r(ve["gauss_phase_fast_U_min"][0], 1),
              r(vp["wall_ratio_to_M0"]["current (pre + post)"], 3), r(ve["wall_ratio_to_M0"]["current (pre + post)"], 3),
              r(vp["wall_ratio_to_M0"]["fast (measured full run)"], 3), r(ve["wall_ratio_to_M0"]["fast (measured full run)"], 3),
              r(vp["wall_ratio_to_M0"]["recorded C3 U_min run"], 3), r(ve["wall_ratio_to_M0"]["recorded C3 U_min run"], 3),
              r(vp["cost_ratio"], 4), r(ve["cost_ratio"], 4)]
    want = [21, 21, 19, 21, 8.8e-15, 1.3e-14, 1, 2, 0.024, 0.020, 41.6, 11.8, 42.0, 12.3, 0.801, 0.805, 0.753, 0.759, 0.805, 0.805, 0.7475, 0.7475]
    check("timing (records/C10_timing): every number of Table 7 recomputed from the json and the C3 records of batch 0 "
          "(same switches as recorded, ratios to the recorded M0 time)", ok and table7 == want, str(table7))
    check("timing: images differ in 0.02% of pixels by at most 2/255; signal differs by at most 1.3e-14 (relative)",
          max(J[n]["images_vs_record"]["max_abs_diff_0_255"] for n in J) == 2 and r(100 * max(J[n]["images_vs_record"]["pixels_differing_fraction"] for n in J), 2) == 0.02
          and max(J[n]["u_trace_max_rel_diff"] for n in J) < 1.5e-14)
    # wall-clock ranges quoted in the text: current implementation from all 12 recorded (network, seed set, rule) groups
    ratios = []
    for net in ("vp", "ve"):
        for s, base in ((0, "C3_results"), (2, "C6_results"), (3, "C6_results")):
            d = os.path.join(REC, base, net, "runs")
            sec = lambda tag: np.mean([json.load(open(os.path.join(d, tag, "b%03d.json" % b), encoding="utf-8"))["seconds"] for b in range(5)])  # noqa: E731
            m0 = sec("M0_N040_s%d" % s)
            ratios += [sec("U_env_N040_s%d" % s) / m0, sec("U_min_c2.38_N040_s%d" % s) / m0]
    fast = [J[n]["wall_ratio_to_M0"]["fast (measured full run)"] for n in ("vp", "ve")]
    check("text: wall-clock 0.78-0.82 of M0 (saving 18-22%) for the implementation used for the runs (12 groups), "
          "24-25% for fast_gauss (0.753, 0.759)",
          len(ratios) == 12 and r(min(ratios), 2) == 0.78 and r(max(ratios), 2) == 0.82
          and int(100 * (1 - max(ratios))) == 18 and int(100 * (1 - min(ratios))) == 21 and r(100 * (1 - min(ratios)), 0) == 22
          and [int(100 * (1 - f)) for f in fast] == [24, 24] and r(100 * (1 - fast[0]), 0) == 25,
          "current %.3f-%.3f; fast %.3f, %.3f" % (min(ratios), max(ratios), fast[0], fast[1]))


def t_cifar_source():
    """Demo Part 2 data: the fingerprint is exact, order-independent and layout-sensitive; the loader falls back correctly."""
    import demo_core as DC
    rng = np.random.default_rng(3)
    X = rng.integers(0, 256, (400, 32, 32, 3), dtype=np.uint8)
    a = DC.cifar10_fingerprint(X, chunk=128)
    b = DC.cifar10_fingerprint(X[rng.permutation(400)], chunk=128)
    Y = X.copy()
    Y[17, 3, 4, 1] ^= 1
    c = DC.cifar10_fingerprint(Y, chunk=128)
    hwc = X.reshape(400, -1).reshape(400, 3, 32, 32).transpose(0, 2, 3, 1)  # same bytes read in the wrong layout
    d = DC.cifar10_fingerprint(np.ascontiguousarray(hwc), chunk=128)
    R = DC.to_chw_rows(X)
    ok_chw = np.array_equal(R[0, :1024].reshape(32, 32), X[0, :, :, 0])
    check("CIFAR-10 fingerprint: identical under reordering (byte hash is not), changed by one pixel and by a CHW/HWC mix-up; CHW rows as EDM",
          a["content_sha256"] == b["content_sha256"] and a["byte_sha256_in_order"] != b["byte_sha256_in_order"]
          and a["content_sha256"] != c["content_sha256"] and a["content_sha256"] != d["content_sha256"] and ok_chw)
    good = np.zeros(DC.CIFAR10_SHAPE, dtype=np.uint8)

    def boom():
        raise OSError("network down")
    X1, src1, _, f1 = DC.load_cifar10_train(loaders=[("first", lambda: good), ("second", boom)])
    X2, src2, _, f2 = DC.load_cifar10_train(loaders=[("first", boom), ("second", lambda: good)])
    X3, src3, _, f3 = DC.load_cifar10_train(loaders=[("first", lambda: good[:10]), ("second", lambda: good)])
    try:
        DC.load_cifar10_train(loaders=[("first", boom), ("second", boom)])
        none_ok = False
    except RuntimeError:
        none_ok = True
    check("CIFAR-10 loader: uses the first source that works, falls back on an error or a wrong shape, and stops if none works",
          (src1, src2, src3) == ("first", "second", "second") and not f1 and len(f2) == 1 and len(f3) == 1 and none_ok)
    import importlib.util
    spec = importlib.util.spec_from_file_location("mkcifar", os.path.join(ROOT, "notebooks", "make_cifar_check_notebook.py"))
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    nb = json.load(open(os.path.join(ROOT, "notebooks", "cifar10_source_check.ipynb"), encoding="utf-8"))
    with tempfile.TemporaryDirectory() as td:
        mk.HERE = td
        fresh = json.load(open(mk.build(), encoding="utf-8"))
    for c_ in nb["cells"]:
        if c_["cell_type"] == "code":
            compile("".join(c_["source"]), "cifar_check", "exec")
    demo = json.load(open(os.path.join(ROOT, "notebooks", "demo_colab.ipynb"), encoding="utf-8"))
    src = "".join("".join(c_["source"]) for c_ in demo["cells"])
    ok_note = (DC.CIFAR10_FINGERPRINT_PREFIX == "4713a5cd450ffa7c"
               and "matches the fingerprint measured" in DC.fingerprint_note({"content_sha256": "4713a5cd450ffa7c" + "0" * 48})
               and DC.fingerprint_note({"content_sha256": "f" * 64}).startswith("NOTE:"))
    check("demo: the fingerprint is compared with 4713a5cd450ffa7c (match -> 'matches ...', mismatch -> a note, no stop); "
          "the description has the new licence, HF_TOKEN and approximate-time sentences and no per-step seconds",
          ok_note and "DC.fingerprint_note(fp)" in src
          and "CIFAR-10 is downloaded from Hugging Face (`uoft-cs/cifar10`), or, if that fails, by `torchvision` from the University of Toronto, under its own terms." in src
          and "A warning about unauthenticated requests to the Hugging Face Hub (HF_TOKEN) may appear; it can be ignored." in src
          and "Part 1 takes seconds on a CPU; Part 2 takes a few minutes on a Colab GPU (downloads included)." in src
          and "5.5 s" not in src and "14 s" not in src)
    check("notebooks: cifar10_source_check.ipynb is what its generator writes and parses; the demo loads CIFAR-10 through "
          "load_cifar10_train (Hugging Face first) and prints the source, the time and the fingerprint",
          fresh == nb and "DC.load_cifar10_train(" in src and "DC.cifar10_fingerprint(" in src and "torchvision.datasets" not in src)


def main():
    t_integrity()
    t_exact()
    t_rules()
    t_cifar()
    t_afhq()
    t_figures()
    t_demo()
    t_window()
    t_fast()
    t_timing()
    t_cifar_source()
    n = sum(x["ok"] for x in RESULTS)
    print("\n%d/%d PASS" % (n, len(RESULTS)))
    out = os.path.join(ROOT, "analysis", "out")
    os.makedirs(out, exist_ok=True)
    json.dump({"pass": n, "total": len(RESULTS), "checks": RESULTS}, open(os.path.join(out, "verify_all_result.json"), "w"),
              indent=1, ensure_ascii=False)
    sys.exit(0 if n == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
