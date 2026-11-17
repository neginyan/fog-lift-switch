# analyze_c6.py -- C6 の結果を記録から計算し直し、報告用の数字と図を作る（MIT）
# 使い方：python analyze_c6.py   （records/C3_results・records/C6_results を読み、analysis/out/c6/ に書く）
# 注意：FID の特徴量（feats_*.npy）は持ち帰りの zip に入っていないので、FID の値は summary_C6.json と summary_C3.json のものを使う。
#       判定の式、費用（各束の記録から）、余裕、u の列、切り替えは、ここで計算し直す。
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import c3core as C3  # noqa: E402
import c6core as C6  # noqa: E402

R3 = os.path.join(ROOT, "records", "C3_results")
R6 = os.path.join(ROOT, "records", "C6_results")
OUT = os.path.join(HERE, "out", "c6")
P = C6.PREREG6
NB = 5
SEEDS = [0, 2, 3]


def runs(net, rule, s):
    base = R3 if s == 0 else R6
    d = os.path.join(C3.net_dir(base, net), "runs", C6.tag(rule, s))
    return [json.load(open(os.path.join(d, "b%03d.json" % b), encoding="utf-8")) for b in range(NB)]


def cost_of_call(c):
    return (0.05 * c + 79 - c) / 79


def main():
    os.makedirs(OUT, exist_ok=True)
    S = json.load(open(os.path.join(R6, "report", "summary_C6.json"), encoding="utf-8"))
    J = S["judgement"]
    out = {"summary_version": S["version"], "extra": S["extra"], "nets": {}}
    for net in P["nets"]:
        o = J["nets"][net]
        r = {"recomputed": {}, "per_seedset": {}, "seed_traces": {}}
        # 1) 判定の式を、summary の FID と、各束の記録から計算し直した費用で、もう一度当てる
        cost_ok = True
        for key, seeds in (("pooled_main", [0, 2, 3]), ("pooled_new_only", [2, 3])):
            for rule in ("U_env", "U_min"):
                d = o[key][rule]
                cs = [x["cost_ratio"] for s in seeds for x in runs(net, rule, s)]
                c = float(np.mean(cs))
                cost_ok &= abs(c - d["cost_rule"]) < 1e-12
                j = C6.judge_one(d["fid_rule"], c, d["fid_M0"], d["fid_grid"], d["cost_grid"], P)
                r["recomputed"]["%s_%s" % (key, rule)] = {
                    "a": j["a"], "b": j["b"], "same": (j["a"], j["b"]) == (d["a"], d["b"]),
                    "fid_rule": d["fid_rule"], "fid_M0": d["fid_M0"], "fid_tol": j["fid_tol"], "fid_grid": d["fid_grid"],
                    "fid_grid_upper": 1.03 * d["fid_grid"], "cost": c, "cost_grid_upper": d["cost_grid"] + 0.10,
                    "margin_fid_to_tol": j["margin_fid_to_tol"], "margin_fid_to_grid": 1.03 * d["fid_grid"] - d["fid_rule"],
                    "margin_cost_to_085": 0.85 - c, "margin_cost_to_grid": j["margin_cost_to_grid"],
                    "diff_rule_minus_M0": d["fid_rule"] - d["fid_M0"], "tol_width": 0.03 * d["fid_M0"]}
        r["costs_recomputed_equal"] = cost_ok
        hc = P["hc_pooled_main"][net]
        r["hc"] = {k: o[k] for rule in ("U_env", "U_min") for k in hc[rule]}
        r["all_three"] = {k: v for k, v in o.items() if k.startswith("all_three")}
        # 2) 種の組ごと
        for s in SEEDS:
            ps = o["per_seedset"][str(s)]
            r["per_seedset"][s] = {rule: {"a": ps[rule]["a"], "b": ps[rule]["b"], "fid": ps[rule]["fid_rule"],
                                          "fid_M0": ps[rule]["fid_M0"], "fid_grid": ps[rule]["fid_grid"],
                                          "diff": ps[rule]["fid_rule"] - ps[rule]["fid_M0"],
                                          "margin_fid_to_tol": ps[rule]["margin_fid_to_tol"], "cost": ps[rule]["cost_rule"],
                                          "margin_cost_to_grid": ps[rule]["margin_cost_to_grid"]} for rule in ("U_env", "U_min")}
        r["fid_M0_by_seedset"] = o["fid_M0_by_seedset"]
        r["fid_M0_spread"] = o["fid_M0_spread"]
        # 種の組 0 の値が C3 の記録と同じか
        s3 = json.load(open(os.path.join(C3.net_dir(R3, net), "report", "summary_C3.json"), encoding="utf-8"))["results"]
        r["seed0_matches_C3"] = all(abs(o["per_seedset"]["0"][rule]["fid_rule"] - s3[C6.tag(rule, 0)]["fid"]) < 1e-12
                                    for rule in ("U_env", "U_min")) and abs(o["fid_M0_by_seedset"][0] - s3["M0_N040_s0"]["fid"]) < 1e-12
        # 3) FID の枚数による違い：FID ≈ a + b/N で、M0 と規則それぞれ（5,000 枚は種の組 0・2・3 の平均）
        ext = {}
        for name in ("M0", "U_env", "U_min"):
            if name == "M0":
                f5 = float(np.mean([o["per_seedset"][str(s)]["U_env"]["fid_M0"] for s in SEEDS]))
                f10 = o["pooled_new_only"]["U_env"]["fid_M0"]
                f15 = o["pooled_main"]["U_env"]["fid_M0"]
            else:
                f5 = float(np.mean([o["per_seedset"][str(s)][name]["fid_rule"] for s in SEEDS]))
                f10 = o["pooled_new_only"][name]["fid_rule"]
                f15 = o["pooled_main"][name]["fid_rule"]
            x = np.array([1 / 5000, 1 / 10000, 1 / 15000])
            y = np.array([f5, f10, f15])
            b_, a_ = np.polyfit(x, y, 1)
            ext[name] = {"fid_5k_mean": f5, "fid_10k": f10, "fid_15k": f15, "a_inf": float(a_), "b": float(b_),
                         "fit_resid_max": float(np.max(np.abs(a_ + b_ * x - y)))}
        for name in ("U_env", "U_min"):
            ext[name]["diff_inf"] = ext[name]["a_inf"] - ext["M0"]["a_inf"]
        ext["tol_width_inf"] = 0.03 * ext["M0"]["a_inf"]
        r["fid_vs_n"] = ext
        # 4) u の列：種の組ごとに、霧の中の跳ね、U_min の最大の比、切り替え、二つの規則の差
        for s in SEEDS:
            te, tm = runs(net, "U_env", s), runs(net, "U_min", s)
            rows = []
            for b in range(NB):
                a_, m_ = te[b]["trace"], tm[b]["trace"]
                n = min(len(a_), len(m_))
                same = all(x[1] == y[1] and x[3] == y[3] for x, y in zip(a_[:n], m_[:n]))
                long = a_ if len(a_) >= len(m_) else m_
                u = [x[3] for x in long]
                sg = [x[1] for x in long]
                km, ke = tm[b]["switch_call"], te[b]["switch_call"]
                ratio = [None] + [u[k] / min(u[:k]) for k in range(1, len(u))]
                pre = [(k, ratio[k], sg[k]) for k in range(1, km)]
                kp, rp, sp = max(pre, key=lambda t: t[1])
                fog = [(k, ratio[k], sg[k]) for k in range(1, km) if sg[k] >= 20]
                kf, rf, sf = max(fog, key=lambda t: t[1])
                rows.append({"batch": b, "prefix_same": same, "switch_call_min": km, "switch_sigma_min": tm[b]["switch_sigma"],
                             "switch_call_env": ke, "switch_sigma_env": te[b]["switch_sigma"], "delay_evals": ke - km,
                             "max_ratio_before_umin_switch": rp, "at_sigma": sp, "fog_max_ratio": rf, "fog_at_sigma": sf,
                             "umin_ratio_at_switch": ratio[km], "u": u, "sigma": sg,
                             "cost_env_from_call": cost_of_call(ke), "cost_env_record": te[b]["cost_ratio"],
                             "cost_min_from_call": cost_of_call(km), "cost_min_record": tm[b]["cost_ratio"],
                             "seconds_env": te[b]["seconds"], "seconds_min": tm[b]["seconds"]})
            r["seed_traces"][s] = rows
        out["nets"][net] = r
    # 5) B
    B = J["B"]
    rules0 = {rule: json.load(open(os.path.join(C3.net_dir(R3, "ve"), "report", "summary_C3.json"), encoding="utf-8"))["results"][C6.tag(rule, 0)]
              for rule in ("U_env", "U_min")}
    k11 = B["grid"]["11"]
    out["B"] = {"grid": B["grid"], "tol": B["tol"], "fid_M0_s0": B["fid_M0_s0"], "HC25": B["HC25"], "k_best_fine": B["k_best_fine"],
                "recomputed_within": {k: v["fid"] <= 1.03 * B["fid_M0_s0"] for k, v in B["grid"].items()},
                "rules_seed0": {k: {"fid": v["fid"], "cost": v["cost_ratio_mean"]} for k, v in rules0.items()},
                "price_vs_fine_grid_seed0": {k: v["cost_ratio_mean"] - k11["cost"] for k, v in rules0.items()},
                "price_vs_fine_grid_pooled": {rule: J["nets"]["ve"]["pooled_main"][rule]["cost_rule"] - k11["cost"] for rule in ("U_env", "U_min")},
                "coarse_grid": {"9": {"fid": 3.9310887933154017, "sigma": float(C6.SIG[9]), "cost": 61 / 79},
                                "13": {"fid": None, "sigma": float(C6.SIG[13]), "cost": 53 / 79}}}
    s3ve = json.load(open(os.path.join(C3.net_dir(R3, "ve"), "report", "summary_C3.json"), encoding="utf-8"))["results"]
    out["B"]["coarse_grid"]["9"]["fid"] = s3ve["WV_tele_k09_N040_s0"]["fid"]
    out["B"]["coarse_grid"]["13"]["fid"] = s3ve["WV_tele_k13_N040_s0"]["fid"]
    # 6) 実時間（種の組 2・3）
    tm = {}
    for net in P["nets"]:
        tm[net] = {}
        d = os.path.join(C3.net_dir(R6, net), "runs")
        for t in sorted(os.listdir(d)):
            xs = [json.load(open(os.path.join(d, t, "b%03d.json" % b)))["seconds"] for b in range(NB)]
            tm[net][t] = {"sec_per_batch": float(np.mean(xs)), "hours": float(np.sum(xs) / 3600)}
    out["timing"] = tm
    json.dump(out, open(os.path.join(OUT, "c6_recomputed.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    figures(out)
    return out


def figures(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {0: "gray", 2: "C0", 3: "C1"}
    # 図 1：種の組ごとの u と切り替え
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    for ax, net in zip(axes, P["nets"]):
        for s in SEEDS:
            for i, row in enumerate(out["nets"][net]["seed_traces"][s]):
                ax.loglog(row["sigma"], row["u"], ".-", color=cols[s], alpha=0.35, lw=0.8, label=("seed set %d" % s) if i == 0 else None)
                ax.plot(row["sigma"][row["switch_call_min"]], row["u"][row["switch_call_min"]], "rx", ms=7)
                ax.plot(row["sigma"][row["switch_call_env"]], row["u"][row["switch_call_env"]], "b+", ms=9)
        ax.axvline(C6.SIG[P["k_best_c3"][net]], color="C2", lw=2, alpha=0.5, label="k_best (C3 grid)")
        ax.invert_xaxis()
        ax.set_xlabel("sigma")
        ax.set_title("%s: u (15 batches); x = U_min, + = U_env switch" % net.upper())
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figC7_1_u_three_seedsets.png"), dpi=130)
    plt.close(fig)
    # 図 2：FID と枚数（1/N）
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for ax, net in zip(axes, P["nets"]):
        e = out["nets"][net]["fid_vs_n"]
        xs = np.array([0, 1 / 15000, 1 / 10000, 1 / 5000])
        for name, mk in (("M0", "ko-"), ("U_env", "bs-"), ("U_min", "r^-")):
            ax.plot([1 / 5000, 1 / 10000, 1 / 15000], [e[name]["fid_5k_mean"], e[name]["fid_10k"], e[name]["fid_15k"]], mk, label=name)
            ax.plot(xs, e[name]["a_inf"] + e[name]["b"] * xs, ":", color=mk[0], lw=0.8)
        ax.plot(xs, 1.03 * (e["M0"]["a_inf"] + e["M0"]["b"] * xs), "k--", lw=0.8, label="1.03 x FID(M0)")
        ax.set_xlabel("1 / number of images")
        ax.set_ylabel("FID")
        ax.set_title("%s: FID vs 1/N (5k = mean of seed sets 0,2,3)" % net.upper())
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figC7_2_fid_vs_n.png"), dpi=130)
    plt.close(fig)
    # 図 3：B の細かい格子（VE 型、種の組 0）
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    B = out["B"]
    pts = sorted([(v["sigma"], v["fid"], int(k)) for k, v in B["grid"].items()] +
                 [(v["sigma"], v["fid"], int(k)) for k, v in B["coarse_grid"].items()])
    ax.semilogx([p[0] for p in pts], [p[1] for p in pts], "s-", color="gray", label="WV_tele(k), seed set 0")
    for p in pts:
        ax.annotate("k=%d" % p[2], (p[0], p[1]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(B["tol"], color="gray", ls=":", label="1.03 x FID(M0)")
    for rule, mk in (("U_env", "b*"), ("U_min", "r*")):
        sw = [row["switch_sigma_env" if rule == "U_env" else "switch_sigma_min"] for row in out["nets"]["ve"]["seed_traces"][0]]
        ax.plot(np.exp(np.mean(np.log(sw))), B["rules_seed0"][rule]["fid"], mk, ms=12, label="%s (seed 0, geo-mean switch sigma)" % rule)
    ax.invert_xaxis()
    ax.set_xlabel("sigma of the switch / teleport target")
    ax.set_ylabel("FID (5,000 images)")
    ax.set_title("VE: fine grid (k = 10, 11, 12) and the rules")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figC7_3_fine_grid.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    o = main()
    for net in P["nets"]:
        r = o["nets"][net]
        print("==", net, r["hc"], r["all_three"], "costs ok", r["costs_recomputed_equal"], "seed0=C3", r["seed0_matches_C3"])
        for k, v in r["recomputed"].items():
            print("  %-22s %s/%s same %s  FID %.4f M0 %.4f diff %+.4f tol %.4f (width %.4f) mFID %.4f  grid %.4f→up %.4f mgrid %.4f  cost %.4f m085 %.4f mgridcost %.4f" % (
                k, v["a"], v["b"], v["same"], v["fid_rule"], v["fid_M0"], v["diff_rule_minus_M0"], v["fid_tol"], v["tol_width"],
                v["margin_fid_to_tol"], v["fid_grid"], v["fid_grid_upper"], v["margin_fid_to_grid"], v["cost"], v["margin_cost_to_085"], v["margin_cost_to_grid"]))
        for s, d in r["per_seedset"].items():
            print("  seed", s, {k: (v["a"], v["b"], round(v["fid"], 4), round(v["diff"], 4), round(v["margin_fid_to_tol"], 4), round(v["cost"], 4), round(v["margin_cost_to_grid"], 4)) for k, v in d.items()})
        print("  M0 by seed", [round(x, 4) for x in r["fid_M0_by_seedset"]], "spread %.4f" % r["fid_M0_spread"])
        print("  fid_vs_n", {k: ({kk: round(vv, 4) for kk, vv in v.items()} if isinstance(v, dict) else round(v, 4)) for k, v in r["fid_vs_n"].items()})
        for s in SEEDS:
            rows = r["seed_traces"][s]
            print("  seed %d: prefix_same %s | U_min sw σ %s | U_env sw σ %s | delay %s | fog max %s at σ %s | max before sw %s | ratio@sw %s" % (
                s, all(x["prefix_same"] for x in rows), [round(x["switch_sigma_min"], 2) for x in rows], [round(x["switch_sigma_env"], 2) for x in rows],
                [x["delay_evals"] for x in rows], [round(x["fog_max_ratio"], 2) for x in rows], [round(x["fog_at_sigma"], 2) for x in rows],
                [round(x["max_ratio_before_umin_switch"], 3) for x in rows], [round(x["umin_ratio_at_switch"], 2) for x in rows]))
    print("B", json.dumps({k: v for k, v in o["B"].items() if k != "grid"}, ensure_ascii=False))
    print("timing", {n: {t: round(v["sec_per_batch"]) for t, v in d.items()} for n, d in o["timing"].items()})
