# analyze_c3.py -- 本番（C3、AFHQv2 64×64 の VP 型・VE 型）の結果を記録から計算し直し、報告用の数字と図を作る（MIT）
# 使い方：python analyze_c3.py   （records/C3_results を読み、analysis/out/c3/ に書く）
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import c1core as C1  # noqa: E402
import c2core as C2  # noqa: E402
import c3core as C3  # noqa: E402

R = os.environ.get("C3_RESULTS", os.path.join(ROOT, "records", "C3_results"))
OUT = os.path.join(HERE, "out", "c3")
P = C3.PREREG3
NB = 5
TAG_E, TAG_M = "U_env_N040_s0", "U_min_c2.38_N040_s0"


def runs(net, tag):
    d = os.path.join(R, net, "runs", tag)
    return [json.load(open(os.path.join(d, "b%03d.json" % b), encoding="utf-8")) for b in range(NB)]


def cost_of_call(c, n_calls=79):
    return (0.05 * c + n_calls - c) / n_calls


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {}
    both = json.load(open(os.path.join(R, "report", "summary_C3_both.json"), encoding="utf-8"))
    sig = C3.sigmas3()
    ev_sig = C3.eval_sigmas(sig)
    for net in C3.NETS:
        s = json.load(open(os.path.join(R, net, "report", "summary_C3.json"), encoding="utf-8"))
        res = s["results"]
        # 1) 判定を結果から計算し直す
        J = C3.judge_all3(res, None, P, net=net)
        hc = P["hc_names"][net]["U_env"] + P["hc_names"][net]["U_min"]
        same = all(J[k] == both["nets"][net][k] for k in hc)
        f0 = res["M0_N040_s0"]["fid"]
        f1 = res["M0_N040_s1"]["fid"]
        tol = 1.03 * f0
        best = J["detail_B22"]["WV_tele_best"]
        o = {"judgement": {k: J[k] for k in hc}, "same_as_notebook": same, "fid_M0": f0, "fid_M0_seed1": f1,
             "seed_spread": abs(f1 - f0), "tol": tol, "grid_best": best,
             "grid_fid_upper": 1.03 * best["fid"], "grid_cost_upper": best["cost"] + 0.10,
             "fid_real_5k": json.load(open(os.path.join(R, "fid_real_5k.json")))["fid_real_5k"],
             "net_info": json.load(open(os.path.join(R, net, "net_info.json")))}
        rules = {}
        for name, tag, key in (("U_env", TAG_E, "detail_B22"), ("U_min", TAG_M, "detail_U_min")):
            d = J[key]
            rr = runs(net, tag)
            calls = [r["switch_call"] for r in rr]
            rules[name] = {
                "fid": d["fid_rule"], "cost": d["cost_rule"], "a": d["a"], "b": d["b"],
                "switch_calls": calls, "switch_sigmas": [r["switch_sigma"] for r in rr],
                "cost_from_calls": float(np.mean([cost_of_call(c) for c in calls])),
                "cost_record": [r["cost_ratio"] for r in rr],
                "seconds": [r["seconds"] for r in rr],
                "margin_fid_to_tol": tol - d["fid_rule"],
                "margin_fid_to_grid": 1.03 * best["fid"] - d["fid_rule"],
                "margin_cost_to_085": 0.85 - d["cost_rule"],
                "margin_cost_to_grid": best["cost"] + 0.10 - d["cost_rule"],
                "fid_minus_M0": d["fid_rule"] - f0,
            }
        o["rules"] = rules
        # 2) 同じ u の列（二つの規則は切り替えまで同じ Gall の軌跡・同じ 5% の点）
        tr_e, tr_m = runs(net, TAG_E), runs(net, TAG_M)
        per = []
        for b in range(NB):
            te, tm = tr_e[b]["trace"], tr_m[b]["trace"]
            n = min(len(te), len(tm))
            prefix_same = all(x[1] == y[1] and x[3] == y[3] for x, y in zip(te[:n], tm[:n]))
            long = te if len(te) >= len(tm) else tm
            u = [r[3] for r in long]
            sg = [r[1] for r in long]
            km = tr_m[b]["switch_call"]
            ke = tr_e[b]["switch_call"]
            # U_min の統計量：u_k / min(u_0..u_{k-1})（k ≥ 1）
            ratio = [None] + [u[k] / min(u[:k]) for k in range(1, len(u))]
            pre = [(k, ratio[k], sg[k]) for k in range(1, km)]  # U_min が切り替える前
            kmax, rmax, smax = max(pre, key=lambda t: t[1])
            fog = [(k, ratio[k], sg[k]) for k in range(1, len(u)) if sg[k] >= 20.0 and k < km]
            kf, rf, sf = max(fog, key=lambda t: t[1])
            # 霧の中の跳ね：隣り合う σ の段の比（同じ σ の 2 回の平均どうし）
            lev = {}
            for k in range(len(u)):
                lev.setdefault(round(sg[k], 9), []).append(u[k])
            levs = sorted(lev.items(), key=lambda t: -t[0])
            neigh = []
            for i in range(1, len(levs)):
                a, bb = np.mean(levs[i - 1][1]), np.mean(levs[i][1])
                if levs[i][0] >= 20:
                    neigh.append((levs[i][0], max(a / bb, bb / a)))
            # U_min の切り替えの後（同じ列の上）で、u が U_min の線（2.38 × 最小値）より下に戻ったか
            mmin = min(u[:km])
            after = u[km:ke + 1] if ke is not None and ke >= km else []
            dips = [u_ / mmin for u_ in after]
            per.append({
                "batch": b, "prefix_same": prefix_same, "len_trace_env": len(te), "len_trace_min": len(tm),
                "switch_call_min": km, "switch_sigma_min": tr_m[b]["switch_sigma"],
                "switch_call_env": ke, "switch_sigma_env": tr_e[b]["switch_sigma"],
                "delay_evals": ke - km, "sigma_ratio_min_over_env": tr_m[b]["switch_sigma"] / tr_e[b]["switch_sigma"],
                "umin_ratio_at_switch": ratio[km],
                "umin_max_ratio_before_switch": rmax, "at_call": kmax, "at_sigma": smax,
                "umin_max_ratio_in_fog_sigma_ge_20": rf, "fog_at_call": kf, "fog_at_sigma": sf,
                "margin_to_238": 2.38 - rmax,
                "max_neighbor_level_ratio_sigma_ge_20": max(t[1] for t in neigh),
                "after_umin_switch_u_over_min": dips, "min_after_over_min": min(dips) if dips else None,
                "u": u, "sigma": sg, "ratio": [None if x is None else float(x) for x in ratio],
                "uenv_state": tr_e[b].get("rule_state"),
            })
        o["per_batch"] = per
        # 実時間
        tm_ = {}
        for tag in sorted(res):
            rr = [json.load(open(os.path.join(R, net, "runs", tag, "b%03d.json" % b))) for b in range(NB)]
            tm_[tag] = {"sec_per_batch_mean": float(np.mean([r["seconds"] for r in rr])),
                        "total_hours": float(np.sum([r["seconds"] for r in rr]) / 3600), "cost": res[tag]["cost_ratio_mean"]}
        o["timing"] = tm_
        out[net] = o
    json.dump(out, open(os.path.join(OUT, "c3_recomputed.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    figures(out)
    return out


def figures(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 図 1：u の列（5 束）と二つの規則の切り替え
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    for ax, net in zip(axes, C3.NETS):
        o = out[net]
        for i, p in enumerate(o["per_batch"]):
            ax.loglog(p["sigma"], p["u"], ".-", color="gray", alpha=0.45, label="u (5 batches, Gall traj.)" if i == 0 else None)
            k = p["switch_call_min"]
            ax.plot(p["sigma"][k], p["u"][k], "rx", ms=9, label="U_min 2.38 switch" if i == 0 else None)
            k = p["switch_call_env"]
            ax.plot(p["sigma"][k], p["u"][k], "b+", ms=11, mew=2, label="U_env switch" if i == 0 else None)
        g = o["grid_best"]
        ax.axvline(g["sigma"], color="C2", lw=2, alpha=0.6, label="best WV_tele (k=%d)" % g["k"])
        ax.invert_xaxis()
        ax.set_xlabel("sigma")
        ax.set_title("AFHQv2 64x64 %s: u = median |D_G - D_net|^2 / d" % net.upper())
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figC3_1_u_and_switches.png"), dpi=130)
    plt.close(fig)
    # 図 2：U_min の統計量 u_k / min(u_0..u_{k-1})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=True)
    for ax, net in zip(axes, C3.NETS):
        for i, p in enumerate(out[net]["per_batch"]):
            k = p["switch_call_min"]
            ax.semilogx(p["sigma"][1:k + 1], p["ratio"][1:k + 1], ".-", alpha=0.6, label="batch %d" % p["batch"])
        ax.axhline(2.38, color="r", ls="--", label="2.38")
        ax.invert_xaxis()
        ax.set_xlabel("sigma")
        ax.set_title("%s: u_k / min(u_0..u_{k-1}) up to the U_min switch" % net.upper())
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figC3_2_umin_ratio.png"), dpi=130)
    plt.close(fig)
    # 図 3：FID と費用（基準、格子、種の揺れ）
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for ax, net in zip(axes, C3.NETS):
        o = out[net]
        s = json.load(open(os.path.join(R, net, "report", "summary_C3.json"), encoding="utf-8"))["results"]
        g = sorted((r["cost_ratio_mean"], r["fid"]) for r in s.values() if r["method"] == "WV_tele")
        ax.plot([a for a, b in g], [b for a, b in g], "s-", color="gray", label="WV_tele(k) run")
        ax.plot(1.0, o["fid_M0"], "ko", ms=8, label="M0 (seed set 0)")
        ax.plot(1.0, o["fid_M0_seed1"], "ko", mfc="none", ms=8, label="M0 (seed set 1)")
        for name, mk in (("U_env", "b*"), ("U_min", "r*")):
            r = o["rules"][name]
            ax.plot(r["cost"], r["fid"], mk, ms=13, label=name)
        ax.axhline(o["tol"], color="gray", ls=":", label="1.03 x FID(M0)")
        ax.axvline(0.85, color="gray", ls="--", lw=0.8)
        ax.axvline(o["grid_cost_upper"], color="C2", ls="--", lw=0.8, label="grid best + 0.10")
        lo = min(o["fid_M0"], o["fid_M0_seed1"], o["rules"]["U_env"]["fid"], o["rules"]["U_min"]["fid"]) - 0.08
        ax.set_ylim(lo, o["tol"] + 0.25)
        ax.set_xlabel("net evaluations / M0")
        ax.set_ylabel("FID (5,000 images)")
        ax.set_title(net.upper())
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figC3_3_fid_cost.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    o = main()
    for net in C3.NETS:
        x = o[net]
        print("==", net, x["judgement"], "一致" if x["same_as_notebook"] else "不一致")
        print("  M0 %.4f  seed1 %.4f  spread %.4f  tol %.4f  grid %s" % (x["fid_M0"], x["fid_M0_seed1"], x["seed_spread"], x["tol"],
              {k: round(v, 4) for k, v in x["grid_best"].items()}))
        for n, r in x["rules"].items():
            print("  %-6s FID %.4f cost %.4f (from calls %.4f) calls %s σ %s  margin FID→tol %.4f  FID−M0 %+.4f  cost→0.85 %.4f  cost→grid %.4f  sec %s" % (
                n, r["fid"], r["cost"], r["cost_from_calls"], r["switch_calls"], [round(v, 2) for v in r["switch_sigmas"]],
                r["margin_fid_to_tol"], r["fid_minus_M0"], r["margin_cost_to_085"], r["margin_cost_to_grid"], [round(v) for v in r["seconds"]]))
        for p in x["per_batch"]:
            print("  b%d prefix_same %s  U_min sw c%d σ%.2f ratio@sw %.2f | max ratio before %.3f at c%d σ%.2f | fog(σ≥20) max %.3f at c%d σ%.2f | U_env sw c%d σ%.2f delay %d evals | neigh %.2f | after/min min %s" % (
                p["batch"], p["prefix_same"], p["switch_call_min"], p["switch_sigma_min"], p["umin_ratio_at_switch"],
                p["umin_max_ratio_before_switch"], p["at_call"], p["at_sigma"], p["umin_max_ratio_in_fog_sigma_ge_20"], p["fog_at_call"], p["fog_at_sigma"],
                p["switch_call_env"], p["switch_sigma_env"], p["delay_evals"], p["max_neighbor_level_ratio_sigma_ge_20"],
                None if p["min_after_over_min"] is None else round(p["min_after_over_min"], 2)))
        print("  timing", {k: (round(v["sec_per_batch_mean"]), round(v["total_hours"], 2)) for k, v in x["timing"].items()})
