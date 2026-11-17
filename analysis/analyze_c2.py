# analyze_c2.py -- C2 の結果を記録から計算し直し、報告用の数字と図を作る（MIT）
# 使い方：python analyze_c2.py   （records/C2_results と records/C1_results を読む）
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import c1core as C1  # noqa: E402
import c2core as C2  # noqa: E402

C2R = os.environ.get("C2_RESULTS", os.path.join(ROOT, "records", "C2_results"))
C1R = os.environ.get("C1_RESULTS", os.path.join(ROOT, "records", "C1_results"))
OUT = os.path.join(HERE, "out", "c2")
SPIKE_SIGMA = 40.78557379650796  # 18 歩の目盛りの σ_2


def load_runs(net, tag):
    d = os.path.join(C2R, net, "runs", tag)
    return [json.load(open(os.path.join(d, "b%03d.json" % b), encoding="utf-8")) for b in range(10)]


def main():
    os.makedirs(OUT, exist_ok=True)
    s = json.load(open(os.path.join(C2R, "report", "summary_C2.json"), encoding="utf-8"))
    sr = tuple(s["extra"]["srange_ve"]) if s["extra"].get("srange_ve") else None
    # 1) 判定を結果から計算し直す
    J = C2.judge_all(s["results_VP"], s["results_VE"], srange_ve=sr)
    same = {k: (J[k] == s["judgement"][k]) for k in ("HC5", "HC6", "HC7", "HC8")}
    out = {"judgement": {k: J[k] for k in ("HC5", "HC6", "HC7", "HC8")}, "same_as_summary": same}
    for key in ("detail_VE_main", "detail_VP_main", "obs_VE_sub", "obs_VP_sub", "obs_VE_S_gap", "obs_VP_S_gap_C1"):
        d = J[key]
        out[key] = {x: d.get(x) for x in ("fid_M0", "fid_tol", "fid_rule", "cost_rule", "switch_sigmas",
                                          "WV_tele_best", "a", "b", "fid_M0_other_seeds", "fid_seed_spread",
                                          "within_seed_spread")}
    # 2) 束ごと：最小値を作った評価と、切り替えた評価
    per = {}
    for net in ("vp", "ve"):
        for c in ("2.38", "7.35"):
            rows = []
            for i in load_runs(net, "U_min_c%s_N018_s0" % c):
                tr = i["trace"]
                us = [r[2] for r in tr]
                kmin = int(np.argmin(us[:-1])) if i["switched"] else int(np.argmin(us))
                c_sw = i["switch_call"]
                rows.append({"batch": i["batch"], "switch_call": c_sw, "switch_sigma": i["switch_sigma"],
                             "cost": i["cost_ratio"], "cost_from_call": (0.05 * c_sw + 35 - c_sw) / 35,
                             "min_call": tr[kmin][0], "min_sigma": tr[kmin][1], "min_u": us[kmin],
                             "switch_u": us[-1], "ratio_switch_over_min": us[-1] / us[kmin],
                             "u": us, "sigma": [r[1] for r in tr]})
            per["%s_%s" % (net, c)] = rows
    out["per_batch"] = per
    # 跳ね：VE の σ = 40.79 の 2 回の評価（呼び出し 3・4）と、その前の最小値
    spike = []
    for i in load_runs("ve", "U_min_c7.35_N018_s0"):
        tr = i["trace"]
        pre = min(r[2] for r in tr[:3])
        sp = [r[2] for r in tr if abs(r[1] - SPIKE_SIGMA) < 1e-9]
        after = [r[2] for r in tr if 19 < r[1] < 30]
        spike.append({"batch": i["batch"], "min_calls_0_2": pre, "u_at_40.79": sp, "ratio": [v / pre for v in sp],
                      "u_sigma_19_28": after, "floor_over_pre": min(after) / pre})
    out["ve_spike"] = spike
    # 2.38 と 7.35 の軌跡は、切り替えまで同じ（どちらも Gall の軌跡、同じ乱数）
    pref = all(all(a[2] == b[2] for a, b in zip(x["trace"], y["trace"]))
               for net in ("vp", "ve")
               for x, y in zip(load_runs(net, "U_min_c2.38_N018_s0"), load_runs(net, "U_min_c7.35_N018_s0")))
    out["prefix_2.38_equals_7.35"] = pref
    # 50 点の中央値の揺れの大きさ：同じ σ での 2 回の評価（Heun の補正と次の歩）の比
    noise = {}
    for net in ("vp", "ve"):
        rs, rsp = [], []
        for i in load_runs(net, "U_min_c7.35_N018_s0"):
            by = {}
            for r in i["trace"]:
                by.setdefault(round(r[1], 9), []).append(r[2])
            for sgm, v in by.items():
                if len(v) == 2:
                    (rsp if abs(sgm - SPIKE_SIGMA) < 1e-6 else rs).append(max(v) / min(v))
        noise[net] = {"n_pairs": len(rs), "median": float(np.median(rs)), "q95": float(np.quantile(rs, 0.95)),
                      "max": float(np.max(rs)), "spike_pairs": rsp}
    out["same_sigma_pair_ratio"] = noise
    # 3) パイロット（50 歩、256 点）：隣り合う σ の比
    pil = {}
    for net, p in (("VE", os.path.join(C2R, "ve", "pilot.json")), ("VP", os.path.join(C1R, "pilot.json"))):
        rows = json.load(open(p, encoding="utf-8"))["rows"][::2]
        sg = np.array([r["sigma"] for r in rows])
        u = np.array([r["dD2_per_dim_median"] for r in rows])
        rat = np.maximum(u[1:] / u[:-1], u[:-1] / u[1:])
        fog = sg[1:] >= 10
        pil[net] = {"sigma": sg.tolist(), "u": u.tolist(), "max_neighbor_ratio_sigma_ge_10": float(rat[fog].max()),
                    "argmin_sigma_ge_10": float(sg[sg >= 10][np.argmin(u[sg >= 10])]),
                    "decision": json.load(open(p, encoding="utf-8"))["decision"]}
    out["pilot"] = pil
    # 4) 事後の観察：跳ねの段（σ = 40.79）の評価を除いて、規則を当て直す（判定には使わない）
    post = []
    for i in load_runs("ve", "U_min_c7.35_N018_s0"):
        tr = i["trace"]
        keep = [r for r in tr if abs(r[1] - SPIKE_SIGMA) > 1e-9]
        k = C2.replay_umin([r[2] for r in keep], 2.38)
        c = keep[k][0] if k is not None else None
        post.append({"batch": i["batch"], "switch_call": c, "switch_sigma": keep[k][1] if k is not None else None,
                     "cost_if_evaluated": (0.05 * c + 35 - c) / 35 if c is not None else None,
                     "cost_if_spike_step_skipped": (0.05 * (c - 2) + 35 - c) / 35 if c is not None else None})
    out["posthoc_ve_2.38_without_spike"] = post
    json.dump(out, open(os.path.join(OUT, "c2_recomputed.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    figures(out)
    return out


def figures(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    for net, col in (("VP", "C0"), ("VE", "C3")):
        p = out["pilot"][net]
        ax.loglog(p["sigma"], p["u"], "o-", ms=3, color=col, label="%s (pilot, 50 steps, 256 pts)" % net)
    ax.axvline(10, color="gray", ls=":")
    ax.invert_xaxis()
    ax.set_xlabel("sigma")
    ax.set_ylabel("u = median |D_G - D_net|^2 / d")
    ax.set_title("C2: pilot curves along M0 trajectories")
    ax.legend(fontsize=8)
    ax = axes[1]
    for net, col in (("vp", "C0"), ("ve", "C3")):
        for j, r in enumerate(out["per_batch"]["%s_7.35" % net]):
            ax.loglog(r["sigma"], r["u"], ".-", color=col, alpha=0.35, label=("%s, 10 batches (Gall traj.)" % net.upper()) if j == 0 else None)
        for j, r in enumerate(out["per_batch"]["%s_2.38" % net]):
            ax.plot([r["switch_sigma"]], [r["switch_u"]], "x", color=col, ms=9,
                    label=("%s: switch of U_min 2.38" % net.upper()) if j == 0 else None)
    ax.axvline(SPIKE_SIGMA, color="C3", ls="--", alpha=0.5, label="sigma = 40.79 (18-step grid)")
    ax.axvline(8.40, color="C2", lw=2, alpha=0.6, label="best WV_tele (k=6, both nets)")
    ax.invert_xaxis()
    ax.set_xlabel("sigma")
    ax.set_title("C2: u on 18-step runs (5% subset median)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figC2_u_VP_vs_VE.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    o = main()
    print(json.dumps(o["judgement"], ensure_ascii=False), o["same_as_summary"])
    print("prefix 2.38 == 7.35:", o["prefix_2.38_equals_7.35"])
    print("pilot max neighbor ratio σ≥10:", {k: round(v["max_neighbor_ratio_sigma_ge_10"], 2) for k, v in o["pilot"].items()})
    for r in o["ve_spike"]:
        print("VE b%d  min(0-2) %.2e  u@40.79 %s  ratio %s  floor/pre %.2f" % (
            r["batch"], r["min_calls_0_2"], ["%.2e" % v for v in r["u_at_40.79"]], ["%.1f" % v for v in r["ratio"]], r["floor_over_pre"]))
    print("post-hoc:", [(p["switch_sigma"] and round(p["switch_sigma"], 2), round(p["cost_if_evaluated"], 4)) for p in o["posthoc_ve_2.38_without_spike"]])
    print("post-hoc mean cost", np.mean([p["cost_if_evaluated"] for p in o["posthoc_ve_2.38_without_spike"]]))
