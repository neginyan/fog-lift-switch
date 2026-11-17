# uenv_window.py -- the U_env window on the two EDM schedules, and the switching evaluations on AFHQv2 (MIT License, (c) 2026 Taishi Namba)
#
# Part A (schedule only): for every evaluation of the 18-step (CIFAR-10) and 40-step (AFHQv2) schedules, the number of
#   evaluations in the U_env window, the span of the window in ln(sigma), and the factor by which the current value must
#   exceed the baseline when u rises smoothly as sigma^-2 (C * exp(2 * span)). Uses c3core.uenv_window_sizes unchanged.
# Part B (records): at the switching evaluations of both rules on all 30 AFHQv2 batches, the U_env baseline P_k, the
#   window minimum, the U_min running minimum and the current value (reviewer point G2).
# Usage: python analysis/uenv_window.py   (writes analysis/out/uenv_window.json)
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import c1core as C1  # noqa: E402
import c3core as C3  # noqa: E402

K = C3.PREREG3["b22_constants"]
REC = os.path.join(ROOT, "records")
OUT = os.path.join(HERE, "out")


def window_table(steps):
    ev = C3.eval_sigmas(C1.edm_sigmas(steps))
    rows = C3.uenv_window_sizes(ev, K["DELTA"], K["M_MIN"], K["window_tol"])
    for r in rows:
        span = math.log(ev[r["a"]]) - math.log(r["sigma"])
        r["span_ln_sigma"] = span
        r["smooth_rise_factor"] = K["C"] * math.exp(2 * span)  # u_k / P_k needed if u ~ sigma^-2 inside the window
    return rows


def summarize(rows):
    dec = [r for r in rows if r["k"] >= 2]  # the first two evaluations cannot have three points
    counts = {}
    for r in dec:
        counts[r["window_n"]] = counts.get(r["window_n"], 0) + 1
    return {"n_eval": len(rows), "counts_window_n_from_k2": dict(sorted(counts.items())),
            "first_step_dln_sigma": math.log(rows[0]["sigma"]) - math.log(rows[1]["sigma"])}


def _runs(net, rule, s):
    base = os.path.join(REC, "C3_results" if s == 0 else "C6_results", net, "runs")
    tag = ("U_env_N040_s%d" if rule == "U_env" else "U_min_c2.38_N040_s%d") % s
    d = os.path.join(base, tag)
    return [json.load(open(os.path.join(d, "b%03d.json" % b), encoding="utf-8")) for b in range(5)]


def switch_states():
    """For every AFHQv2 batch: the quantities of both rules at both switching evaluations (on the shared trace)."""
    out = []
    for net in ("vp", "ve"):
        for s in (0, 2, 3):
            for je, jm in zip(_runs(net, "U_env", s), _runs(net, "U_min", s)):
                long = je if len(je["trace"]) >= len(jm["trace"]) else jm
                sg = [t[1] for t in long["trace"]]
                u = [t[3] for t in long["trace"]]
                rule = C3.UEnvRule(K["DELTA"], K["C"], K["M_MIN"], K["window_tol"])
                st = []
                for k, (a, b) in enumerate(zip(sg, u)):
                    rule.switched = False  # keep observing past the switch to read the state at every evaluation
                    rule.observe(a, b)
                    L = dict(rule.last)
                    L["running_min_before"] = min(u[:k]) if k else None
                    L["u"] = b
                    L["sigma"] = a
                    st.append(L)
                ke = len(je["trace"]) - 1
                km = len(jm["trace"]) - 1

                def pick(k):
                    x = st[k]
                    return {"k": k, "sigma": x["sigma"], "window_n": x["window_n"], "u": x["u"], "P_env": x["P"],
                            "min_window": x["min_w"], "running_min": x["running_min_before"],
                            "env_ratio": x["min_w"] / x["P"], "umin_ratio": x["u"] / x["running_min_before"],
                            "P_over_running_min": x["P"] / x["running_min_before"]}
                out.append({"net": net, "seedset": s, "batch": je["batch"], "k_env": ke, "k_min": km,
                            "delay_evals": ke - km, "at_env": pick(ke), "at_min": pick(km)})
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {"constants": {k: K[k] for k in ("DELTA", "C", "M_MIN")}}
    for steps in (18, 40):
        rows = window_table(steps)
        res["steps_%d" % steps] = {"summary": summarize(rows), "rows": rows}
    res["switch_states"] = switch_states()
    json.dump(res, open(os.path.join(OUT, "uenv_window.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    r = main()
    for steps in (18, 40):
        x = r["steps_%d" % steps]
        print(steps, x["summary"])
        for row in x["rows"][:40]:
            print("  k=%2d sigma=%7.3f window=%d span=%.3f factor=%.2f" % (row["k"], row["sigma"], row["window_n"],
                                                                         row["span_ln_sigma"], row["smooth_rise_factor"]))
    for z in r["switch_states"]:
        if z["seedset"] == 0 or True:
            e, m = z["at_env"], z["at_min"]
            print("%s s%d b%d  env k%d(n%d) minW/P=%.2f  u/runmin=%.2f P/runmin=%.2f | min k%d  minW/P=%.2f u/runmin=%.2f" % (
                z["net"], z["seedset"], z["batch"], z["k_env"], e["window_n"], e["env_ratio"], e["umin_ratio"], e["P_over_running_min"],
                z["k_min"], m["env_ratio"], m["umin_ratio"]))
