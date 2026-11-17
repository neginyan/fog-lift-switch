# fast_gauss.py -- an equivalent implementation of the Gaussian phase with negligible Gaussian cost (MIT License, (c) 2026 Taishi Namba)
#
# Before the switch, c3core.run_signal_rule advances every point of the batch with the Gaussian denoiser D_G, which costs
# two d x d products for the whole batch at every call. But D_G is linear in x: in the eigenbasis of Sigma,
#     y = U^T (x - mu),   D_G - mu = U diag(r) y,   r_j = lambda_j / (lambda_j + sigma^2),
# so a deterministic Heun step with D_G multiplies each coordinate y_j by a number:
#     slope        (x - D_G)/sigma  ->  g_j(sigma) y_j,           g_j(sigma) = sigma / (lambda_j + sigma^2)
#     predictor    F_p = F_i (1 + h g_j(t_c))                       (h = t_n - t_c)
#     corrector    F_{i+1} = F_i + h (0.5 g_j(t_c) F_i + 0.5 g_j(t_n) F_p)
# with y(step i) = F_i * y0 and y0 = U^T (x_init - mu). These are the same Heun steps as c1core.heun_sample with D_G,
# multiplied out per eigen-direction (not the analytical teleport, which would differ by the Heun discretization error).
# So, in the fog, only the 5% subset is needed at each call (computed from y0 of that subset), and the whole batch is
# reconstructed once, at the switch. Results agree with c3core.run_signal_rule up to floating-point rounding; the
# subsets, the switching evaluation and the network evaluations are the same. c1core-c3core are not changed.
import time

import numpy as np

import c1core as C1
import c2core as C2
import c3core as C3


def _ones_like(v):
    return v * 0 + 1


def run_signal_rule_fast(rule, x_init, sigmas, net_fn, G, rng, stat_fn=C3.stat_u, frac=None, time_thin=1, sync=None):
    """Same interface and same result as c3core.run_signal_rule(pre_switch='DG'), with the Gaussian phase multiplied out."""
    assert int(time_thin) == 1, "fast_gauss implements the T1 schedule used in the paper (every call evaluated)"
    frac = C3.PREREG3["subset_frac"] if frac is None else frac
    n = int(x_init.shape[0])
    net = C1.CountingNet(net_fn)
    N = len(sigmas) - 1
    m = max(1, int(round(frac * n)))
    lam, U, mu = G.lam, G.U, G.mu
    trace = []
    st = {"switched": False, "switch_sigma": None, "switch_call": None, "full_products": 0}
    if sync:
        sync()
    t0 = time.time()

    def g(s):
        return float(s) / (lam + float(s) ** 2)

    def r(s):
        return lam / (lam + float(s) ** 2)

    def to_x(Fv, y):
        return mu + (Fv * y) @ U.T

    def look(Fs, s):
        """One pre-switch evaluation at the state Fs * y0, on a fresh subset. Returns (switch?, subset info)."""
        sub = np.sort(rng.choice(n, size=m, replace=False))
        isub = C1._idx(sub, x_init)
        y0s = (x_init[isub] - mu) @ U
        xs = to_x(Fs, y0s)
        DGs = mu + (Fs * r(s) * y0s) @ U.T
        Dn_sub = net(xs, s)
        val = float(stat_fn(DGs, Dn_sub, xs, float(s)))
        u = float(np.median(C2.u_points(DGs, Dn_sub)))
        sw = bool(rule.observe(float(s), val))
        return sw, sub, isub, Dn_sub, val, u

    def net_all(x, s, sub=None, isub=None, Dn_sub=None):
        """Network on the whole batch, reusing the subset already evaluated at this call (as c3core does)."""
        if sub is None:
            return net(x, s)
        out = C1._clone(x)
        out[isub] = Dn_sub
        rest = np.setdiff1d(np.arange(n), sub)
        if len(rest) > 0:
            irest = C1._idx(rest, x)
            out[irest] = net(x[irest], s)
        return out

    F = _ones_like(lam)
    x = None
    c = 0
    for i in range(N):
        tc, tn = float(sigmas[i]), float(sigmas[i + 1])
        h = tn - tc
        if not st["switched"]:
            # predictor call
            sw, sub, isub, Dn_sub, val, u = look(F, tc)
            trace.append([int(c), tc, val, u])
            if sw:
                y0 = (x_init - mu) @ U
                xi = to_x(F, y0)
                st["full_products"] += 2
                st.update(switched=True, switch_sigma=tc, switch_call=int(c))
                D = net_all(xi, tc, sub, isub, Dn_sub)
                c += 1
                d = (xi - D) / tc
                xn = xi + h * d
                if i < N - 1:
                    D2 = net(xn, tn)
                    c += 1
                    xn = xi + h * (0.5 * d + 0.5 * (xn - D2) / tn)
                x = xn
                continue
            c += 1
            Fp = F * (1 + h * g(tc))
            if i == N - 1:
                F = Fp
                continue
            # corrector call
            sw, sub, isub, Dn_sub, val, u = look(Fp, tn)
            trace.append([int(c), tn, val, u])
            if sw:
                y0 = (x_init - mu) @ U
                xi = to_x(F, y0)
                d = (g(tc) * F * y0) @ U.T
                xn = to_x(Fp, y0)
                st["full_products"] += 4
                st.update(switched=True, switch_sigma=tn, switch_call=int(c))
                D2 = net_all(xn, tn, sub, isub, Dn_sub)
                c += 1
                x = xi + h * (0.5 * d + 0.5 * (xn - D2) / tn)
                continue
            c += 1
            F = F + h * (0.5 * g(tc) * F + 0.5 * g(tn) * Fp)
        else:
            d = (x - net(x, tc)) / tc
            c += 1
            xn = x + h * d
            if i < N - 1:
                d2 = (xn - net(xn, tn)) / tn
                c += 1
                xn = x + h * (0.5 * d + 0.5 * d2)
            x = xn
    if not st["switched"]:
        x = to_x(F, (x_init - mu) @ U)
        st["full_products"] += 2
    if sync:
        sync()
    info = {"n": n, "steps": N, "subset_m": m, "time_thin": 1, "trace": trace,
            "trace_columns": ["call", "sigma", "stat", "u"], "switched": st["switched"],
            "switch_sigma": st["switch_sigma"] if st["switched"] else 0.0, "switch_call": st["switch_call"],
            "seconds": time.time() - t0, "net_points": net.points, "gauss_full_batch_products": st["full_products"],
            "implementation": "fast_gauss"}
    if hasattr(rule, "state"):
        info["rule_state"] = rule.state()
    info["cost_units"] = net.points / n
    info["cost_ratio"] = info["cost_units"] / C1.nfe_m0(N)
    return x, info


class SwitchReached(Exception):
    """Raised by StopAtSwitch when the wrapped rule switches (used only to time the Gaussian phase)."""


class StopAtSwitch:
    """Wrap a rule; raise SwitchReached at its switch, so a run can be stopped there and timed.

    The pre-switch phase is the only part in which the two implementations differ; after the switch both run the
    network on the whole batch in the same way."""

    def __init__(self, rule):
        self.rule = rule
        self.trace = []

    @property
    def switched(self):
        return self.rule.switched

    def observe(self, sigma, stat):
        sw = self.rule.observe(sigma, stat)
        self.trace.append((float(sigma), float(stat)))
        if sw:
            raise SwitchReached()
        return False


def gaussian_phase(run_fn, rule, *args, sync=None, **kw):
    """Run run_fn (c3core.run_signal_rule or run_signal_rule_fast) up to the switch of `rule`; return
    (seconds, switching evaluation index, list of (sigma, stat))."""
    import time as _t
    w = StopAtSwitch(rule)
    if sync:
        sync()
    t0 = _t.time()
    try:
        run_fn(w, *args, **kw)
    except SwitchReached:
        pass
    if sync:
        sync()
    return _t.time() - t0, len(w.trace) - 1, w.trace
