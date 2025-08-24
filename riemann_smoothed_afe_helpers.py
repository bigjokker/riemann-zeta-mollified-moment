
# riemann_smoothed_afe_helpers.py (updated)
# Helpers for smoothed AFE of zeta with G(w)=exp(w^2), and estimation of C_V, C_P
# Adds: estimate_CV_mode(.., mode="abs2"|"square") and a small CLI.

import argparse
import mpmath as mp

CONFIG = {
    "dps": 40,
    "C": 1.5,
    "VMAX": 12.0,
    "ABS_CUTOFF": mp.mpf("1e-30"),
    "MAX_N": 120000,
    "grid_pts": 160,
    "u_min": 1e-6,
    "u_max": 12.0
}

def chi(s):
    return mp.pi**(s-mp.mpf('0.5')) * mp.gamma((1-s)/2) / mp.gamma(s/2)

def G(w):
    return mp.e**(w*w)

def V_weight_raw(u, s, C=None, VMAX=None):
    if u <= 0:
        return mp.mpc(0)
    if C is None:
        C = CONFIG["C"]
    if VMAX is None:
        VMAX = CONFIG["VMAX"]
    def integrand(v):
        w = mp.mpc(C, v)
        term = (G(w)/w) * (mp.gamma((s+w)/2) / mp.gamma(s/2)) * (mp.pi*u)**(-w/2)
        return term * 1j/(2*mp.pi)
    return mp.quad(integrand, [-VMAX, VMAX])

class VCache:
    def __init__(self, s, X, u_min=None, u_max=None, grid_pts=None):
        self.s = s
        self.X = X
        self.u_min = u_min if u_min is not None else CONFIG["u_min"]
        self.u_max = u_max if u_max is not None else CONFIG["u_max"]
        self.grid_pts = grid_pts if grid_pts is not None else CONFIG["grid_pts"]
        self.us = [self.u_min * (self.u_max/self.u_min)**(i/(self.grid_pts-1)) for i in range(self.grid_pts)]
        self.log_us = [mp.log(u) for u in self.us]
        self.Vs = [V_weight_raw(u, s) for u in self.us]
        self.V1s = [V_weight_raw(u, 1-s) for u in self.us]

    def _interp_complex(self, logu, grid_logs, grid_vals):
        if logu <= grid_logs[0]:
            return grid_vals[0]
        if logu >= grid_logs[-1]:
            return grid_vals[-1]
        lo, hi = 0, len(grid_logs)-1
        while hi - lo > 1:
            mid = (lo + hi)//2
            if logu < grid_logs[mid]:
                hi = mid
            else:
                lo = mid
        x0, x1 = grid_logs[lo], grid_logs[hi]
        y0, y1 = grid_vals[lo], grid_vals[hi]
        t = (logu - x0) / (x1 - x0)
        return y0 + t*(y1 - y0)

    def V_s(self, u):
        if u <= 0 or u >= self.u_max:
            return mp.mpc(0)
        if u <= self.u_min:
            return self.Vs[0]
        return self._interp_complex(mp.log(u), self.log_us, self.Vs)

    def V_1minus_s(self, u):
        if u <= 0 or u >= self.u_max:
            return mp.mpc(0)
        if u <= self.u_min:
            return self.V1s[0]
        return self._interp_complex(mp.log(u), self.log_us, self.V1s)

def zeta_smoothed_half_it(t, cache: VCache):
    s = cache.s
    X = cache.X
    chi_s = chi(s)

    acc1 = mp.mpc(0)
    for n in range(1, CONFIG["MAX_N"]+1):
        v = cache.V_s(n/X)
        if abs(v) < CONFIG["ABS_CUTOFF"]:
            break
        acc1 += v / (mp.mpf(n)**s)

    acc2 = mp.mpc(0)
    for n in range(1, CONFIG["MAX_N"]+1):
        v = cache.V_1minus_s(n/X)
        if abs(v) < CONFIG["ABS_CUTOFF"]:
            break
        acc2 += v / (mp.mpf(n)**(1-s))

    return acc1 + chi_s * acc2

def mobius_sieve(n):
    mu = [1]*(n+1)
    prime = [True]*(n+1)
    prime[0]=prime[1]=False
    primes = []
    for i in range(2, n+1):
        if prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i*p > n:
                break
            prime[i*p] = False
            if i % p == 0:
                mu[i*p] = 0
                break
            else:
                mu[i*p] = -mu[i]
    return mu

def estimate_CP(y, P="u"):
    y_int = int(y)
    if y_int < 2:
        raise ValueError("y must be >= 2")
    mu = mobius_sieve(y_int)
    logy = mp.log(y_int)
    S = mp.mpf('0')
    for m in range(1, y_int+1):
        if mu[m] == 0:
            continue
        u = mp.log(y_int/m) / logy
        Pm = u if P=="u" else u
        S += (Pm**2) / m
    main = (6/(mp.pi**2)) * logy * (mp.mpf('1')/3)  # ∫_0^1 u^2 du = 1/3
    CP_est = S - main
    return CP_est, S, main

def estimate_CV_mode(t, u_max_for_sum=12.0, r_cap=None, mode="abs2"):
    s = mp.mpc('0.5', t)
    X = mp.sqrt(t/(2*mp.pi))
    cache = VCache(s, X, u_max=u_max_for_sum, grid_pts=CONFIG["grid_pts"])
    R = max(10, int(mp.ceil(u_max_for_sum * X)))
    if r_cap is not None:
        R = min(R, r_cap)
    S = mp.mpf('0')
    for r in range(1, R+1):
        v = cache.V_s(r/X)
        if mode == "square":
            S += (v*v).real / r
        else:
            S += (v*v.conjugate()).real / r
    CV_est = S - mp.log(X)
    return CV_est, S, X, cache, R

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Estimate C_V or C_P; quick zeta AFE check")
    ap.add_argument("--t", type=float, default=1000.0, help="height for C_V")
    ap.add_argument("--cv_mode", choices=["abs2","square"], default="abs2", help="use |V|^2 (abs2) or V^2 (square)")
    ap.add_argument("--u_max", type=float, default=12.0, help="u_max for r-sum via r<=u_max*X")
    ap.add_argument("--r_cap", type=int, default=400, help="cap R for speed")
    ap.add_argument("--y", type=int, default=200000, help="y for C_P estimation")
    ap.add_argument("--dps", type=int, default=50, help="mpmath precision")
    args = ap.parse_args()

    CONFIG["dps"] = args.dps
    mp.mp.dps = args.dps

    CV_est, *_ = estimate_CV_mode(args.t, u_max_for_sum=args.u_max, r_cap=args.r_cap, mode=args.cv_mode)
    CP_est, *_ = estimate_CP(args.y)

    print(f"C_V({args.cv_mode}) ≈ {CV_est}")
    print(f"C_P ≈ {CP_est}")
