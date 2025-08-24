# mollified_second_moment_driver.py (patched with gmpy2 backend)
# Parallel batch + Simpson/Gauss + CSV/JSON export + progress bar.

import argparse
import mpmath as mp
import gmpy2  # Added for faster backend
mp.backend = 'gmpy'  # Set mpmath to use gmpy2 for speed boost
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import json
from riemann_smoothed_afe_helpers import CONFIG, VCache, zeta_smoothed_half_it, mobius_sieve

def build_mollifier(T, theta=0.4):
    y = int(T**theta)
    y = max(y, 2)
    mu = mobius_sieve(y)
    logy = mp.log(y)
    ms, coeffs, logs = [], [], []
    for m in range(1, y+1):
        if mu[m] == 0:
            continue
        u = mp.log(y/m)/logy
        Pm = u  # P(u)=u
        cm = mu[m] * Pm / mp.sqrt(m)
        ms.append(m)
        coeffs.append(cm)
        logs.append(mp.log(m))
    return y, ms, coeffs, logs

def M_value(t, ms, coeffs, logs):
    it = mp.mpc(0, t)
    acc = mp.mpc(0)
    for cm, lm in zip(coeffs, logs):
        acc += cm * mp.exp(-it*lm)
    return acc

def integrand_abs_square(t, use_smoothed_afe, cache_dict, mollifier_data):
    _, ms, coeffs, logs = mollifier_data
    s = mp.mpc('0.5', t)
    if use_smoothed_afe:
        cache_key = float(t)  # dict key must be hashable
        cache = cache_dict.get(cache_key)
        if cache is None:
            X = mp.sqrt(t/(2*mp.pi))
            cache = VCache(s, X, u_max=CONFIG["u_max"], grid_pts=CONFIG["grid_pts"])
            cache_dict[cache_key] = cache
        z = zeta_smoothed_half_it(t, cache)
    else:
        z = mp.zeta(s)
    M = M_value(t, ms, coeffs, logs)
    return abs(z*M)**2

def integrate_simpson(T, theta, panels, use_smoothed_afe=False):
    a, b = mp.mpf(T), mp.mpf(2*T)
    n = int(panels)
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    mollifier_data = build_mollifier(T, theta=theta)
    cache_dict = {}
    ts = [a + mp.mpf(i)*h for i in range(n+1)]
    total = integrand_abs_square(ts[0], use_smoothed_afe, cache_dict, mollifier_data) \
          + integrand_abs_square(ts[-1], use_smoothed_afe, cache_dict, mollifier_data)
    for i in range(1, n):
        w = 4 if i % 2 == 1 else 2
        total += w * integrand_abs_square(ts[i], use_smoothed_afe, cache_dict, mollifier_data)
    return (h/3) * total

def integrate_gauss(T, theta, rel_err=1e-8, use_smoothed_afe=False):
    a, b = mp.mpf(T), mp.mpf(2*T)
    mollifier_data = build_mollifier(T, theta=theta)
    cache_dict = {}
    def f(t):
        return integrand_abs_square(t, use_smoothed_afe, cache_dict, mollifier_data)
    return mp.quad(f, [a, b], error=True, maxdegree=10, method='gauss-legendre', tol=rel_err)[0]

def predicted_main(T, theta):
    return (2/(mp.pi**2)) * theta * T * (mp.log(T)**2)

def run_single(args_tuple):
    T, theta, method, panels, rel_err, use_afe, dps = args_tuple
    mp.mp.dps = dps
    start = time.perf_counter()
    if method == "gauss":
        integral = integrate_gauss(T, theta, rel_err=rel_err, use_smoothed_afe=use_afe)
    else:
        integral = integrate_simpson(T, theta, panels, use_smoothed_afe=use_afe)
    main = predicted_main(T, theta)
    ratio = integral / main
    runtime = time.perf_counter() - start
    return {
        "T": float(T),
        "theta": float(theta),
        "I_numeric": float(integral),
        "predicted_main": float(main),
        "ratio": float(ratio),
        "method": method,
        "dps": dps,
        "use_afe": use_afe,
        "runtime_sec": runtime
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, nargs="+", default=[1000.0], help="one or more T values")
    ap.add_argument("--theta", type=float, nargs="+", default=[0.4], help="one or more theta values")
    ap.add_argument("--panels", type=int, default=96, help="Simpson panels (even)")
    ap.add_argument("--method", choices=["simpson","gauss"], default="simpson", help="integration method")
    ap.add_argument("--rel_err", type=float, default=1e-8, help="relative error for gauss quad")
    ap.add_argument("--dps", type=int, default=60, help="mpmath precision")
    ap.add_argument("--use_afe", action="store_true", help="use smoothed AFE instead of mp.zeta (slower)")
    ap.add_argument("--batch", action="store_true", help="batch over all pairs (T x theta) and print a Markdown table")
    ap.add_argument("--output_format", choices=["csv","json"], default="csv", help="output file format")
    ap.add_argument("--output_path", type=str, default=None, help="output file path (defaults to mollified_results.{csv|json})")
    ap.add_argument("--jobs", type=int, default=cpu_count()-1, help="number of parallel jobs (0 for serial)")
    args = ap.parse_args()

    CONFIG["dps"] = args.dps

    pairs = [(T, th) for T in args.T for th in args.theta]
    tasks = [(T, th, args.method, args.panels, args.rel_err, args.use_afe, args.dps) for T, th in pairs]

    if args.jobs > 0:
        with Pool(args.jobs) as pool:
            results = list(tqdm(pool.imap(run_single, tasks), total=len(tasks), desc="Batch progress"))
    else:
        results = [run_single(task) for task in tqdm(tasks, desc="Batch progress")]

    if args.batch:
        # Print Markdown table
        print("| T | θ | I_numeric | Predicted Main | Ratio I/Main |")
        print("|---:|---:|---:|---:|---:|")
        for res in results:
            print(f"| {res['T']:g} | {res['theta']:g} | {res['I_numeric']:.6g} | {res['predicted_main']:.6g} | {res['ratio']:.3f} |")

    # Dump to file
    if args.output_path is None:
        ext = args.output_format
        args.output_path = f"mollified_results.{ext}"
    if args.output_format == "csv":
        with open(args.output_path, "w") as f:
            f.write("T,theta,I_numeric,predicted_main,ratio,method,dps,use_afe,runtime_sec\n")
            for res in results:
                f.write(f"{res['T']},{res['theta']},{res['I_numeric']},{res['predicted_main']},{res['ratio']},{res['method']},{res['dps']},{res['use_afe']},{res['runtime_sec']}\n")
    else:
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()