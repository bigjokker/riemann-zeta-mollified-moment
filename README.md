# Mollified Second Moment Driver

This Python script computes the mollified second moment of the Riemann zeta function over intervals [T, 2T], useful for analytic number theory research, particularly in studying the distribution of zeta zeros related to the Riemann Hypothesis. It supports parallel batch processing, high-precision arithmetic via mpmath (with optional gmpy2 speedup), and two integration methods: Simpson's rule or Gauss-Legendre quadrature.

## Features
- Batch computations over multiple T and θ values.
- Optional use of smoothed Approximate Functional Equation (AFE) for zeta evaluations.
- Parallel execution using multiprocessing.
- Output in CSV or JSON format, including computed integral, predicted asymptotic main term, ratio, and runtime.
- Progress bar for long runs (via tqdm).

## Requirements
- Python 3.6+
- Libraries: mpmath, tqdm, gmpy2 (optional for speedup), and the companion `riemann_smoothed_afe_helpers.py` script (included in this repo).

Install dependencies:
pip install mpmath tqdm gmpy2

Usage

Run the script with command-line arguments. Basic example:
```bash
python mollified_second_moment_driver.py --T 1000 --theta 0.4 --method simpson --panels 96 --dps 60
```

For batch mode with parallel jobs:
```bash
python mollified_second_moment_driver.py \
  --batch \
  --T 1000 5000 10000 \
  --theta 0.4 0.45 \
  --method gauss --rel_err 1e-8 \
  --use_afe \
  --dps 80 \
  --jobs 4 \
  --output_format csv \
  --output_path results.csv
```
 This will compute for all combinations of T and θ, save to CSV, and print a Markdown table summary.

Arguments:
```text
--T: List of starting heights (e.g., 1000 5000). Default: [1000.0]

--theta: Mollifier exponents (θ < 0.5). Default: [0.4]

--panels: Number of Simpson panels (even number). Default: 96

--method: Integration method ("simpson" or "gauss"). Default: "simpson"

--rel_err: Relative error tolerance for Gauss. Default: 1e-8

--dps: mpmath precision digits. Default: 60

--use_afe: Use smoothed AFE instead of mp.zeta (slower but customizable).

--batch: Enable batch mode with table output.

--output_format: "csv" or "json". Default: "csv"

--output_path: File path for results. Default: mollified_results.{csv|json}

--jobs: Number of parallel processes (0 for serial). Default: cpu_count()-1
```
## How It Works:

The script integrates |ζ(1/2 + it) M(1/2 + it)|^2 from T to 2T, where M is a mollifier polynomial. It compares the result to the predicted asymptotic main term (2/π² θ T (log T)²) and computes the ratio (should approach 1 for large T under RH assumptions).

For high precision or large T, expect longer runtimes—use --jobs for speedup on multi-core systems.


## Credits:

- Built with insights from analytic number theory resources (e.g., Titchmarsh, Ivić).

- Thanks to xAI for Grok assistance in development.

- Thanks to OpenAI for ChatGPT assistance in development.

If you find bugs or have suggestions, open an issue!
