# CST-305 Benchmark Project 6 – README

## Install Dependencies

```bash
pip install numpy matplotlib scipy
```

## Run the Program

```bash
python benchmark6.py
```

Or in PyCharm, right-click `benchmark6.py` and select **Run 'benchmark6'**.

## What It Does

The program runs all three parts automatically in sequence:

- **Part 1a** – Solves `y'' − 2x·y' + x²·y = 0` via Taylor polynomial (n≤4), prints result at x=3.5
- **Part 1b** – Solves `y'' − (x−2)·y' + 2y = 0` via 2nd-order Taylor about x=3
- **Part 2**  – Power series solution of `(x²+4)·y'' + y = x` at x=0
- **Part 3**  – Simulates computer system performance using an ODE model

## Output

Console results are printed for each part. The following figures are saved automatically:

- `part1a_taylor.png`
- `part1b_taylor.png`
- `part2_power_series.png`
- `part3_performance.png`
