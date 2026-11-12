# OpenFormula

**A physics-informed symbolic regression tool: start from a known physics
relationship and let evolutionary search discover what it's missing.**

## The idea in one sentence

You give OpenFormula a dataset and (optionally) a known physics formula; it
runs an evolutionary search — Gene Expression Programming (GEP) — to find
symbolic math terms that explain the *leftover* part of your data the
formula doesn't already capture, and tells you which terms and variable
interactions show up again and again across many independent runs.

If you don't have a known formula for your problem, it will just search
directly for terms that explain the target itself.

```
Fitness:  error( target, physics(x) + GEP(x) )
```

GEP is only rewarded for the part the known physics doesn't already
explain (or, in multiplicative mode, the fractional correction the physics
baseline is still missing).

## Why this exists

A lot of applied science and engineering already has decades of
closed-form equations — design codes, simplified physical models,
empirical laws — that get you most of the way to predicting some outcome,
but not all the way. The usual next step is either:

- **Throw the whole thing at a black-box ML model** (random forest, neural
  net, etc.), which can fit the data well but gives you no insight into
  *what* it learned or *why* — and quietly discards the physics you
  already know is true.
- **Manually guess-and-check additional correction terms**, which is slow
  and easy to bias toward whatever you expected to find.

OpenFormula sits in between: it keeps your known physics as a fixed
baseline, and only asks the evolutionary search to explain the residual.
Because GEP evolves actual symbolic expressions (not black-box weights),
every result it reports is a human-readable term like `d**2/a` or
`1/sqrt(fc)`, not an opaque number.

## Why trust what it finds

A single evolutionary run can "discover" a term just by random luck. To
guard against that, OpenFormula never relies on one run: it repeats the
search across many independent random seeds, *and* across several
different training-data sizes (e.g. "what if you'd only had 5% of this
data" vs. "what if you'd had 75% of it"), then reports the terms and
interactions that keep showing up across that whole sweep, not just the
single best-scoring run. A term that survives dozens of random restarts
and multiple training-data regimes is much more likely to be a genuine
signal than a one-off fluke.

## Quick start (app)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

or just run `./run_app.sh` (Linux/macOS) / double-click `run_app.bat`
(Windows) after installing the requirements once.

Then in the browser:

1. **Dataset** — upload your own `.csv`/`.xlsx` file, or pick one of the
   built-in demo datasets to try it out immediately (see below).
2. **Response & feature variables** — pick the target column and the
   candidate feature columns. Non-numeric columns are flagged and you'll
   be prompted to one-hot encode or drop them.
3. **Physical knowledge (optional)** — pick a built-in known formula from
   the dropdown (each shown with its equation and purpose inline), or type
   your own, referencing the feature columns directly, e.g.
   `0.17*sqrt(fc)*bw*d/1000`. Allowed operators: `+ - * / **`, and
   functions `sqrt log log10 log2 exp abs min max sin cos tan power pi`.
   Choose additive (`physics + GEP(x)`) or multiplicative
   (`physics * (1 + GEP(x))`) combination mode. Leave the formula blank to
   have GEP fit the target directly.
4. **Search settings** — equation complexity (head length / number of
   genes), generations, population size, tournament size, Hall-of-Fame
   size, the training-data-availability sweep (e.g. `0.05, 0.1, 0.25, 0.5`),
   and how many random seeds to run per level.
5. **Run search** — progress and a live ETA are shown as it works through
   every seed × training-level combination.
6. **Results** — ranked term/interaction tables, predicted-vs-actual plots
   for the top recurring terms (compared against the physics-only
   baseline), equation-building tools (OLS/Ridge regression on the
   discovered terms, or hand-picking your own), and a downloadable HTML
   report documenting the dataset, formulas, fit statistics, plots, and
   every hyperparameter actually used in that run.

Note: column names that aren't valid Python identifiers (e.g. `a/d`,
`f'c`) are automatically mapped to safe names — use the mapping shown in
the app when writing your own physics formula.

## Command-line pipeline (no UI)

Useful for large/unattended sweeps.

```bash
python run_sweep.py --data mydata.csv --target y \
    --physics "0.17*sqrt(fc)*bw*d/1000" \
    --seeds 0-19 --ptrains 0.05,0.1,0.25,0.5 \
    --generations 750 --pop-size 100 --head-length 2 --n-genes 2 \
    --out results/sweep_results.csv

python parse_terms.py --in results/sweep_results.csv \
    --out-dir results/term_report --by-ptrain
```

`run_sweep.py --help` lists every knob (equation complexity, tournament
size, Hall-of-Fame size, validation fraction, etc). Results are appended to
the CSV after every run, so an interrupted sweep loses nothing.

## Built-in demo datasets

| Demo dataset | Domain |
|---|---|
| Beam shear (ACI 318) | Structural — concrete beam shear capacity |
| Concrete mix strength (Yeh 1998) | Materials — concrete compressive strength from mix design |
| Lap-splice bond strength (ACI 408 / Orangun-Jirsa-Breen) | Structural — reinforcing bar bond/splice strength |
| Corroded RC beam flexural capacity (ACI 318, corroded) | Structural — flexural capacity under steel corrosion |
| Corrosion processed | Structural — corrosion-reduced flexural capacity, cleaned feature set |
| LC3_Processed | Materials — LC3 (limestone calcined clay cement) mortar compressive strength |
| Punching shear — ML features (ACI 318 / CSA A23.3) | Structural — two-way (punching) shear at slab-column connections |

Each demo comes with a sensible known-formula default already selected
(ACI 318 shear, Bolomey's Law, Waldron and Skillen, an empirical
corrosion-reduction factor, a source-calibrated Abrams/fib strength-age
law, and full ACI 318-19/-25 / CSA A23.3-19/-24 punching-shear equations,
among others) — pick a different one from the dropdown, or clear it back
to "(none / custom)" to have GEP fit the target directly.

## What you get out of it

- **Results tab** — ranked term/interaction tables, auto-generated
  predicted-vs-actual plots for the top recurring terms, a breakdown of
  which input ranges the model can actually be trusted over, and one-click
  CSV downloads of the raw results, ranked main effects, and ranked terms.
- **Equation building tab** — turn the discovered terms into an actual
  usable equation: fit a proper held-out-tested weighted regression (OLS
  or Ridge) combining the physics baseline with the top terms, or
  hand-pick and tune terms yourself, then generate a polished, shareable
  HTML report summarizing the whole run (dataset, formulas, search
  settings/hyperparameters, fit statistics, and the supporting plots).

## Files

- `gep_core.py` — the GEP evolution engine (dataset/physics-formula
  agnostic).
- `run_sweep.py` — CLI + `sweep()` generator that drives multi-seed ×
  multi-training-availability runs (used by both the CLI and the app).
- `parse_terms.py` — sympy-based term/interaction tallying, CLI + reusable
  `parse_results_df()`.
- `app.py` — the Streamlit UI wrapping all of the above, including the
  built-in demo datasets and known-formula library.
- `*.xlsx`, `*.csv` — bundled demo datasets (see table above).

## Practical notes

- **Runs entirely on your own machine** — nothing is uploaded anywhere,
  there's no external service involved.
- **Runtime depends on your search settings** — a small, quick sweep (few
  seeds, few training levels, low generations) can finish in a minute or
  two; a thorough sweep (many seeds × many training levels × hundreds of
  generations) can take from several minutes to a few hours. The app shows
  live progress and an ETA after the first run completes.
- **You don't need to know GEP or evolutionary search to use it** — the
  defaults are sensible starting points, and every control has an inline
  explanation of what it does and why you'd change it.

## Caveats

- The physics-formula parser whitelists a small set of operators/functions
  via `ast` inspection — it's meant to stop typos/unsupported syntax, not
  to sandbox arbitrary untrusted input from strangers on the internet.
  Don't expose a hosted instance of this app to untrusted users without
  additional hardening.
