"""
Multi-seed x multi-training-data-availability GEP sweep.

Runs gep_core.run_single() once per (seed, ptrain) combination and appends
every Hall-of-Fame individual's record to a CSV as soon as it finishes, so
progress is never lost if the sweep is interrupted.

Dataset-agnostic: pass --data pointing at any .csv/.xlsx file, --target for
the column to predict, --features for the columns to use as GEP terminals
(defaults to "every other column"), and optionally --physics for a known
closed-form term (e.g. a code equation) to subtract off before GEP fits the
residual.

Usage:
    python run_sweep.py --data mydata.csv --target y
    python run_sweep.py --data beam_v1_Deep.xlsx --target Vu \
        --physics "0.17*sqrt(fc)*bw*d/1000" \
        --seeds 10,28,50 --ptrains 0.05,0.15,0.5 --generations 750
    python run_sweep.py --out results/smoketest.csv --data beam_v1_Deep.xlsx --target Vu \
        --seeds 10 --ptrains 0.15 --generations 60 --pop-size 60
"""
import argparse
import csv
import os
import time

from gep_core import load_dataset, run_single, sanitize_feature_names

DEFAULT_SEEDS = [10, 28, 50, 75, 95, 104, 105, 118, 187, 188]
DEFAULT_PTRAINS = [0.05, 0.10, 0.15, 0.25, 0.50, 0.75]

FIELDNAMES = [
    "seed", "ptrain", "rank", "n_train", "n_val", "n_test",
    "train_mae", "val_mae", "val_r2", "trainval_r2", "test_mae", "test_rmse", "test_r2", "expr",
    "combine_mode", "n_physics_clipped",
]


def parse_float_list(s):
    return [float(x) for x in s.split(",") if x.strip() != ""]


def parse_int_list(s):
    """Accepts comma-separated ints and/or ranges, e.g. "0-19" or "1,3,5-8"."""
    out = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk[1:]:  # allow leading '-' for negative numbers, split on inner '-'
            a, b = chunk.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(chunk))
    return out


def parse_str_list(s):
    return [x.strip() for x in s.split(",") if x.strip() != ""]


def sweep(
    df, feature_names, target_name, seeds, ptrains,
    physics_formula=None, combine_mode="add", generations=750, pop_size=100, head_length=2,
    n_genes=2, tournsize=5, champs=3, val_frac_of_pool=0.15, verbose=False,
):
    """Generator yielding one dict per completed (seed, ptrain) run:
    {done, total, seed, ptrain, dt, records}. The caller is responsible for
    persisting `records` (e.g. appending to a CSV, or an in-memory list for
    the Streamlit app) and reporting progress.

    `combine_mode` ("add" or "mult") controls how physics and GEP's
    discovered term are combined -- see gep_core.combine_physics(). It's a
    single whole-sweep setting (not per seed/ptrain), matching how
    `physics_formula` itself is already handled."""
    total = len(seeds) * len(ptrains)
    done = 0
    for ptrain in ptrains:
        for seed in seeds:
            t0 = time.time()
            records = run_single(
                df, feature_names, target_name,
                seed=seed, ptrain=ptrain, physics_formula=physics_formula,
                combine_mode=combine_mode,
                generations=generations, pop_size=pop_size, head_length=head_length,
                n_genes=n_genes, tournsize=tournsize, champs=champs,
                val_frac_of_pool=val_frac_of_pool, verbose=verbose,
            )
            done += 1
            dt = time.time() - t0
            yield dict(done=done, total=total, seed=seed, ptrain=ptrain, dt=dt, records=records)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="beam_v1_Deep.xlsx", help="path to source dataset (.csv or .xlsx)")
    ap.add_argument("--target", default=None, help="target column name (default: last column)")
    ap.add_argument("--features", type=parse_str_list, default=None,
                     help="comma-separated feature column names (default: every column except --target)")
    ap.add_argument("--physics", default=None,
                     help="optional physics formula referencing feature columns, "
                          "e.g. \"0.17*sqrt(fc)*bw*d/1000\" (default: none, GEP fits the target directly)")
    ap.add_argument("--combine-mode", choices=["add", "mult"], default="add",
                     help="how physics and GEP's discovered term are combined: 'add' (physics + GEP(X), "
                          "the default) or 'mult' (physics * (1 + GEP(X)), a fractional correction)")
    ap.add_argument("--out", default="results/sweep_results.csv", help="output CSV path (appended to)")
    ap.add_argument("--seeds", type=parse_int_list, default=DEFAULT_SEEDS,
                     help="comma-separated ints and/or ranges, e.g. 0-19 or 10,28,50")
    ap.add_argument("--ptrains", type=parse_float_list, default=DEFAULT_PTRAINS)
    ap.add_argument("--generations", type=int, default=750)
    ap.add_argument("--pop-size", type=int, default=100)
    ap.add_argument("--head-length", type=int, default=2, help="equation complexity: max sub-tree depth per gene")
    ap.add_argument("--n-genes", type=int, default=2, help="equation complexity: number of gene modules summed together")
    ap.add_argument("--tournsize", type=int, default=5)
    ap.add_argument("--champs", type=int, default=3, help="Hall-of-Fame size (solutions recorded per run)")
    ap.add_argument("--val-frac", type=float, default=0.15,
                     help="fraction of each run's train+val pool held out to score/rank candidates during the search")
    ap.add_argument("--fresh", action="store_true", help="overwrite --out instead of appending")
    ap.add_argument("--verbose", action="store_true", help="print gep_simple generation logs")
    args = ap.parse_args()

    df = load_dataset(args.data)
    target_name = args.target or df.columns[-1]
    feature_names_raw = args.features or [c for c in df.columns if c != target_name]

    # GEP/sympy symbol names must be valid, non-keyword Python identifiers;
    # real spreadsheet headers (e.g. "a/d") often aren't, so sanitize.
    name_map = sanitize_feature_names(feature_names_raw)
    if any(k != v for k, v in name_map.items()):
        print("Renamed feature columns to valid identifiers for GEP:")
        for k, v in name_map.items():
            if k != v:
                print(f"  {k!r} -> {v}")
        print("(use these safe names, not the original column headers, in --physics)")
    df = df.rename(columns=name_map)
    feature_names = [name_map[c] for c in feature_names_raw]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_header = args.fresh or not os.path.exists(args.out)
    mode = "w" if args.fresh else "a"

    total = len(args.seeds) * len(args.ptrains)
    print(f"Sweeping {len(args.seeds)} seeds x {len(args.ptrains)} data-availability levels = {total} runs")
    print(f"target={target_name}")
    print(f"features={feature_names}")
    print(f"physics={args.physics!r}")
    print(f"combine_mode={args.combine_mode!r}")
    print(f"seeds={args.seeds}")
    print(f"ptrains={args.ptrains}")

    t_start = time.time()
    with open(args.out, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
            f.flush()

        for progress in sweep(
            df, feature_names, target_name, args.seeds, args.ptrains,
            physics_formula=args.physics, combine_mode=args.combine_mode, generations=args.generations,
            pop_size=args.pop_size, head_length=args.head_length,
            n_genes=args.n_genes, tournsize=args.tournsize, champs=args.champs,
            val_frac_of_pool=args.val_frac, verbose=args.verbose,
        ):
            for rec in progress["records"]:
                writer.writerow(rec)
            f.flush()
            best = progress["records"][0]
            print(
                f"[{progress['done']}/{progress['total']}] seed={progress['seed']} ptrain={progress['ptrain']} "
                f"({progress['dt']:.1f}s) test_r2={best['test_r2']:.3f} test_mae={best['test_mae']:.2f} expr={best['expr']}"
            )

    print(f"Done in {time.time()-t_start:.1f}s. Results written to {args.out}")


if __name__ == "__main__":
    main()
