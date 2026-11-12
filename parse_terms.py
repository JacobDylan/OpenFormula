"""
Parse every recorded GEP solution (from run_sweep.py's output CSV, or an
in-memory DataFrame with the same columns) and tally which features /
feature-interactions recur most often, across seeds and training-data-
availability levels.

Two tallies are produced:
  1. Main-effect frequency: for how many recorded solutions does each single
     feature appear anywhere in the (expanded) expression at all.
  2. Interaction-term frequency: treating the GEP expression as a sum of
     terms (after sympy.expand), each term is canonicalized to a
     variable/exponent signature (e.g. "bw * d", "astm / ag", "fc"), and we
     count how many recorded solutions contain each unique signature. Each
     row's `example` column shows that term combined with the run's
     injected physics formula (if any), i.e. a complete example candidate
     model: physics(features) + this row's term.

This module is dataset-agnostic: it discovers feature names directly from
the symbols present in each recorded expression (via sympy), so it does not
need to know the feature list up front. It exposes reusable functions
(`parse_results_df`) for programmatic use (e.g. from the Streamlit app) in
addition to a CLI entry point.

Usage:
    python parse_terms.py --in results/sweep_results.csv --out-dir results/term_report
"""
import argparse
import os
from collections import Counter, defaultdict

import pandas as pd
import sympy


def canonical_term_signature(term):
    """Return a hashable, canonical (var, exponent) signature for a sympy
    expanded term, dropping numeric coefficients/signs. Returns None for a
    pure-constant term (no feature variables)."""
    powers = term.as_powers_dict()
    sig = tuple(sorted(
        (str(base), exp) for base, exp in powers.items()
        if isinstance(base, sympy.Symbol)
    ))
    return sig if sig else None


def signature_label(sig):
    if not sig:
        return "(constant)"
    parts = []
    for var, exp in sig:
        if exp == 1:
            parts.append(var)
        else:
            parts.append(f"{var}^{exp}")
    return " * ".join(parts)


def term_expr_string(sig):
    """Render a term signature as a clean math expression string (e.g.
    "fc/ad" rather than the "ad^-1 * fc" label), via sympy so formatting
    (division, powers) matches how expressions are printed elsewhere."""
    expr = sympy.Mul(*[sympy.Symbol(v) ** e for v, e in sig])
    return str(expr)


def example_model_string(sig, physics_formula=None):
    """Build the 'what this row's discovered model looks like' example:
    the user's injected physics formula (if any) plus this row's term."""
    term_str = term_expr_string(sig)
    if physics_formula:
        return f"{physics_formula} + {term_str}"
    return term_str


def _fmt_exp(exp):
    """Render an exponent for prose, dropping a trailing .0 when it's a whole number."""
    return str(int(exp)) if float(exp) == int(exp) else str(exp)


def describe_term(sig):
    """Return a short, rule-based (no LLM) plain-English description of what
    kind of relationship a term signature represents -- e.g. product,
    ratio, or power-law dependence. This is a placeholder for a future
    LLM-generated, domain-aware description; it only describes the
    mathematical *shape* of the term, not its physical meaning."""
    if not sig:
        return "A constant term with no dependence on any feature."

    if len(sig) == 1:
        var, exp = sig[0]
        if exp == 1:
            return f"A direct (linear) dependence on `{var}` alone."
        if exp == -1:
            return f"An inverse dependence on `{var}` alone (proportional to 1/{var})."
        if exp > 0:
            return f"A power-law dependence on `{var}` alone, raised to the {_fmt_exp(exp)}."
        return f"An inverse power-law dependence on `{var}` alone (proportional to 1/{var}^{_fmt_exp(-exp)})."

    numerator = [(v, e) for v, e in sig if e > 0]
    denominator = [(v, e) for v, e in sig if e < 0]

    def _side(vars_exps, invert=False):
        bits = []
        for v, e in vars_exps:
            e_disp = -e if invert else e
            bits.append(f"`{v}`" if e_disp == 1 else f"`{v}`^{_fmt_exp(e_disp)}")
        return " and ".join(bits) if len(bits) <= 2 else ", ".join(bits[:-1]) + f", and {bits[-1]}"

    if numerator and denominator:
        num_str = _side(numerator)
        den_str = _side(denominator, invert=True)
        return (
            f"A ratio combining {num_str} (numerator) and {den_str} (denominator) -- "
            f"the term grows with {num_str} and shrinks with {den_str}."
        )
    if numerator and not denominator:
        var_str = _side(numerator)
        n = len(numerator)
        return (
            f"A multiplicative interaction between {var_str} -- {n} features multiplied "
            f"together, so the term only grows large when {'all of them are' if n > 2 else 'both are'} large simultaneously, "
            "not from any one alone."
        )
    var_str = _side(denominator, invert=True)
    return f"An inverse (dividing) dependence on {var_str}."


def parse_expression(expr_str):
    """Return (set_of_variables_present, list_of_term_signatures) for one
    recorded GEP solution string. Feature symbols are discovered directly
    from the expression text (no fixed feature list needed)."""
    expr = sympy.sympify(expr_str)
    expanded = sympy.expand(expr)
    terms = expanded.as_ordered_terms() if expanded != 0 else []

    term_sigs = []
    variables_present = set()
    for term in terms:
        sig = canonical_term_signature(term)
        if sig is None:
            continue
        term_sigs.append(sig)
        variables_present.update(v for v, _ in sig)

    return variables_present, term_sigs


def parse_results_df(df, by_ptrain=False, physics_formula=None):
    """Parse every row of a results DataFrame (columns: at least `expr`, and
    `ptrain` if by_ptrain=True). `physics_formula`, if given, is the injected
    physics term the user supplied for the sweep (e.g.
    "0.17*sqrt(fc)*bw*d/1000") -- it's shown combined with each row's term in
    the `example` column, so each row reads as a complete candidate model:
    injected physics + that row's discovered feature term.

    Returns a dict with:
      - var_df: main-effect ranking (feature, n_solutions, pct_of_solutions) --
        secondary info, not the main focus; report it last
      - term_df: term/interaction ranking (term, n_variables, n_solutions,
        pct_of_solutions, example) -- this is the primary result
      - term_sigs: list of raw (var, exponent) signatures, term_sigs[i]
        corresponds to term_df row i (same order) -- lets callers re-evaluate
        a specific term's numeric value on data (e.g. for plotting)
      - by_ptrain_df: (only if by_ptrain) long-format per-ptrain term ranking
      - var_by_ptrain: (only if by_ptrain) {ptrain: Counter(feature->count)}
      - n_solutions, parse_failures, failure_msgs
    """
    n_solutions = len(df)
    var_counts = Counter()
    term_counts = Counter()
    parse_failures = 0
    failure_msgs = []

    var_counts_by_ptrain = defaultdict(Counter)
    term_counts_by_ptrain = defaultdict(Counter)

    for _, row in df.iterrows():
        try:
            variables_present, term_sigs = parse_expression(row["expr"])
        except Exception as e:
            parse_failures += 1
            failure_msgs.append(f"failed to parse expr={row['expr']!r}: {e}")
            continue

        for v in variables_present:
            var_counts[v] += 1
            if by_ptrain:
                var_counts_by_ptrain[row["ptrain"]][v] += 1

        for sig in set(term_sigs):  # de-dup within one expression
            term_counts[sig] += 1
            if by_ptrain:
                term_counts_by_ptrain[row["ptrain"]][sig] += 1

    var_rows = [
        {"feature": v, "n_solutions": c, "pct_of_solutions": round(100 * c / n_solutions, 1)}
        for v, c in var_counts.most_common()
    ] if n_solutions else []
    var_df = pd.DataFrame(var_rows)

    ranked_sigs = [sig for sig, _ in term_counts.most_common()]
    term_rows = [
        {
            "term": signature_label(sig),
            "n_variables": len(sig),
            "n_solutions": c,
            "pct_of_solutions": round(100 * c / n_solutions, 1),
            "example": example_model_string(sig, physics_formula),
        }
        for sig, c in term_counts.most_common()
    ] if n_solutions else []
    term_df = pd.DataFrame(term_rows)

    result = dict(
        var_df=var_df, term_df=term_df,
        # term_sigs[i] is the raw (var, exponent) signature for term_df row i --
        # lets callers (e.g. the app's plots) re-evaluate a specific term on data.
        term_sigs=ranked_sigs,
        n_solutions=n_solutions, parse_failures=parse_failures,
        failure_msgs=failure_msgs,
    )

    if by_ptrain:
        by_ptrain_rows = []
        for ptrain in sorted(term_counts_by_ptrain):
            n_this = len(df[df["ptrain"] == ptrain])
            for sig, c in term_counts_by_ptrain[ptrain].most_common():
                by_ptrain_rows.append({
                    "ptrain": ptrain, "term": signature_label(sig),
                    "n_solutions": c, "pct_of_solutions": round(100 * c / n_this, 1),
                })
        result["by_ptrain_df"] = pd.DataFrame(by_ptrain_rows)
        result["var_by_ptrain"] = var_counts_by_ptrain

    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="infile", default="results/sweep_results.csv")
    ap.add_argument("--out-dir", default="results/term_report")
    ap.add_argument("--by-ptrain", action="store_true", help="also break down tallies per training-data-availability level")
    ap.add_argument("--physics", default=None,
                     help="the injected physics formula used for this sweep (if any), shown "
                          "combined with each term in the 'example' column, e.g. \"0.17*sqrt(fc)*bw*d/1000\"")
    args = ap.parse_args()

    df = pd.read_csv(args.infile)
    print(f"Loaded {len(df)} recorded solutions from {args.infile}")

    res = parse_results_df(df, by_ptrain=args.by_ptrain, physics_formula=args.physics)
    n_solutions = res["n_solutions"]

    for msg in res["failure_msgs"]:
        print(f"  ! {msg}")
    if res["parse_failures"]:
        print(f"WARNING: {res['parse_failures']}/{n_solutions} expressions failed to parse")

    os.makedirs(args.out_dir, exist_ok=True)

    var_df = res["var_df"]
    term_df = res["term_df"]
    var_df.to_csv(os.path.join(args.out_dir, "main_effects_ranked.csv"), index=False)
    term_df.to_csv(os.path.join(args.out_dir, "terms_ranked.csv"), index=False)

    # Primary result first: which terms/interactions GEP kept re-discovering.
    print("\n=== TERM / INTERACTION FREQUENCY (most -> least common) ===")
    print(term_df.to_string(index=False))

    interactions_only = term_df[term_df["n_variables"] >= 2] if len(term_df) else term_df
    print("\n=== MULTI-VARIABLE INTERACTIONS ONLY (most -> least common) ===")
    if len(interactions_only):
        print(interactions_only.to_string(index=False))
    else:
        print("(none found)")

    if args.by_ptrain:
        res["by_ptrain_df"].to_csv(
            os.path.join(args.out_dir, "terms_ranked_by_ptrain.csv"), index=False
        )

    # Secondary/supporting info last: raw per-feature presence, not the main focus.
    print("\n=== MAIN-EFFECT FEATURE FREQUENCY (most -> least common) ===")
    print(var_df.to_string(index=False))

    if args.by_ptrain:
        print("\n=== MAIN EFFECTS BY TRAINING-DATA-AVAILABILITY LEVEL ===")
        var_by_ptrain = res["var_by_ptrain"]
        for ptrain in sorted(var_by_ptrain):
            n_this = len(df[df["ptrain"] == ptrain])
            print(f"\n-- ptrain={ptrain} (n={n_this} solutions) --")
            for v, c in var_by_ptrain[ptrain].most_common():
                print(f"  {v:8s} {c:4d}  ({100*c/n_this:.0f}%)")

    print(f"\nReports written to {args.out_dir}/")


if __name__ == "__main__":
    main()
