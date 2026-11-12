"""
Core GEP (Gene Expression Programming) machinery for physics-informed feature
discovery, generalized from the original beam-shear pipeline
(FeatureInteractionGEP.ipynb, "Source Model" / cell 19, evaluate_MAE_GEP with
PIML='Beam').

Fitness = MAE(y_experimental, physics(X) + GEP(X))                  [additive]
       or MAE(y_experimental, physics(X) * (1 + GEP(X)))      [multiplicative]

where `physics(X)` is an OPTIONAL, user-supplied closed-form term (e.g. a
known code equation / physical law) computed directly from the feature
columns, and GEP is left to discover whatever additional feature
relationships explain the remaining gap (e.g. arch action in deep beams,
a/d effects, etc., in the original beam-shear use case). If no physics
formula is supplied, GEP simply fits the target directly (physics(X) == 0,
so the multiplicative form degenerates to always-zero and isn't useful).

Which of the two combination modes is used is controlled by the caller via
`combine_mode` ("add", the default, or "mult"); see `combine_physics()` and
`run_single()` below. In "mult" mode, GEP still evolves toward 0 meaning "no
adjustment needed" (not toward 1), so the correction term is a *fractional*
adjustment: physics(X) * (1 + GEP(X)).

This module is dataset-agnostic: the caller supplies the feature column
names, target column name, and (optionally) a physics formula string
referencing those feature names. Nothing here is hard-coded to a specific
dataset any more (see git history / FeatureInteractionGEP.ipynb for the
original beam-shear-specific version with the ACI 318 Vc formula baked in).

Notes on deviations from the original notebook (see conversation for why):
  - The original control panel had the physics term hard-coded to 0 and
    PIML='Default' (i.e. disabled), and the 'Beam' branch referenced an
    undefined `phys_train` variable. This module re-enables and generalizes
    that mechanism via a user-supplied formula string.
  - `ptrain` here means: fraction of the WHOLE dataset put into the
    train+val pool; the pool is split 85/15 into train/val; the remainder
    (1-ptrain) is held out as test. This differs from the notebook's
    `1 - (ptrain/0.5)` formula, which produces invalid (negative) test
    sizes for ptrain >= 0.5. The direct definition here is monotonic and
    valid for any ptrain in (0, 1).
  - Each (seed, ptrain) run does ONE GEP evolution (not the notebook's
    "Val=7 restarts, pick best" scheme) so that "seed" has an unambiguous,
    fully-reproducible meaning: it drives both the train/val/test split AND
    all GEP stochasticity (population init, mutation, crossover, selection).
  - The linker across genes is a variadic sum (`link_sum`), not
    `operator.add`, so any `n_genes >= 1` works (geppy calls
    `linker(*(gene_result for each gene))`, so a fixed-arity add only works
    for n_genes==2).
"""
import ast
import contextlib
import io
import keyword
import operator
import random
import re
import warnings

import numpy as np
import pandas as pd
from deap import base, creator, tools
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# geppy prints (not warns) a "graphviz not available" notice at import time if
# the optional `graphviz` package isn't installed -- harmless, since we never
# use geppy's tree-visualization feature (geppy.graph / export_expression_tree),
# but it's noisy in the terminal. warnings.filterwarnings above doesn't catch
# it since it's a plain print(), so swallow stdout for just this import.
with contextlib.redirect_stdout(io.StringIO()):
    import geppy as gep  # noqa: E402


# --------------------------------------------------------------------------
# Protected operators (verbatim from FeatureInteractionGEP.ipynb, cell 8)
# --------------------------------------------------------------------------
def protected_div(x1, x2):
    if abs(x2) < 1e-6:
        return 1
    return x1 / x2


def link_sum(*genes):
    """Variadic sum linker: geppy calls linker(*(gene_result for each gene)),
    so a fixed-arity operator.add only works for n_genes==2. This generalizes
    to any number of genes (equivalent to operator.add when n_genes==2)."""
    return sum(genes)


# --------------------------------------------------------------------------
# Protected vectorized versions of division/log for evaluating user formulas
# and re-evaluated GEP-discovered terms against real data (see
# _ProtectDivisions and make_physics_fn below). Real datasets routinely have
# features that are legitimately, structurally zero for many rows (e.g. a
# concrete mix's fly-ash content when that mix uses none) -- a bare `x/feature`
# or `log(feature)` blows up to inf/-inf at those rows, which then poisons
# R^2, fitness, or a whole sweep with an unhandled exception. GEP's own
# internal search is already immune to this (its primitive set uses
# `protected_div`, not raw division) -- these mirror that same protection for
# every OTHER place a formula/term gets evaluated numerically.
# --------------------------------------------------------------------------
_ZERO_EPS = 1e-6


def protected_div_arr(x1, x2):
    """Vectorized division: wherever |x2| < _ZERO_EPS, returns 1 there instead
    of dividing (matching `protected_div`'s scalar convention below), so a
    near-zero or exactly-zero denominator can't produce inf/nan."""
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        near_zero = np.abs(x2) < _ZERO_EPS
        result = np.where(near_zero, 1.0, x1 / np.where(near_zero, 1.0, x2))
    return result


def _protected_log_like(np_log_fn):
    """Build a protected version of a numpy log-family function: returns 0
    wherever |x| <= _ZERO_EPS (instead of -inf at 0, or nan for negative x),
    else np_log_fn(|x|)."""
    def fn(x):
        x = np.asarray(x, dtype=float)
        ax = np.abs(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(ax > _ZERO_EPS, np_log_fn(np.where(ax > _ZERO_EPS, ax, 1.0)), 0.0)
    return fn


def protected_sqrt(x):
    """sqrt(|x|) -- avoids nan for a formula term that goes slightly negative
    due to floating point, without changing behavior at/near zero (sqrt(0)==0
    already)."""
    return np.sqrt(np.abs(np.asarray(x, dtype=float)))


# --------------------------------------------------------------------------
# User-supplied physics formula: parsed with `ast` and restricted to a
# whitelist of feature names + a small set of numpy functions, then compiled
# to a callable physics_fn(X) -> np.ndarray of shape (n,).
# --------------------------------------------------------------------------
SAFE_FUNCS = {
    "sqrt": protected_sqrt, "log": _protected_log_like(np.log),
    "log10": _protected_log_like(np.log10), "log2": _protected_log_like(np.log2),
    "exp": np.exp, "abs": np.abs, "min": np.minimum, "max": np.maximum,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "power": np.power,
    "pi": np.pi,
}


class _ProtectDivisions(ast.NodeTransformer):
    """Rewrite every `a / b` in a formula's AST into a call to the protected
    division helper (`__protected_div__`), so a zero (or near-zero)
    denominator can't silently produce inf/nan when a physics formula or a
    re-evaluated GEP-discovered term is run against real data -- mirroring
    the protection GEP's own search already gets internally."""

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Div):
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(id="__protected_div__", ctx=ast.Load()),
                    args=[node.left, node.right],
                    keywords=[],
                ),
                node,
            )
        return node

_ALLOWED_AST_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub,
    ast.UAdd, ast.Mod, ast.keyword,
)


class FormulaError(ValueError):
    """Raised when a user-supplied physics formula is invalid or unsafe."""


def validate_formula(formula, feature_names):
    """Parse `formula` and ensure it only references `feature_names` and the
    whitelisted SAFE_FUNCS. Raises FormulaError on any violation. Returns the
    compiled code object on success."""
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as e:
        raise FormulaError(f"Invalid syntax: {e}")

    allowed_names = set(feature_names)
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise FormulaError(
                f"Disallowed expression element: {type(node).__name__} "
                "(only arithmetic on feature columns + a small function whitelist is allowed)"
            )
        if isinstance(node, ast.Name) and node.id not in allowed_names and node.id not in SAFE_FUNCS:
            raise FormulaError(
                f"Unknown name '{node.id}': must be one of the feature columns "
                f"({', '.join(feature_names)}) or a function in {sorted(SAFE_FUNCS)}"
            )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_FUNCS:
                raise FormulaError(
                    f"Only these functions may be called: {', '.join(sorted(SAFE_FUNCS))}"
                )

    return compile(tree, "<physics_formula>", "eval")


def make_physics_fn(formula, feature_names):
    """Compile a user-supplied physics formula string (referencing feature
    column names and SAFE_FUNCS) into a callable physics_fn(X) -> np.ndarray,
    where X is (n, n_features) ordered per `feature_names`. If `formula` is
    None/blank, returns the zero function (i.e. GEP fits the target
    directly).

    Every `/` in the formula is silently rewritten to a protected division
    (and `sqrt`/`log`/`log10`/`log2` resolve to protected versions via
    SAFE_FUNCS) before compiling, so a feature that's legitimately zero for
    some rows (common in real data -- e.g. a mix-design ingredient a given
    sample doesn't use) can't produce inf/nan here, the way raw division/log
    would. This mirrors the protection GEP's own internal search already has
    via `protected_div`, extended to formula/term evaluation against real
    data (physics formulas, and re-evaluating a discovered term for plots)."""
    if formula is None or not str(formula).strip():
        return lambda X: np.zeros(X.shape[0], dtype=float)

    validate_formula(formula, feature_names)  # raises FormulaError if invalid; return unused
    tree = ast.parse(formula, mode="eval")
    tree = ast.fix_missing_locations(_ProtectDivisions().visit(tree))
    code = compile(tree, "<physics_formula>", "eval")
    idx = {name: i for i, name in enumerate(feature_names)}

    def physics_fn(X):
        ns = dict(SAFE_FUNCS)
        ns["__protected_div__"] = protected_div_arr
        for name, i in idx.items():
            ns[name] = X[:, i].astype(float)
        result = eval(code, {"__builtins__": {}}, ns)
        return np.broadcast_to(np.asarray(result, dtype=float), (X.shape[0],)).copy()

    return physics_fn


# --------------------------------------------------------------------------
# Combining physics with GEP's discovered term: additive (default) or
# multiplicative. In multiplicative mode, GEP evolves a *fractional*
# correction (0 == physics already correct, matching the additive case's
# "0 == no adjustment" meaning) rather than a direct multiplier, so
# combined_pred = physics * (1 + GEP(X)) instead of physics + GEP(X). This
# keeps the GA's natural search bias (fresh/mutated expressions tend to
# hover near 0, not near 1) working the same way in both modes, and lets the
# app's weighted-regression stage stay a plain linear fit (regressing
# y/physics - 1 on the discovered terms) instead of needing a nonlinear
# solver.
# --------------------------------------------------------------------------
def clip_physics_for_mult(phys_arr, eps):
    """Clip values of a physics baseline array whose magnitude is < eps away
    from zero (keeping their sign; treating an exact zero as positive), so a
    multiplicative combination (physics * (1 + correction)) never divides by
    -- or multiplies through -- an unstably-small baseline. Returns
    (clipped_array, n_clipped)."""
    phys_arr = np.asarray(phys_arr, dtype=float)
    mask = np.abs(phys_arr) < eps
    if not np.any(mask):
        return phys_arr, 0
    out = phys_arr.copy()
    signs = np.sign(out[mask])
    signs[signs == 0] = 1.0
    out[mask] = signs * eps
    return out, int(mask.sum())


def physics_clip_eps(y_ref, rel=1e-3, floor=1e-9):
    """A reasonable epsilon for `clip_physics_for_mult`, scaled to the data:
    a small fraction of the typical (median absolute, nonzero) magnitude of
    `y_ref` (e.g. the target column), with an absolute floor so an all-zero
    reference doesn't collapse eps to 0."""
    y_ref = np.asarray(y_ref, dtype=float)
    nonzero = y_ref[y_ref != 0]
    scale = float(np.median(np.abs(nonzero))) if nonzero.size else 0.0
    return max(floor, rel * scale)


def combine_physics(physics, correction, combine_mode="add"):
    """Combine a physics baseline array with GEP's (or a fitted model's)
    correction term, per `combine_mode`:
      - "add":  physics + correction
      - "mult": physics * (1 + correction)  (correction is a fractional
                adjustment; 0 means physics is left unchanged)
    """
    if combine_mode == "mult":
        return physics * (1.0 + correction)
    return physics + correction


# --------------------------------------------------------------------------
# Column-name sanitization: geppy's PrimitiveSet terminals (and sympy symbols
# built from them) require valid, non-keyword Python identifiers. Real-world
# spreadsheet headers rarely satisfy this ("a/d", "a:M/V", "f'c (MPa)", ...),
# so we map arbitrary column names to safe, unique identifiers and use those
# internally for GEP/sympy, while keeping the mapping around so results can
# be reported back using the safe names (which are also what a physics
# formula must reference).
# --------------------------------------------------------------------------
def sanitize_identifier(name, used):
    """Turn an arbitrary string into a safe, unique Python identifier not
    already present in `used` (which is mutated to include the result)."""
    s = re.sub(r"\W", "_", str(name))
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "f"
    if not (s[0].isalpha() or s[0] == "_"):
        s = f"f_{s}"
    if keyword.iskeyword(s):
        s = f"{s}_"
    base, i, candidate = s, 1, s
    while candidate in used:
        candidate = f"{base}_{i}"
        i += 1
    used.add(candidate)
    return candidate


def sanitize_feature_names(names):
    """Return an ordered dict {original_column_name: safe_identifier} for a
    list of (possibly not-identifier-safe) column names, guaranteeing the
    safe identifiers are unique."""
    used = set()
    return {n: sanitize_identifier(n, used) for n in names}


# --------------------------------------------------------------------------
# Data loading / splitting
# --------------------------------------------------------------------------
def load_dataset(path_or_buffer, filename_hint=None):
    """Load a .csv or .xlsx dataset into a DataFrame (deduplicated). Accepts
    a path string or a file-like object (e.g. a Streamlit upload); in the
    latter case pass `filename_hint` (e.g. the original filename) so the
    format can be detected."""
    name = filename_hint if filename_hint is not None else path_or_buffer
    is_csv = str(name).lower().endswith(".csv") or str(name).lower().endswith(".tsv")
    if is_csv:
        sep = "\t" if str(name).lower().endswith(".tsv") else ","
        df = pd.read_csv(path_or_buffer, sep=sep)
    else:
        df = pd.read_excel(path_or_buffer)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def split_data(df, feature_names, target_name, ptrain, seed, val_frac_of_pool=0.15):
    """
    ptrain: fraction of the WHOLE dataset allocated to the train+val pool.
    The pool is split into train/val with val_frac_of_pool of the pool held
    out for val. The remaining (1 - ptrain) of the whole dataset is the test
    set. All splitting uses `seed` as random_state (also reused to seed
    `random`/`numpy` for GEP's own stochasticity -> the run is fully
    reproducible from `seed` alone).
    """
    X = df[feature_names].to_numpy(dtype=float)
    y = df[target_name].to_numpy(dtype=float)

    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, train_size=ptrain, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_pool, y_pool, test_size=val_frac_of_pool, random_state=seed
    )
    return dict(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        X_whole=X, y_whole=y,
    )


def _cols(X):
    """Split a (n, n_features) array into a list of 1D column arrays, matching
    the calling convention `func(*cols)` expected by geppy-compiled individuals
    whose terminals are ordered as feature_names."""
    return [X[:, i] for i in range(X.shape[1])]


# --------------------------------------------------------------------------
# GEP run for a single (seed, ptrain) combination
# --------------------------------------------------------------------------
def run_single(
    df,
    feature_names,
    target_name,
    seed,
    ptrain,
    physics_formula=None,
    combine_mode="add",
    generations=750,
    pop_size=100,
    head_length=2,
    n_genes=2,
    tournsize=5,
    champs=3,
    rnc_array_length=10,
    val_frac_of_pool=0.15,
    verbose=False,
):
    """Run one GEP evolution for a given seed & training-data-availability
    level, fitting GEP(features) to explain the gap between the target and
    physics(features) -- either additively (target - physics, the default)
    or multiplicatively (target / physics - 1, if combine_mode="mult"; GEP
    then evolves a fractional correction rather than the residual itself).
    Returns a list of result dicts, one per Hall-of-Fame individual
    (rank 1 = best)."""
    split = split_data(
        df, feature_names, target_name, ptrain=ptrain, seed=seed,
        val_frac_of_pool=val_frac_of_pool,
    )

    # Seed all stochastic GEP machinery (population init, mutation, crossover,
    # tournament selection) from the same seed used for the data split.
    random.seed(seed)
    np.random.seed(seed)

    physics_fn = make_physics_fn(physics_formula, feature_names)
    phys_train = physics_fn(split["X_train"])
    phys_val = physics_fn(split["X_val"])
    phys_test = physics_fn(split["X_test"])

    # In multiplicative mode, physics * (1 + correction) breaks down for any
    # row where physics is at/near zero (the correction can never move the
    # prediction away from zero there, and computing the fractional target
    # target/physics - 1 for the "Build your own equation" style refits
    # elsewhere would divide by ~0). Clip those rows' physics baseline away
    # from zero and report how many were affected so the caller can warn.
    n_physics_clipped = 0
    if combine_mode == "mult":
        eps = physics_clip_eps(split["y_train"])
        phys_train, n_tr = clip_physics_for_mult(phys_train, eps)
        phys_val, n_va = clip_physics_for_mult(phys_val, eps)
        phys_test, n_te = clip_physics_for_mult(phys_test, eps)
        n_physics_clipped = n_tr + n_va + n_te

    X_train_cols = _cols(split["X_train"])
    X_val_cols = _cols(split["X_val"])
    X_test_cols = _cols(split["X_test"])
    y_train, y_val, y_test = split["y_train"], split["y_val"], split["y_test"]

    # --- primitive set (add, sub, mul, protected_div + RNC terminal) ---
    pset = gep.PrimitiveSet("Main", input_names=feature_names)
    pset.add_function(operator.add, 2)
    pset.add_function(operator.sub, 2)
    pset.add_function(protected_div, 2)
    pset.add_function(operator.mul, 2)
    pset.add_rnc_terminal()

    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMin)

    toolbox = gep.Toolbox()
    toolbox.register("rnc_gen", random.randint, a=-10, b=10)
    toolbox.register(
        "gene_gen", gep.GeneDc, pset=pset, head_length=head_length,
        rnc_gen=toolbox.rnc_gen, rnc_array_length=rnc_array_length,
    )
    toolbox.register(
        "individual", creator.Individual, gene_gen=toolbox.gene_gen,
        n_genes=n_genes, linker=link_sum,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gep.compile_, pset=pset)

    def evaluate(individual):
        func = toolbox.compile(individual)
        raw = np.array(list(map(func, *X_train_cols)))
        yp = combine_physics(phys_train, raw, combine_mode)
        return (mean_absolute_error(y_train, yp),)

    toolbox.register("evaluate", evaluate)
    toolbox.register("select", tools.selTournament, tournsize=tournsize)
    toolbox.register("mut_uniform", gep.mutate_uniform, pset=pset, ind_pb=0.05, pb=1)
    toolbox.register("mut_invert", gep.invert, pb=0.1)
    toolbox.register("mut_is_transpose", gep.is_transpose, pb=0.1)
    toolbox.register("mut_ris_transpose", gep.ris_transpose, pb=0.1)
    toolbox.register("mut_gene_transpose", gep.gene_transpose, pb=0.1)
    toolbox.register("cx_1p", gep.crossover_one_point, pb=0.3)
    toolbox.register("cx_2p", gep.crossover_two_point, pb=0.2)
    toolbox.register("cx_gene", gep.crossover_gene, pb=0.1)

    stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(champs)

    _, _ = gep.gep_simple(
        pop, toolbox, n_generations=generations, n_elites=1,
        stats=stats, hall_of_fame=hof, verbose=verbose,
    )

    records = []
    for rank, ind in enumerate(hof, start=1):
        func = toolbox.compile(ind)

        yp_train = combine_physics(phys_train, np.array(list(map(func, *X_train_cols))), combine_mode)
        yp_val = combine_physics(phys_val, np.array(list(map(func, *X_val_cols))), combine_mode)
        yp_test = combine_physics(phys_test, np.array(list(map(func, *X_test_cols))), combine_mode)

        # R² over the whole train+val pool (the data this individual was fit
        # and scored against during the search, combined) -- as opposed to
        # val_r2 (just the smaller validation slice) or test_r2 (the
        # separate held-out set). Useful as a filtering signal that doesn't
        # touch the held-out test set at all.
        y_trainval = np.concatenate([y_train, y_val])
        yp_trainval = np.concatenate([yp_train, yp_val])

        expr = gep.simplify(ind)

        records.append(dict(
            seed=seed,
            ptrain=ptrain,
            rank=rank,
            n_train=len(y_train),
            n_val=len(y_val),
            n_test=len(y_test),
            train_mae=mean_absolute_error(y_train, yp_train),
            val_mae=mean_absolute_error(y_val, yp_val),
            val_r2=r2_score(y_val, yp_val),
            trainval_r2=r2_score(y_trainval, yp_trainval),
            test_mae=mean_absolute_error(y_test, yp_test),
            test_rmse=mean_squared_error(y_test, yp_test) ** 0.5,
            test_r2=r2_score(y_test, yp_test),
            expr=str(expr),
            combine_mode=combine_mode,
            n_physics_clipped=n_physics_clipped,
        ))

    return records
