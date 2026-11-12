"""
Physics-Informed GEP Feature Discovery — Streamlit app.

A shareable, self-contained UI around the gep_core/run_sweep/parse_terms
pipeline: the user uploads a dataset, optionally supplies a known physics
formula (a closed-form term to subtract off before GEP fits the residual),
sets equation-complexity / generations / training-availability-sweep /
random-seed knobs, and gets back a ranked table of the feature terms and
interactions GEP discovered most consistently across the sweep.

Run with:
    streamlit run app.py
"""
import ast
import base64
import io
import re
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression, QuantileRegressor, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from gep_core import (
    FormulaError, clip_physics_for_mult, combine_physics, load_dataset, make_physics_fn,
    physics_clip_eps, sanitize_feature_names, validate_formula,
)
from parse_terms import describe_term, parse_results_df, term_expr_string
from run_sweep import parse_float_list, parse_int_list, sweep

st.set_page_config(page_title="OpenFormula", layout="wide")

# "OpenFormula" wordmark: a DNA double-helix icon + white "Open" + blue
# "Formula" in a bold Poppins face, sitting on a diagonally-rounded badge.
# Kept as its own markdown block (rather than folded into the shared .gep-*
# style block below) so the title styling stays easy to find/tweak on its own.
# Badge background (#0E2A47) is a dark, desaturated shade of the same blue
# family as the "Formula" accent (#4FC3F7) -- monochromatic for cohesion,
# but dark enough to give both white and light-blue content strong contrast.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@700;800&display=swap');
    .openformula-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.65rem;
        padding: 0.55rem 1.6rem;
        margin: 0 0 0.3rem 0;
        background: #0E2A47;
        border-radius: 30px 10px 30px 10px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
        width: fit-content;
    }
    .openformula-dna {
        height: clamp(2.1rem, 4vw, 2.75rem);
        width: auto;
        flex-shrink: 0;
    }
    .openformula-title {
        font-family: 'Poppins', 'Trebuchet MS', 'Segoe UI', Helvetica, Arial, sans-serif !important;
        font-weight: 800 !important;
        font-size: clamp(2.1rem, 4vw, 2.75rem) !important;
        letter-spacing: -0.01em;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.1 !important;
    }
    .openformula-title .of-open { color: #FFFFFF !important; }
    .openformula-title .of-formula { color: #4FC3F7 !important; }
    .openformula-tagline {
        font-family: 'Poppins', 'Trebuchet MS', 'Segoe UI', Helvetica, Arial, sans-serif;
        font-size: 1.08rem;
        font-weight: 500;
        color: inherit;
        opacity: 0.88;
        margin: 0.1rem 0 0.6rem 0;
    }
    </style>
    <div class="openformula-badge">
        <svg class="openformula-dna" viewBox="0 0 28 52" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M5,2 C5,10 23,10 23,17 C23,24 5,24 5,32 C5,40 23,40 23,48"
                  stroke="#FFFFFF" stroke-width="2.5" stroke-linecap="round"/>
            <path d="M23,2 C23,10 5,10 5,17 C5,24 23,24 23,32 C23,40 5,40 5,48"
                  stroke="#FFFFFF" stroke-width="2.5" stroke-linecap="round"/>
            <line x1="5" y1="2" x2="23" y2="2" stroke="#FFFFFF" stroke-width="1.6" stroke-linecap="round" opacity="0.55"/>
            <line x1="5" y1="17" x2="23" y2="17" stroke="#FFFFFF" stroke-width="1.6" stroke-linecap="round" opacity="0.55"/>
            <line x1="5" y1="32" x2="23" y2="32" stroke="#FFFFFF" stroke-width="1.6" stroke-linecap="round" opacity="0.55"/>
            <line x1="5" y1="48" x2="23" y2="48" stroke="#FFFFFF" stroke-width="1.6" stroke-linecap="round" opacity="0.55"/>
        </svg>
        <h1 class="openformula-title">
            <span class="of-open">Open</span><span class="of-formula">ForMuLa</span>
        </h1>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="openformula-tagline">'
    "A physics-informed symbolic regression tool: start from a known physics "
    "relationship and let evolutionary search discover what it's missing."
    "</p>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="gep-equation" style="margin-top: 0; display: inline-block;">'
    '<span class="gep-label">Fitness:</span>'
    "error( <em>target</em>, <em>physics</em>(x) + <em>GEP</em>(x) )"
    "</div>",
    unsafe_allow_html=True,
)
st.caption(
    "Upload a dataset — optionally alongside a known closed-form physics formula — "
    "and it evolves candidate symbolic expressions across many random seeds and "
    "training-data sizes, then reports which feature terms and interactions show up "
    "most consistently."
)

# ------------------------------------------------------------------------------
# Shared styling for the small explanatory blurbs and generated-equation
# call-outs sprinkled through the app: a soft rounded box (instead of plain,
# low-contrast caption text) set in a readable serif face rather than the
# monospace `code` font, so formulas read more like typeset math than raw
# code. Colors use theme-relative rgba() (not hard-coded hex) so it holds up
# in both light and dark Streamlit themes, and `color: inherit` lets the
# surrounding theme's text color carry through into the labels.
st.markdown(
    """
    <style>
    .gep-note, .gep-equation {
        font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
        border-radius: 10px;
        padding: 0.55rem 0.9rem;
        margin: 0.3rem 0 1rem 0;
        line-height: 1.5;
        color: inherit;
    }
    .gep-note {
        background-color: rgba(151, 166, 195, 0.15);
        border: 1px solid rgba(151, 166, 195, 0.35);
        font-size: 1.02rem;
    }
    .gep-equation {
        background-color: rgba(60, 145, 90, 0.10);
        border: 1px solid rgba(60, 145, 90, 0.35);
        font-size: 1.12rem;
    }
    .gep-note .gep-label, .gep-equation .gep-label {
        font-family: "Trebuchet MS", "Segoe UI", sans-serif;
        font-weight: 700;
        margin-right: 0.3rem;
    }
    .gep-note em, .gep-equation em { opacity: 0.85; }
    /* Typeset math for equation text rendered via _equation_to_html:
       stacked fractions, radicals, and superscript exponents, instead of
       raw ASCII '/'/'**'. currentColor (not a hard-coded hex) so the
       fraction bar/radical line holds up in both light and dark themes. */
    .gep-note .frac, .gep-equation .frac {
        display: inline-flex; flex-direction: column; vertical-align: middle;
        text-align: center; margin: 0 0.15em; font-style: normal; line-height: 1.15;
    }
    .gep-note .frac .num, .gep-equation .frac .num {
        display: block; padding: 0 0.2em 0.1em; border-bottom: 1.4px solid currentColor;
    }
    .gep-note .frac .den, .gep-equation .frac .den { display: block; padding: 0.1em 0.2em 0; }
    .gep-note .sqrt, .gep-equation .sqrt { white-space: nowrap; font-style: normal; }
    .gep-note .sqrt .radicand, .gep-equation .sqrt .radicand {
        border-top: 1.4px solid currentColor; padding: 0 0.2em; margin-left: 0.05em;
    }
    .gep-note sup, .gep-equation sup { font-size: 0.72em; }
    /* st.caption() renders noticeably smaller than normal body text by
       default -- it's used throughout the app for real explanatory
       content (not just fine print), so bump it closer to body size
       while keeping it a touch smaller/lighter than regular text so it
       still reads as secondary/supporting. */
    [data-testid="stCaptionContainer"] { font-size: 0.97rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# The setup sidebar (dataset/target/physics/complexity/search/run controls)
# starts wide -- as wide as it can be without crowding the OpenFormula logo
# in the main area -- so there's room to fit more per-feature explanations
# in as the app grows, without every setup step feeling cramped. Once a
# search has actually completed and the results section has something to
# show, it shrinks back down to Streamlit's normal sidebar width so the
# results (rendered in the main/wide area) get the room instead.
#
# Streamlit's sidebar already supports a native drag-to-resize handle: it
# tracks its own width as a plain inline `style.width` on the sidebar's DOM
# node. A CSS rule (even `!important`) that also targets `width` sits in the
# DOM permanently once injected and continuously overrides that inline style
# at the browser level -- not just "on the next Python rerun" -- so it blocks
# dragging in real time for as long as it's present. So instead of fighting
# the sidebar with CSS, we set its inline `style.width` directly via a tiny
# one-shot script (same mechanism the drag handle itself uses). Once that
# single JS write happens, there's nothing left overriding the property, so
# the user's very next drag (which does its own plain `style.width` write)
# just wins outright -- no cascade fight, no rerun needed to "let go".
#
# We apply that forced width exactly once per *transition* between the wide
# (setup) and narrow (results) phases -- tracked via `_sidebar_state_applied`
# in session_state -- rather than on every single rerun. Re-forcing it on
# every rerun (the previous behavior) made any manual drag during the setup
# phase feel like it "snapped back": the instant you touched any other
# widget (which triggers a rerun), the width got reset out from under you.
# Applying it once per transition means a manual drag sticks across reruns
# in both phases, and only resets if the phase itself actually changes.
#
# The wide phase's *initial* width is computed in JS, not a fixed fraction
# of the screen: it measures the OpenFormula logo's natural (unwrapped)
# width, then sets the sidebar as wide as possible while still leaving the
# main area just big enough for that logo to render on one line.
#
# Separately: Streamlit's own resizable-sidebar wrapper ships with a
# hardcoded `max-width: 600px` (and `min-width: 200px`), which caps *any*
# drag at 600px no matter what. That's a plain (non-`!important`) rule, so
# a small always-on override just loosens the ceiling -- it doesn't touch
# `width` itself, so it never fights a drag.
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] { max-width: 95vw !important; }
    </style>
    """,
    unsafe_allow_html=True,
)
_sidebar_state = "narrow" if "results_df" in st.session_state else "wide"
_state_already_applied = st.session_state.get("_sidebar_state_applied") == _sidebar_state
if not _state_already_applied:
    st.session_state["_sidebar_state_applied"] = _sidebar_state
    if _sidebar_state == "narrow":
        _apply_width_js = "sb.style.width = '21rem';"
    else:
        # Clone the logo badge off-screen with no width constraint to
        # measure its true single-line width, then size the main area to
        # fit exactly that (plus its own padding and a small safety
        # buffer), handing every other pixel of the viewport to the
        # sidebar. Falls back to a fixed 75vw if the logo isn't found for
        # any reason (e.g. markup changes elsewhere).
        _apply_width_js = """
            var doc = window.parent.document;
            var badge = doc.querySelector('.openformula-badge');
            var mainBlock = doc.querySelector('[data-testid="stMainBlockContainer"]');
            var targetWidth;
            if (badge && mainBlock) {
                var clone = badge.cloneNode(true);
                clone.style.position = 'fixed';
                clone.style.visibility = 'hidden';
                clone.style.left = '-9999px';
                clone.style.top = '0';
                clone.style.width = 'auto';
                clone.style.maxWidth = 'none';
                clone.style.whiteSpace = 'nowrap';
                var titleClone = clone.querySelector('.openformula-title');
                if (titleClone) { titleClone.style.whiteSpace = 'nowrap'; }
                doc.body.appendChild(clone);
                var naturalWidth = clone.scrollWidth;
                doc.body.removeChild(clone);
                var cs = window.parent.getComputedStyle(mainBlock);
                var padLeft = parseFloat(cs.paddingLeft) || 0;
                var padRight = parseFloat(cs.paddingRight) || 0;
                var neededMainWidth = naturalWidth + padLeft + padRight + 24;
                var viewportWidth = window.parent.innerWidth;
                targetWidth = viewportWidth - neededMainWidth;
                var minWidth = 320;
                var maxWidth = viewportWidth * 0.95;
                targetWidth = Math.max(minWidth, Math.min(maxWidth, targetWidth));
                targetWidth = targetWidth + 'px';
            } else {
                targetWidth = '75vw';
            }
            sb.style.width = targetWidth;
        """
    st.iframe(
        f"""
        <script>
        (function() {{
            const sb = window.parent.document.querySelector('section[data-testid="stSidebar"]');
            if (!sb) return;
            function applyWidth() {{
                {_apply_width_js}
            }}
            if (window.parent.document.fonts && window.parent.document.fonts.ready) {{
                window.parent.document.fonts.ready.then(applyWidth);
            }} else {{
                applyWidth();
            }}
        }})();
        </script>
        <!-- {_sidebar_state} {time.time()} -->
        """,
        height=1,
    )


def gep_note(body_html):
    """A small rounded call-out box for explanatory blurbs (e.g. the text
    under each top-term plot), styled in a serif font instead of plain
    low-contrast caption text. `body_html` may contain simple tags like
    <strong>/<em>/<span class="gep-label"> -- it is NOT escaped, so only
    pass text built from known-safe formula/label fragments."""
    st.markdown(f'<div class="gep-note">{body_html}</div>', unsafe_allow_html=True)


def _combine_formula_str(physics_str, correction_str, combine_mode):
    """Build a human-readable combined-formula string consistent with
    combine_physics(): "physics + correction" (additive) or
    "physics × (1 + correction)" (multiplicative). `physics_str` may be
    falsy (no physics formula was used), in which case just the correction
    is returned, mirroring how combine_physics() treats a zero baseline."""
    if not physics_str:
        # No physics formula -> physics(X) is all zeros. Additively that
        # reduces to just the correction; multiplicatively 0*(1+correction)
        # is always 0 (the degenerate case the sidebar already warns about).
        return "0" if combine_mode == "mult" else correction_str
    if combine_mode == "mult":
        return f"{physics_str} × (1 + {correction_str})"
    return f"{physics_str} + {correction_str}"


def gep_equation(label_html, equation_text):
    """A slightly larger, greener call-out box for a generated equation,
    with the equation itself set in the same serif face (not code font)
    so it reads more like typeset math."""
    st.markdown(
        f'<div class="gep-equation"><span class="gep-label">{label_html}</span>{equation_text}</div>',
        unsafe_allow_html=True,
    )


def _label_beside_input(text):
    """Render a widget's prompt text vertically centered next to where its
    input box will sit (used to pair st.number_input with its label in a
    columns([label, input]) row, since number_input has no built-in
    'label on the same line' option)."""
    st.markdown(
        f'<div style="padding-top:0.55rem; font-size:1rem;">{text}</div>',
        unsafe_allow_html=True,
    )


def _regression_stats(y_true, y_pred):
    """Compact dict of standard regression/conservatism stats, shared by the
    live metrics elsewhere in the app and by the downloadable report below."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "MAE": float(np.mean(np.abs(y_true - y_pred))),
        "pct_conservative": 100 * float(np.mean(y_pred <= y_true)),
        "mean_error": float(np.mean(y_pred - y_true)),
    }


def _fig_to_base64(fig):
    """Render a matplotlib figure to a base64-encoded PNG string, for
    embedding directly in a self-contained HTML report (no separate image
    files to keep track of, and nothing else to attach when sharing it)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


_BINOP_PREC = {ast.Add: 1, ast.Sub: 1, ast.Mult: 2}
_BINOP_SYMBOL = {ast.Add: " + ", ast.Sub: " − ", ast.Mult: "·"}


def _fmt_num(value):
    """Format a numeric AST constant for display: ints as-is, floats to a
    handful of significant figures with no ugly trailing digits."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    return f"{value:.6g}"


def _expr_to_html(node, parent_prec=0):
    """Recursively render a parsed Python expression AST as typeset HTML
    math, for the report only: '/' becomes a stacked fraction, '**' becomes
    a superscript exponent, and sqrt(...) becomes a radical with an
    overline -- so equations read like textbook math instead of raw ASCII
    ('a/b', 'a**2') the way they're shown live in the app's own widgets."""
    if isinstance(node, ast.Expression):
        return _expr_to_html(node.body)

    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Div):
            num = _expr_to_html(node.left)
            den = _expr_to_html(node.right)
            return f'<span class="frac"><span class="num">{num}</span><span class="den">{den}</span></span>'
        if isinstance(node.op, ast.Pow):
            base = _expr_to_html(node.left, parent_prec=3)
            exp = _expr_to_html(node.right)
            return f'{base}<sup>{exp}</sup>'

        op_type = type(node.op)
        prec = _BINOP_PREC.get(op_type, 2)
        sym = _BINOP_SYMBOL.get(op_type, "?")
        left_html = _expr_to_html(node.left, parent_prec=prec)
        # The right side of a subtraction (or anything lower-precedence)
        # needs its own parens so 'a - (b + c)' isn't flattened into the
        # wrong-looking 'a − b + c'.
        right_is_binop = isinstance(node.right, ast.BinOp) and not isinstance(node.right.op, (ast.Div, ast.Pow))
        right_prec = _BINOP_PREC.get(type(node.right.op), 2) if right_is_binop else 99
        right_needs_parens = right_is_binop and (
            right_prec < prec or (isinstance(node.op, ast.Sub) and right_prec == prec)
        )
        right_html = _expr_to_html(node.right, parent_prec=0 if right_needs_parens else prec)
        if right_needs_parens:
            right_html = f'({right_html})'
        combined = f'{left_html}{sym}{right_html}'
        if prec < parent_prec:
            combined = f'({combined})'
        return combined

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return f'−{_expr_to_html(node.operand, parent_prec=3)}'
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _expr_to_html(node.operand, parent_prec=parent_prec)

    if isinstance(node, ast.Call):
        fname = node.func.id if isinstance(node.func, ast.Name) else _expr_to_html(node.func)
        args_html = ", ".join(_expr_to_html(a) for a in node.args)
        if fname == "sqrt" and len(node.args) == 1:
            return f'<span class="sqrt">&radic;<span class="radicand">{_expr_to_html(node.args[0])}</span></span>'
        return f'{fname}({args_html})'

    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        return _fmt_num(node.value)

    # Anything unexpected: fall back to plain source rather than raising, so
    # a formula the app itself already validated can never break the report.
    try:
        return ast.unparse(node)
    except Exception:
        return "?"


def _equation_to_html(expr_str):
    """Convert an equation string (physics formula, or the app's own
    generated combo/fitted/custom equation, which uses '×' for readability)
    into typeset HTML math (stacked fractions, radicals, superscripts) via
    _expr_to_html -- falling back to the plain string if it doesn't parse as
    an expression. Used both for the downloadable report and for the live
    equation call-out boxes (gep_equation/gep_note), which render with
    unsafe_allow_html=True and include matching CSS for these classes."""
    if not expr_str:
        return expr_str
    py_expr = expr_str.replace("×", "*").replace("−", "-")
    try:
        tree = ast.parse(py_expr, mode="eval")
        return _expr_to_html(tree)
    except Exception:
        return expr_str


# Real superscript digits (and minus sign), for plain-Unicode math display in
# contexts that can't render HTML tags at all -- dataframe cells and Plotly
# chart labels/hover text. No stacked-fraction equivalent is attempted here
# (that needs two lines, which none of those surfaces support); '/' is left
# as-is. Full HTML formatting (_equation_to_html above) is used instead
# wherever the surrounding widget already renders raw HTML.
_SUPERSCRIPT_TRANS = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _pretty_term_text(sig):
    """Plain-Unicode rendering of a (var, exponent) term signature -- e.g.
    [("fc", -1), ("bw", 1)] -> "fc⁻¹ · bw" -- for the ranked-term tables and
    the interactions pie chart/hover, mirroring parse_terms.signature_label's
    "fc^-1 * bw" but with real superscripts and a middle dot."""
    if not sig:
        return "(constant)"
    parts = []
    for var, exp in sig:
        if exp == 1:
            parts.append(var)
        else:
            exp_str = str(int(exp)) if float(exp) == int(exp) else str(exp)
            parts.append(f"{var}{exp_str.translate(_SUPERSCRIPT_TRANS)}")
    return " · ".join(parts)


def _prettify_ascii_expr(s):
    """Light cosmetic cleanup of a raw ASCII expression string (e.g. the
    'example' column's 'physics + term' text, which comes from sympy/
    term_expr_string) for the same HTML-free contexts as _pretty_term_text:
    integer '**n' powers become real superscripts, '*' becomes '·'. '/' and
    sqrt(...) calls are left as-is."""
    if not isinstance(s, str):
        return s
    s = re.sub(r"\*\*(-?\d+)(?!\.)", lambda m: m.group(1).translate(_SUPERSCRIPT_TRANS), s)
    s = s.replace("**", "^")  # fallback for any non-integer exponent left over
    s = s.replace("*", " · ")
    return s


def _build_report_html(physics_formula, known_formula_name, combo_formula, custom_equation,
                        phys_stats, fit_stats, plot_b64, top_terms_info, term_plots_info,
                        data_table, target_name, dataset_label, feature_names, hyperparams):
    """Assemble the full downloadable HTML report string: a title/timestamp
    header (with the dataset name and feature space), the physics formula
    (with its known-formula name if one was picked), the original unfitted
    physics+top-terms equation, the fitted/custom equation, a physics-vs-
    fitted stats comparison table, the embedded true-vs-predicted comparison
    plot, a top-terms frequency table, one plot + full data table per
    individual top term on its own (physics + just that one term, not
    combined with the others), the full actual/predicted data table for
    the combined fitted equation, and -- last -- the hyperparameters used
    for the search (both the ones exposed in the sidebar and the fixed,
    non-configurable GEP settings baked into gep_core.run_single) --
    everything a user needs to save, hand off, and reproduce in one file."""
    stats_rows = "".join(
        f"<tr><td>{label}</td><td>{phys_stats[key]:{fmt}}</td><td>{fit_stats[key]:{fmt}}</td></tr>"
        for label, key, fmt in [
            ("R²", "R2", ".4f"),
            ("RMSE", "RMSE", ".4g"),
            ("MAE", "MAE", ".4g"),
            ("% conservative (pred ≤ actual)", "pct_conservative", ".1f"),
            ("Mean error (pred − actual)", "mean_error", ".4g"),
        ]
    )
    terms_rows = "".join(
        f"<tr><td>{t['term']}</td><td>{t['n_solutions']}</td><td>{t['pct_of_solutions']}</td>"
        f"<td>{t['description']}</td></tr>"
        for t in top_terms_info
    ) or "<tr><td colspan='4'><em>(none)</em></td></tr>"

    data_table_html = data_table.to_html(index=False, float_format=lambda x: f"{x:.4g}")

    term_sections_html = "".join(
        f"""
  <h3>Physics + {t['term']}</h3>
  <p><em>{t['description']}</em></p>
  <table>
    <tr><th>R²</th><th>RMSE</th><th>MAE</th><th>% conservative (pred ≤ actual)</th><th>Mean error (pred − actual)</th></tr>
    <tr><td>{t['stats']['R2']:.4f}</td><td>{t['stats']['RMSE']:.4g}</td><td>{t['stats']['MAE']:.4g}</td>
        <td>{t['stats']['pct_conservative']:.1f}</td><td>{t['stats']['mean_error']:.4g}</td></tr>
  </table>
  <img src="data:image/png;base64,{t['plot_b64']}" alt="Physics + {t['term']} true vs predicted plot">
  <div class="data-table-wrap">
    {t['data_table_html']}
  </div>
"""
        for t in term_plots_info
    ) or "<p><em>(none)</em></p>"

    physics_html = _equation_to_html(physics_formula) if physics_formula else "(none used)"
    combo_html = _equation_to_html(combo_formula) if combo_formula else "(no top terms available)"
    custom_html = _equation_to_html(custom_equation)
    physics_name_html = (
        f'<strong style="font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; '
        f'font-style: normal; font-size: 0.8rem; display:block; margin-bottom:0.3rem; color:#2e7d32;">'
        f"{known_formula_name}</strong>"
        if known_formula_name else ""
    )
    feature_list_html = ", ".join(f"<code>{f}</code>" for f in (feature_names or []))

    def _fmt_list(xs, fmt="{}"):
        return ", ".join(fmt.format(x) for x in xs) if xs else "(none)"

    hp = hyperparams or {}
    combine_mode_label = (
        "Multiplicative -- physics × (1 + GEP(X))" if hp.get("combine_mode") == "mult"
        else "Additive -- physics + GEP(X)"
    )
    hyperparams_rows = "".join(
        f"<tr><td>{label}</td><td>{value}</td></tr>"
        for label, value in [
            ("Head length (max operators per gene)", hp.get("head_length", "—")),
            ("Number of genes", hp.get("n_genes", "—")),
            ("Generations per run", hp.get("generations", "—")),
            ("Population size per run", hp.get("pop_size", "—")),
            ("Tournament size", hp.get("tournsize", "—")),
            ("Hall-of-Fame size (solutions recorded per run)", hp.get("champs", "—")),
            ("Validation fraction of train+val pool", hp.get("val_frac", "—")),
            ("Combination mode", combine_mode_label),
            (
                "Training-data-availability levels (ptrain, fraction of whole dataset)",
                _fmt_list(hp.get("ptrains", []), "{:g}"),
            ),
            ("Number of random seeds", len(hp.get("seeds", []))),
            ("Random seeds", _fmt_list(hp.get("seeds", []))),
            (
                "Total GEP runs (seeds × ptrain levels)",
                hp.get("total_runs", len(hp.get("seeds", [])) * len(hp.get("ptrains", []))),
            ),
        ]
    )
    fixed_settings_rows = "".join(
        f"<tr><td>{label}</td><td>{value}</td></tr>"
        for label, value in [
            ("Random numerical constant (RNC) array length per gene", "10"),
            ("Random numerical constant (RNC) range", "&minus;10 to 10, integer"),
            ("Elitism (unconditionally-kept top individuals per generation)", "1"),
            ("Mutation -- uniform point mutation (per-element probability)", "0.05"),
            ("Mutation -- gene inversion probability", "0.1"),
            ("Mutation -- IS transposition probability", "0.1"),
            ("Mutation -- RIS transposition probability", "0.1"),
            ("Mutation -- gene transposition probability", "0.1"),
            ("Crossover -- one-point probability", "0.3"),
            ("Crossover -- two-point probability", "0.2"),
            ("Crossover -- whole-gene probability", "0.1"),
        ]
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>OpenFormula &mdash; GEP Feature Discovery Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@700;800&display=swap');
  body {{ font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
  .of-badge {{ display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.45rem 1.2rem; margin: 0 0 0.7rem 0; background: #0E2A47; border-radius: 22px 8px 22px 8px; width: fit-content; }}
  .of-dna {{ height: 1.7rem; width: auto; flex-shrink: 0; display: block; }}
  .of-title {{ font-family: 'Poppins', -apple-system, Segoe UI, Helvetica, Arial, sans-serif; font-weight: 800; font-size: 1.7rem; letter-spacing: -0.01em; line-height: 1; }}
  .of-open {{ color: #FFFFFF; }}
  .of-formula {{ color: #4FC3F7; }}
  h1 {{ font-size: 1.5rem; margin: 0 0 0.2rem; }}
  h2 {{ font-size: 1.15rem; margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; }}
  .timestamp {{ color: #666; font-size: 0.9rem; margin-bottom: 1.5rem; }}
  .equation-box {{ background: #f4f7f5; border-left: 4px solid #2e7d32; padding: 0.9rem 1rem; margin: 0.6rem 0; font-family: Georgia, serif; font-size: 1.15rem; line-height: 2.1; }}
  .eqn {{ font-style: italic; }}
  .frac {{ display: inline-flex; flex-direction: column; vertical-align: middle; text-align: center; margin: 0 0.2em; font-style: normal; }}
  .frac .num {{ display: block; padding: 0 0.25em 0.1em; border-bottom: 1.4px solid #1a1a1a; }}
  .frac .den {{ display: block; padding: 0.1em 0.25em 0; }}
  .sqrt {{ white-space: nowrap; font-style: normal; }}
  .sqrt .radicand {{ border-top: 1.4px solid #1a1a1a; padding: 0 0.2em; margin-left: 0.05em; }}
  sup {{ font-size: 0.72em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 0.8rem 0; font-size: 0.9rem; }}
  th, td {{ border: 1px solid #ddd; padding: 0.4rem 0.6rem; text-align: left; }}
  th {{ background: #f0f0f0; }}
  img {{ max-width: 100%; margin: 0.8rem 0; }}
  .data-table-wrap {{ max-height: 500px; overflow-y: auto; border: 1px solid #ddd; }}
</style>
</head>
<body>
  <div class="of-badge">
    <svg class="of-dna" viewBox="0 0 28 52" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M5,2 C5,10 23,10 23,17 C23,24 5,24 5,32 C5,40 23,40 23,48" stroke="#FFFFFF" stroke-width="2.5" stroke-linecap="round"/>
      <path d="M23,2 C23,10 5,10 5,17 C5,24 23,24 23,32 C23,40 5,40 5,48" stroke="#FFFFFF" stroke-width="2.5" stroke-linecap="round"/>
      <line x1="5" y1="2" x2="23" y2="2" stroke="#FFFFFF" stroke-width="1.6" stroke-linecap="round" opacity="0.55"/>
      <line x1="5" y1="17" x2="23" y2="17" stroke="#FFFFFF" stroke-width="1.6" stroke-linecap="round" opacity="0.55"/>
      <line x1="5" y1="32" x2="23" y2="32" stroke="#FFFFFF" stroke-width="1.6" stroke-linecap="round" opacity="0.55"/>
      <line x1="5" y1="48" x2="23" y2="48" stroke="#FFFFFF" stroke-width="1.6" stroke-linecap="round" opacity="0.55"/>
    </svg>
    <span class="of-title"><span class="of-open">Open</span><span class="of-formula">ForMuLa</span></span>
  </div>
  <h1>GEP Feature Discovery Report</h1>
  <div class="timestamp">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &middot; dataset: <code>{dataset_label}</code> &middot; target: <code>{target_name}</code></div>
  <div class="timestamp">Feature space ({len(feature_names or [])}): {feature_list_html}</div>

  <h2>Formulas</h2>
  <div class="equation-box"><strong style="font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; font-style: normal; font-size: 0.85rem; display:block; margin-bottom:0.3rem;">Physics formula</strong>{physics_name_html}<span class="eqn">{physics_html}</span></div>
  <div class="equation-box"><strong style="font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; font-style: normal; font-size: 0.85rem; display:block; margin-bottom:0.3rem;">Original equation (physics + top terms, unfitted)</strong><span class="eqn">{combo_html}</span></div>
  <div class="equation-box"><strong style="font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; font-style: normal; font-size: 0.85rem; display:block; margin-bottom:0.3rem;">Fitted / custom equation</strong><span class="eqn">{custom_html}</span></div>

  <h2>Physics-only vs. fitted-equation performance</h2>
  <table>
    <tr><th>Statistic</th><th>Physics only</th><th>Fitted equation</th></tr>
    {stats_rows}
  </table>

  <h2>True vs. predicted comparison</h2>
  <img src="data:image/png;base64,{plot_b64}" alt="True vs predicted comparison plot">

  <h2>Top recurring terms</h2>
  <table>
    <tr><th>Term</th><th># solutions</th><th>% of solutions</th><th>Description</th></tr>
    {terms_rows}
  </table>

  <h2>Individual feature/term performance (physics + one term at a time)</h2>
  <p>Each top term shown on its own, combined only with the physics baseline (not with the other terms) -- so its individual contribution can be judged in isolation, separate from the combined fitted equation above.</p>
  {term_sections_html}

  <h2>Full data table (actual vs. predicted)</h2>
  <div class="data-table-wrap">
    {data_table_html}
  </div>

  <h2>Hyperparameters</h2>
  <p>Everything that controlled this search: the settings exposed in the sidebar (frozen at the moment "Run search" was clicked, so later widget tweaks can't desync this report from what actually ran), plus the fixed GEP settings that aren't user-configurable.</p>
  <table>
    <tr><th>Setting</th><th>Value</th></tr>
    {hyperparams_rows}
  </table>
  <p style="margin-top:1rem;"><em>Fixed (non-configurable) GEP settings:</em></p>
  <table>
    <tr><th>Setting</th><th>Value</th></tr>
    {fixed_settings_rows}
  </table>
</body>
</html>
"""


with st.sidebar:
    # --------------------------------------------------------------------------
    # 1. Dataset
    # --------------------------------------------------------------------------
    @st.cache_data(show_spinner=False)
    def _load_dataset_bytes(file_bytes, filename):
        return load_dataset(io.BytesIO(file_bytes), filename_hint=filename)


    # Bundled demo datasets. `rename` maps a demo's raw column headers to
    # friendlier/formula-matching names (e.g. Yeh's terse "C"/"W" -> "cement"/
    # "water", so the built-in Bolomey/Feret formulas below can reference
    # them directly); pass {} to keep a dataset's original headers as-is.
    # `default_known_formula` is the "Known formula" dropdown entry (from
    # KNOWN_FORMULAS, defined in section 3 below) to pre-select for that demo.
    DEMO_DATASETS = {
        "Beam shear (ACI 318)": {
            "file": "beam_v1_Deep.xlsx",
            "rename": {},
            "info": (
                "Using the bundled beam-shear dataset (Vu = beam shear capacity). "
                "Try target = last column. A known ACI 318 shear formula has been pre-selected "
                "in the \"Known formula\" dropdown below -- pick a different one, edit the box "
                "it fills in, or clear it if you rename columns below or want GEP to fit the "
                "target directly."
            ),
            "default_known_formula": "ACI 318 — Simplified one-way shear (Vc)",
        },
        "Concrete mix strength (Yeh 1998)": {
            "file": "concrete_mix_strength.xlsx",
            "rename": {
                "C": "cement", "BFS": "slag", "FLY": "flyash", "W": "water",
                "SP": "superplasticizer", "CA": "coarse_agg", "FA": "fine_agg",
                "T": "age", "Cs": "strength",
            },
            "info": (
                "Using the bundled concrete mix-design dataset (Yeh, 1998): predicting "
                "compressive strength from mix proportions (cement, water, coarse/fine "
                "aggregate, superplasticizer, slag, fly ash) and curing age. Try target = "
                "last column (`strength`). Bolomey's Law -- a classical mix-design strength "
                "law, not an official ACI equation -- has been pre-selected in the \"Known "
                "formula\" dropdown below; Feret's Law needs an `air` column this dataset "
                "doesn't have, so it won't work here."
            ),
            "default_known_formula": "Bolomey's Law — cement/water ratio (concrete strength)",
        },
        "Lap-splice bond strength (ACI 408 / Orangun-Jirsa-Breen)": {
            "file": "lap_splice_bond_strength.xlsx",
            "rename": {"fsmc ": "fsmc"},
            "info": (
                "Using the bundled lap-splice/bond-strength database (the classic ACI "
                "Committee 408 test set behind the Orangun-Jirsa-Breen bond-stress equation, "
                "which underlies ACI 318's development-length provisions). Columns: `ls`=splice "
                "length, `Asp`=splice bond area, `db`=bar diameter, `Ab`=bar area, `Nb`=number of "
                "bars spliced, `cso`=clear side cover, `csi`=half the clear spacing between bars, "
                "`cb`=bottom cover, `dtr`=stirrup bar diameter, `At`=single stirrup-leg area, "
                "`Ns`=number of stirrups, `Nl`=number of legs, `ΣAt`=total transverse steel area "
                "crossing the splice (= At x Ns x Nl, provided directly), `fcm`=mean concrete "
                "strength (psi), `fyt`=stirrup yield strength (ksi; 0 where no stirrups are "
                "present). Try target = last column (`fsmc`, the measured splice bar stress in "
                "ksi -- NOT normalized by sqrt(f'c), unlike the earlier version of this dataset). "
                "The Waldron and Skillen force-based splice model is pre-selected in the "
                "\"Known formula\" dropdown below -- not a ready-to-use fit on this data as-is, "
                "which is exactly what makes it a good showcase for letting GEP discover/correct "
                "the rest."
            ),
            "default_known_formula": "Waldron and Skillen",
        },
        "Corroded RC beam flexural capacity (ACI 318, corroded)": {
            "file": "corroded_rc_flexure_piml.xlsx",
            "rename": {},
            "info": (
                "646-row database of reinforced-concrete beams tested in flexure after "
                "artificial (impressed current, `CorrMeth`='IC'), environmentally-induced "
                "(`CorrMeth`='EI'), or no (`CorrMeth`='C') corrosion exposure. Target = last "
                "column, `Mmax` (experimental ultimate moment capacity, kNm). Two engineered "
                "0/1 columns, `is_IC` and `is_high` (eta_m >= 27.5%), are included alongside "
                "the raw geometry/material columns -- they select which branch of the corroded-"
                "strength formulas would apply per row, since the physics-formula box can't "
                "express an if/else or categorical test directly. The built-in corroded-flexure "
                "formula that uses them isn't currently offered in the \"Known formula\" dropdown "
                "below (temporarily trimmed down to a smaller default set) -- type a formula in "
                "manually if you want one, or leave it blank and let GEP fit the target directly. "
                "Several other columns (`CS`, `TestType`, `RebarCfg`, `BarType`, `Anchor`, "
                "`StirType`, `Cement`, `TestMeth`, `CorrMeth` itself) are plain categorical/text "
                "fields -- one-hot encode or drop them below as you prefer."
            ),
            "default_known_formula": "(none / custom)",  # matches FORMULA_NONE below
        },
        "Corrosion processed": {
            "file": "corrosion_processed.xlsx",
            "rename": {},
            "info": (
                "669-row reinforced-concrete beam flexure database (combined/cleaned from "
                "Azad et al.-style corrosion test data), raw column headers left as-is -- "
                "non-identifier-safe ones (e.g. `W (mm)`, `f'c (MPa)`, `d0 = db,t (mm)`) get "
                "auto-sanitized below to `W_mm`, `f_c_MPa`, `d0_db_t_mm`, etc. Target = last "
                "column, `Mmax_exp_kNm` (experimental ultimate moment capacity, kNm). Two "
                "beams from the original 671-row source (32mm tension bars, the only 32mm "
                "beams with any corrosion exposure, over-reinforced at a 5.36% tension ratio) "
                "have already been excluded -- they're outside the calibration range of the "
                "built-in corrosion-reduction formula below and break the tension-steel-yields "
                "assumption it relies on; see that formula's help text for details. The `Set` "
                "column (ηm<10% / ηm>=10%) is categorical -- one-hot encode or drop it below. "
                "A built-in `K x Mth` corrosion baseline (simplified nominal capacity reduced "
                "by an empirical corrosion factor) is pre-selected in the \"Known formula\" "
                "dropdown below."
            ),
            "default_known_formula": "ACI 318 — Corrosion-reduced flexural capacity (K·Mth)",
        },
        "LC3_Processed": {
            "file": "lc3_processed.xlsx",
            "rename": {},
            "info": (
                "173-row LC3 (limestone calcined clay cement) mortar compressive-strength "
                "database, no missing values. Target = last column, "
                "`compressive_strength_mpa`. Mix-design columns include `wb_ratio` "
                "(water/binder), `clinker_content`, pozzolan/SO3/CaCO3 content (`added_*` "
                "vs. `total_*`), `sp_kg_per_kg_binder` (superplasticizer dosage), "
                "`fine_agg_binder_ratio`, and `mk_ls_ratio` (metakaolin/limestone ratio). "
                "Raw-material chemistry/particle-size columns are prefixed by component: "
                "`opc_*` (ordinary portland clinker oxide composition + `_d10/_d50/_d90` "
                "particle-size percentiles and `_span`), `ls_*` (limestone purity/oxides/"
                "particle size), `cc_*` (calcined clay purity/particle size/"
                "`sio2_al2o3_ratio`), and `lc3_*` (particle size of the blended LC3 binder "
                "itself). `sp_wr_present` is a 0/1 flag (25 of 173 mixes used a "
                "superplasticizer/water-reducer) -- already numeric, treat as a feature "
                "like any other or drop it. `age_days` is the age at test (1-91 days; "
                "mostly 1/3/7/28/90). All columns are already numeric identifier-safe "
                "names, so no renaming/sanitizing is needed. Pre-selected known formula is "
                "a static Abrams/fib strength-age baseline, source-calibrated on external "
                "conventional-concrete data (not LC3) -- see its own entry below for why -- "
                "pick \"Multiplicative\" combine mode to pair with it, or clear it back to "
                "\"(none / custom)\" to have GEP fit the target directly instead."
            ),
            "default_known_formula": "Abrams/fib strength-age law — source-calibrated static baseline (LC3)",
        },
        "Punching shear — ML features (ACI 318 / CSA A23.3)": {
            "file": "punching_ml_features.csv",
            "rename": {},
            "drop": ["target_V"],
            "info": (
                "436-row two-way (punching) shear database for slab-column connections, "
                "with the variables needed for both the ACI 318-19/-25 and CSA A23.3-19/-24 "
                "punching-shear provisions. Target = last column, `target_V_over_bod` -- the "
                "shear stress vc = V/(bo*d) in MPa (N/mm^2), already unit-consistent with "
                "`fc`/`d`/`bo` below, so neither known formula needs a conversion factor. "
                "`target_V` (total shear force, kN) is dropped entirely: it's deterministically "
                "target_V_over_bod*bo*d/1000, so keeping it as a feature would leak the "
                "target almost perfectly. `connection` (interior/edge/corner) and "
                "`column_shape` (square/rectangular/circular) are categorical text columns -- "
                "you'll be prompted to one-hot encode or drop them before they're usable as "
                "features. Geometry/material columns: `c1`, `c2` (column plan dimensions, "
                "mm), `d` (slab effective depth, mm), `fc` (concrete strength, MPa), "
                "`lambda_conc` (lightweight-concrete factor, 1.0 = normalweight), `bo` "
                "(critical shear perimeter, mm), `beta_c` (column long/short side ratio), "
                "`alpha_s_ACI`/`alpha_s_CSA` (column-location constants: 40/30/20 and 4/3/2 "
                "for interior/edge/corner respectively), `gamma_v_y`/`gamma_v_x` (fraction of "
                "unbalanced moment transferred by shear, each axis), `load_angle_y`/"
                "`load_angle_x`, `k_ecc` (eccentric-shear stress-concentration factor -- both "
                "known formulas below divide the code's nominal vc by this to get the design "
                "value), `rho_avg` (average flexural reinforcement ratio, %), `fibre_vol`/"
                "`fibre_mass` (steel-fibre dosage, 0 for non-fibre-reinforced slabs). All "
                "columns are already numeric identifier-safe names (aside from the two "
                "categorical ones above), so no renaming/sanitizing is needed. Two known "
                "formulas below implement the full ACI 318-19/-25 and CSA A23.3-19/-24 code "
                "equations for vc, lambdas and all -- ACI is pre-selected; switch to the CSA "
                "entry to compare the two codes directly on the same data."
            ),
            "default_known_formula": "ACI 318-19/-25 — Two-way shear stress vc (λs, βc, αs, k_ecc)",
        },
    }
    demo_options = list(DEMO_DATASETS)

    st.header("1. Dataset")
    source = st.radio(
        "Data source", ["Upload a file", "Use a built-in demo dataset"],
        horizontal=True,
    )

    df_raw = None
    demo_choice = None
    if source == "Upload a file":
        uploaded = st.file_uploader("CSV or Excel file", type=["csv", "tsv", "xlsx", "xls"])
        if uploaded is not None:
            df_raw = _load_dataset_bytes(uploaded.getvalue(), uploaded.name)
        dataset_label = uploaded.name if uploaded is not None else "(no file uploaded)"
    else:
        demo_choice = st.selectbox("Demo dataset", demo_options, key="demo_dataset_select")
        demo = DEMO_DATASETS[demo_choice]
        try:
            with open(demo["file"], "rb") as f:
                df_raw = _load_dataset_bytes(f.read(), demo["file"])
            if demo["rename"]:
                df_raw = df_raw.rename(columns=demo["rename"])
            if demo.get("drop"):
                df_raw = df_raw.drop(columns=[c for c in demo["drop"] if c in df_raw.columns])
            st.info(demo["info"])
        except FileNotFoundError:
            st.error(f"Demo dataset not found next to app.py ({demo['file']}).")
        dataset_label = demo_choice

    if df_raw is None:
        st.stop()

    st.write(f"Loaded **{len(df_raw)}** rows, **{len(df_raw.columns)}** columns (duplicates dropped).")
    st.dataframe(df_raw.head(10), width="stretch")

    numeric_cols = [c for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c])]
    non_numeric = [c for c in df_raw.columns if c not in numeric_cols]

    if non_numeric:
        st.warning(f"Non-numeric column(s) found: {non_numeric}. GEP needs numeric inputs -- "
                   "choose how to handle each one below.")
        encode_cols, drop_cols = [], []
        for col in non_numeric:
            n_unique = df_raw[col].nunique(dropna=True)
            default_idx = 0 if n_unique <= 15 else 1  # default to drop for high-cardinality columns
            action = st.selectbox(
                f"`{col}` ({n_unique} unique value{'s' if n_unique != 1 else ''})",
                ["One-hot encode", "Drop this column"],
                index=default_idx, key=f"nonnum_action_{col}",
                help=(
                    f"One-hot encode adds {n_unique} new 0/1 indicator column(s), one per category, so GEP "
                    "can use it as a numeric feature. Drop removes it entirely. High-cardinality columns "
                    "(many unique values, e.g. free-text or IDs) default to drop since one-hot encoding them "
                    "would add a column per value."
                ),
            )
            (encode_cols if action == "One-hot encode" else drop_cols).append(col)

        # Build one-hot columns per encoded source column (kept separate, rather
        # than one combined get_dummies call, so each can be spliced back into
        # its own column's original position below).
        dummies_by_col = {}
        for col in encode_cols:
            dummies_by_col[col] = pd.get_dummies(df_raw[col], prefix=col, dtype=int)
        if encode_cols:
            st.caption(
                "One-hot encoded: " + "; ".join(
                    f"`{c}` -> {list(dummies_by_col[c].columns)}" for c in encode_cols
                )
            )
        if drop_cols:
            st.caption(f"Dropped: {drop_cols}")

        # Rebuild df_raw column-by-column in the ORIGINAL column order, splicing
        # each encoded column's dummies in at that same spot (instead of
        # appending them all at the end) -- otherwise, if the target column was
        # originally last, the dummies would push past it and one of them could
        # get auto-selected as the default target below.
        pieces = []
        for col in df_raw.columns:
            if col in drop_cols:
                continue
            elif col in encode_cols:
                pieces.append(dummies_by_col[col])
            else:
                pieces.append(df_raw[[col]])
        df_raw = pd.concat(pieces, axis=1)

        numeric_cols = [c for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c])]

    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns (1 target + 1 feature).")
        st.stop()

    # --------------------------------------------------------------------------
    # 2. Target & features
    # --------------------------------------------------------------------------
    st.header("2. Response & feature variables")
    col1, col2 = st.columns([1, 2])
    with col1:
        target_name = st.selectbox(
            "Response variable (variable to predict)", numeric_cols, index=len(numeric_cols) - 1,
        )
    with col2:
        feature_candidates = [c for c in numeric_cols if c != target_name]
        feature_names_raw = st.multiselect(
            "Feature variables", feature_candidates, default=feature_candidates,
        )
    if not feature_names_raw:
        st.error("Select at least one feature column.")
        st.stop()

    # --------------------------------------------------------------------------
    # Outlier detection (optional row removal) -- only shown if any are found.
    # Uses the standard-deviation (z-score) method, applied only to the target
    # / response column: flag rows where |value - mean| > n_std * std. Detection
    # is first checked at a default n_std; the widget to change n_std (and the
    # rest of this section's UI) is only shown if that default flags something.
    # --------------------------------------------------------------------------
    DEFAULT_OUTLIER_N_STD = 3.0
    target_mean = df_raw[target_name].mean()
    target_std = df_raw[target_name].std()


    def _outlier_mask_for(n_std):
        if not target_std or pd.isna(target_std):
            return pd.Series(False, index=df_raw.index)
        return (df_raw[target_name] - target_mean).abs() > n_std * target_std


    default_mask = _outlier_mask_for(DEFAULT_OUTLIER_N_STD)
    outlier_mask = pd.Series(False, index=df_raw.index)
    if int(default_mask.sum()) > 0:
        OUTLIER_N_STD_OPTIONS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.markdown("Outlier threshold: flag response values more than")
            n_std = st.selectbox(
                "n_std", OUTLIER_N_STD_OPTIONS,
                index=OUTLIER_N_STD_OPTIONS.index(DEFAULT_OUTLIER_N_STD),
                key="outlier_n_std", label_visibility="collapsed", width=90,
                help=f"Applied only to the response variable (`{target_name}`).",
            )
            st.markdown("standard deviations from the mean")
        outlier_mask = _outlier_mask_for(n_std)
        n_outliers = int(outlier_mask.sum())

        if n_outliers > 0:
            st.warning(
                f"{n_outliers} row(s) ({100 * n_outliers / len(df_raw):.1f}%) flagged as outliers in the "
                f"response variable `{target_name}` (more than {n_std:g} standard deviations from its mean)."
            )
            if st.checkbox(f"Drop these {n_outliers} outlier row(s) before analysis", value=False, key="drop_outliers"):
                df_raw = df_raw[~outlier_mask].reset_index(drop=True)
                # All flagged rows are now gone, so nothing left counts as an
                # outlier -- reset the mask to match the surviving rows so
                # downstream code (e.g. the results plots) doesn't try to
                # highlight rows that no longer exist.
                outlier_mask = pd.Series(False, index=df_raw.index)
                st.caption(f"Dropped {n_outliers} outlier row(s); {len(df_raw)} row(s) remain.")
        else:
            st.caption(f"No outliers in `{target_name}` at {n_std:g} standard deviations.")


    name_map = sanitize_feature_names(feature_names_raw)
    renamed = {k: v for k, v in name_map.items() if k != v}
    if renamed:
        st.caption(
            "Some column names aren't valid variable names, so they're mapped to safe "
            "names below. **Use the renamed variables in your physics formula.**"
        )
        st.dataframe(
            pd.DataFrame({"original variable": list(renamed.keys()), "renamed variable": list(renamed.values())}),
            hide_index=True, width="content",
        )
    feature_names = [name_map[c] for c in feature_names_raw]
    df = df_raw.rename(columns=name_map)


    # --------------------------------------------------------------------------
    # 3. Physical knowledge (optional)
    # --------------------------------------------------------------------------
    st.header("3. Physical knowledge (optional)")
    st.caption(
        "If you already know a closed-form equation that predicts (part of) the target from "
        "the features — a design code formula, a physical law, a simplified model — enter it "
        "here. GEP will then only need to discover what's *missing* from it. Leave blank to have "
        "GEP fit the target directly. Allowed operators: +, -, *, /, **, and sqrt/log/log10/log2/exp/abs/"
        "min/max/sin/cos/tan/power/pi, referencing the feature names above."
    )
    # A small built-in library of known formulas that can be used to pre-fill
    # the physics-formula box below: official ACI 318 design-code equations,
    # plus a couple of classical concrete mix-design strength laws (Bolomey,
    # Feret) that predate ACI but underlie its ACI 211.1 mix-selection
    # tables -- those two carry an empirical constant K that (like Abrams'
    # law) depends on local materials, so the value baked in here is just a
    # common textbook default, not a fixed code value; recalibrate it against
    # real data before trusting it. Each formula only references specific
    # feature names (e.g. `fc`, `bw`, `d`, `pw`, `bo`, `cement`, `water`) --
    # picking one whose names aren't present among this dataset's feature
    # columns will surface the normal "unknown name" error below, which is
    # expected (not every formula applies to every dataset).
    KNOWN_FORMULAS = {
        "ACI 318 — Simplified one-way shear (Vc)": {
            "formula": "0.17*sqrt(fc)*bw*d/1000",
            "purpose": (
                "Concrete's shear contribution for a nonprestressed beam without axial load "
                "(ACI 318 §22.5.5.1(a), the \"simplified\" method). SI units: f'c in MPa, "
                "bw & d in mm, result in kN."
            ),
        },
        "ACI 318 — One-way shear w/ stirrups (Vn = Vc + Vs)": {
            "formula": "(0.17*sqrt(fc)*bw*d + Av*fy*d/s) / 1000",
            "purpose": (
                "Total nominal one-way shear strength, adding the steel/stirrup contribution "
                "Vs = Av·fy·d/s (ACI 318 §22.5.10.5.3) on top of the simplified concrete "
                "contribution Vc (ACI 318 §22.5.5.1(a)) above, i.e. Vn = Vc + Vs. `fy`, `bw` & "
                "`d` reuse this dataset's own columns; `Av` (total stirrup leg area within "
                "spacing `s`) and `s` (stirrup spacing) are made-up placeholder names, since "
                "the bundled beam-shear demo is a without-stirrups database and has no such "
                "columns -- this formula will error out there, that's expected, and is meant "
                "for a dataset of stirrup-reinforced beams instead. SI units: f'c & fy in MPa; "
                "bw, d, s in mm; Av in mm²; result in kN."
            ),
        },
        "ACI 318 — Detailed one-way shear (Vc, w/ ρw & M/V)": {
            "formula": "min(0.16*sqrt(fc) + 17*(pw/100)*d/a_M_V, 0.29*sqrt(fc)) * bw * d / 1000",
            "purpose": (
                "Refines the simplified Vc using the longitudinal reinforcement ratio (pw, as a "
                "%) and the moment-to-shear ratio a=M/V, capped at 0.29·√f'c·bw·d "
                "(ACI 318-11/14 Eq. 11-5 / §22.5.5.1(b), SI units)."
            ),
        },
        "ACI 318-19 — Size-effect one-way shear (Vc)": {
            "formula": "0.66 * min(sqrt(2/(1+0.004*d)), 1) * (pw/100)**(1/3) * sqrt(fc) * bw * d / 1000",
            "purpose": (
                "Accounts for the reduced unit shear strength observed in larger, "
                "lightly-reinforced members without stirrups via the size-effect factor "
                "λs (ACI 318-19 Table 22.5.5.1.1, SI units)."
            ),
        },
        "ACI 318 — Two-way (punching) shear (Vc)": {
            "formula": "0.33*sqrt(fc)*bo*d/1000",
            "purpose": (
                "Concrete's punching-shear contribution around a column/slab critical section "
                "(ACI 318 Eq. 22.6.5.2, SI units). Needs a critical-perimeter column `bo` -- not "
                "present in every dataset (e.g. the bundled beam-shear demo), so this one will "
                "error out there; that's expected."
            ),
        },
        "Bolomey's Law — cement/water ratio (concrete strength)": {
            "formula": "20*(cement/water - 0.5)",
            "purpose": (
                "Classical mix-design strength law (predates ACI, but is the empirical basis "
                "behind ACI 211.1's water-cement-ratio selection tables): strength rises "
                "roughly linearly with the cement-to-water ratio by mass. K=20 (MPa) here is a "
                "common textbook default for average-quality cement/aggregate -- it's an "
                "empirical constant that should be recalibrated against your own materials, not "
                "a fixed code value. Needs `cement` & `water` columns -- not present in the "
                "bundled beam-shear demo."
            ),
        },
        "Feret's Law — cement/water/air volumes (concrete strength)": {
            "formula": "200*(cement/(cement+water+air))**2",
            "purpose": (
                "Classical mix-design strength law relating strength to the volume fraction of "
                "cement in the paste (cement, water, and entrapped/entrained air, as absolute "
                "volumes -- convert mass quantities using material densities first). K=200 (MPa) "
                "is a common textbook default; like Bolomey's K, it's empirical and "
                "materials-dependent, not a fixed code value. Needs `cement`, `water` & `air` "
                "columns -- not present in the bundled beam-shear demo."
            ),
        },
        "Orangun-Jirsa-Breen — bond stress / lap splice (u/sqrt(fc))": {
            "formula": "1.2 + 3*(cb/db) + 50*(db/ls)",
            "purpose": (
                "Classical (1977) empirical bond-stress equation behind ACI 318's development-"
                "length provisions: normalized bond stress at splice failure rises with cover "
                "relative to bar diameter (cb/db) and falls as splice length grows relative to "
                "bar diameter (db/ls). This is the *simplified* form only -- the full equation "
                "adds a confinement term +Ktr/db from transverse steel (Ktr = Atr*fyt/(500*s*n), "
                "where s is stirrup spacing), which isn't included here because spacing isn't a "
                "direct column in the bundled lap-splice demo (only stirrup count/area are). On "
                "that dataset this simplified form only reaches R^2 ~ 0.5 -- a real gap for GEP "
                "to help close, not a bug. US customary units: db, ls, cover in inches."
            ),
        },
        "Waldron and Skillen": {
            "formula": "(5*sqrt(fcm)*Asp + fyt*ΣAt) / (Ab*Nb) / 1000",
            "purpose": (
                "Computed splice-bar stress capacity (f_s,calc) as a sum of two force "
                "contributions, normalized by the total spliced-bar area: a concrete/bond term "
                "C*sqrt(f'cm)*Asp (Asp = splice bond area) plus a transverse-steel term "
                "fyt*ΣAt (ΣAt = total transverse steel area crossing the splice, = At*Ns*Nl -- "
                "provided directly as its own column), divided by Ab*Nb; the /1000 converts the "
                "psi-scale concrete term onto the same ksi scale as the `fyt` steel term and the "
                "`fsmc` target. C=5 is a placeholder, not a fitted value -- a true fitted C "
                "requires knowing each specimen's experimental behavior, which defeats the point "
                "of a predictive formula, so that's exactly the gap left for GEP to close. At "
                "C=5 against the bundled dataset's `fsmc` target: correlation is a real, moderate "
                "r~0.76, and it's on the conservative (under-predicting) side for ~90% of rows -- "
                "closer than it looks (R^2 still below 0), but not a ready-to-use fit as-is."
            ),
        },
        "ACI 408R-03 (Zuo & Darwin) — splice bar stress at bond failure (f_s,calc)": {
            "formula": (
                "fcm**0.25 * (1970*min(0.1*(max(min(csi+0.25, cso), cb)/min(min(csi+0.25, cso), cb))+0.9, 1.25) "
                "+ 62*((min(min(csi+0.25, cso), cb)+0.5*db)*min(0.1*(max(min(csi+0.25, cso), cb)/min(min(csi+0.25, cso), cb))+0.9, 1.25)"
                "+0.52*min(9.6*0.0727+0.28,1.72)*(0.78*db+0.22)*ΣAt/(ls*Nb)*sqrt(fcm))*ls/db**2) / 1000"
            ),
            "purpose": (
                "ACI Committee 408's own recommended splice-strength model (ACI 408R-03, based on "
                "Zuo & Darwin 2000), solved for the calculated bar stress at bond/splice failure "
                "f_s,calc rather than for a required length -- i.e. exactly the quantity `fsmc` "
                "measures. Built from: c = min(min(csi+0.25, cso), cb) + 0.5*db (effective cover); "
                "ω = min(0.1*(cmax/cmin) + 0.9, 1.25) (cover-imbalance factor, cmin/cmax = the "
                "smaller/larger of that same min(csi+0.25, cso) side term and cb); Ktr = 0.52*tr*td*"
                "ΣAt/(ls*Nb)*sqrt(fcm) (transverse-steel confinement term -- tr (rib-area factor) "
                "uses the paper's suggested average relative rib area Rr=0.0727 since this dataset "
                "doesn't record it per bar, so tr is a fixed ~0.978 constant here, not measured; td "
                "= 0.78*db + 0.22 is the bar-size factor; ΣAt/(ls*Nb) stands in for Atr/(s*n) since "
                "s (tie spacing) isn't its own column but ls/Ns and ΣAt/Ns are, and Ns cancels "
                "algebraically). f_s,calc = fcm**0.25 * (1970*ω + 62*(c*ω+Ktr)*ls/db²), using the "
                "φ=0.82 (ACI 318 Ch. 9 load-combo) calibration -- the /1000 just converts the "
                "psi-scale result to the `fsmc` target's ksi scale. Assumes uncoated, bottom-cast "
                "bars in normal-weight concrete (α=β=λ=1, not columns in this dataset). Works well "
                "on the bundled dataset as-is: r~0.85 correlation against `fsmc`, R²~0.43, and "
                "conservative (f_s,calc ≤ measured) for ~93% of rows -- by far the strongest of the "
                "three reference formulas here, though still a real gap (R² well below 1) for GEP "
                "to help close."
            ),
        },
        "ACI 318 — Corroded RC flexural capacity (strain-compatibility, Mn)": {
            "formula": (
                "((((n_t*(pi/4)*db_t**2*(1-eta_m/100))-(n_c*(pi/4)*db_c**2*(1-eta_m/100)))*(fy0"
                "*(is_high*(0.82-0.0086*eta_m)+(1-is_high)*(is_IC*(1-0.0183*eta_m)+(1-is_IC)*(1"
                "-0.0101*eta_m))))*(((h-c_bot))-((((n_t*(pi/4)*db_t**2*(1-eta_m/100))-(n_c*(pi/"
                "4)*db_c**2*(1-eta_m/100)))*(fy0*(is_high*(0.82-0.0086*eta_m)+(1-is_high)*(is_I"
                "C*(1-0.0183*eta_m)+(1-is_IC)*(1-0.0101*eta_m))))/(0.85*fc*b)))/2)+((n_c*(pi/4)"
                "*db_c**2*(1-eta_m/100)))*(fy0*(is_high*(0.82-0.0086*eta_m)+(1-is_high)*(is_IC*"
                "(1-0.0183*eta_m)+(1-is_IC)*(1-0.0101*eta_m))))*(((h-c_bot))-(c_top)))/1000000)"
            ),
            "purpose": (
                "Ordinary ACI 318 strain-compatibility flexural capacity (Whitney stress "
                "block, tension steel assumed to yield), recomputed with corroded steel "
                "properties in place of the original as-built ones: corroded tension/"
                "compression bar area As = n*(pi/4)*db^2*(1-eta_m/100) (uniform "
                "cross-section loss from mass-loss % eta_m); corroded yield strength "
                "fy,c = fy0*(1-k*eta_m) for eta_m<27.5%, with k=0.0183 for impressed-current "
                "(`is_IC`=1) or k=0.0101 for environmentally-induced/natural corrosion "
                "(`is_IC`=0), switching to fy,c=(0.82-0.0086*eta_m)*fy0 once `is_high`=1 "
                "(eta_m>=27.5%, either mechanism) -- `is_IC`/`is_high` are pre-computed 0/1 "
                "columns rather than an if/else inside the formula, since the physics-formula "
                "parser only supports arithmetic. Nominal moment uses the standard doubly-"
                "reinforced equivalent-couple method: a = (As-As')*fy,c/(0.85*f'c*b) "
                "(stress-block depth, compression steel assumed to also yield), "
                "Mn = (As-As')*fy,c*(d-a/2) + As'*fy,c*(d-d'), with effective depths "
                "d=h-c_bot and d'=c_top; reduces automatically to a singly-reinforced "
                "section when there's no compression steel (As'=0, since n_c=0 there). The "
                "/1,000,000 converts N*mm to kN*m to match the `Mmax` target's units. On the "
                "bundled corroded-beam dataset this is a strong, genuinely physics-only "
                "baseline: r~0.94, R²~0.82 unfitted, conservative (under-predicting) on "
                "average by about 23% -- real headroom left for GEP to find (bond/anchorage "
                "degradation, non-uniform pitting corrosion, stirrup confinement, etc. are "
                "all deliberately left out of this simplified baseline)."
            ),
        },
        "ACI 318 — Corrosion-reduced flexural capacity (K·Mth)": {
            "formula": (
                "(1 - min((d0_db_t_mm**3.11 * (icorr_μA_cm2/1000*Duration_days)**0.708) "
                "/ (W_mm*D_mm)**1.082, 1)) "
                "* ((Tension_Bars*pi*d0_db_t_mm**2/4*(1-ηm/100)*fy_MPa) "
                "* ((D_mm-Bottom_Cover_mm) "
                "- ((Tension_Bars*pi*d0_db_t_mm**2/4*(1-ηm/100)*fy_MPa)/(0.85*f_c_MPa*W_mm))/2) "
                "/ 1000000)"
            ),
            "purpose": (
                "Predicted capacity = K x Mth: an empirical corrosion reduction factor K "
                "multiplying a simplified theoretical nominal moment Mth. Mth is the ordinary "
                "ACI 318 Whitney-stress-block capacity for a singly-reinforced rectangular "
                "section (tension steel assumed to yield, compression steel deliberately "
                "omitted for simplicity), computed with the corroded (mass-loss-reduced) "
                "steel section rather than the as-built one: As = Tension_Bars*(pi/4)*"
                "d0_db_t_mm^2*(1-ηm/100) (ηm = mass loss, %), T = As*fy_MPa (`fy_MPa` is this "
                "dataset's own corrosion-degraded tensile yield stress -- combined with the "
                "ηm reduction on area, both the steel's area and its strength now respond to "
                "corrosion, so K only has to account for bond loss and everything else), "
                "a = T/(0.85*f_c_MPa*W_mm), Mth = T*(d-a/2) with effective depth "
                "d = D_mm - Bottom_Cover_mm (`Bottom Cover (mm)` in this dataset is measured "
                "to the *center* of the tension bar already -- source column \"Bottom Cover "
                "to Ctr of Tension Bar (mm)\" -- so no extra -d0_db_t_mm/2 term is subtracted; "
                "doing so would remove half a bar diameter twice and bias every prediction "
                "low). K = 1 - min(X, 1), X = d0_db_t_mm^3.11*(icorr_μA_cm2/1000*Duration_days)"
                "^0.708 / (W_mm*D_mm)^1.082 (Imam & Kazmi 2017-style calibration; the /1000 "
                "converts icorr from μA/cm² to the mA/cm² the calibration expects), is a "
                "power-law fit of bar diameter, corrosion current density x exposure duration, "
                "and cross-section size, approximating the ratio of experimental-to-theoretical "
                "capacity; X is clipped to 1 (element-wise) so K can't go negative -- without "
                "the clip, K blows up for the most severe bar-size/corrosion combinations in "
                "this dataset and flips the predicted moment's sign, which has no physical "
                "meaning. Two corroded beams with 32mm tension bars (the only 32mm beams with "
                "any corrosion exposure, and over-reinforced at a 5.36% tension ratio -- "
                "roughly double the balanced ratio, so the tension-steel-yields assumption "
                "this formula relies on doesn't hold for them either) are excluded from the "
                "bundled `Corrosion processed` dataset entirely, since even with X clipped "
                "they'd still predict 0 kN·m against measured capacities of 65.6 and "
                "47.4 kN·m. On the remaining 669 rows: R²~0.61, RMSE~14.7 kN·m, MAE~9.1 kN·m, "
                "conservative (under-predicting) for ~88% of rows -- real headroom left for "
                "GEP (bond/anchorage loss, pitting corrosion non-uniformity, stirrup "
                "confinement, compression steel, etc. are all deliberately left out of this "
                "simplified baseline). SI units: W_mm, D_mm, Bottom_Cover_mm, d0_db_t_mm in "
                "mm; f_c_MPa & fy_MPa in MPa; icorr_μA_cm2 in μA/cm²; Duration_days in days; "
                "ηm in %; result in kN·m."
            ),
        },
        "Abrams/fib strength-age law — source-calibrated static baseline (LC3)": {
            "formula": "124.7 * 14.5**(-wb_ratio) * exp(0.424*(1 - sqrt(28/age_days)))",
            "purpose": (
                "Combines Abrams' law (1918) for water content with the fib Model Code 2010 "
                "strength-age function: f = A * B^(-wb_ratio) * exp(s*(1 - sqrt(28/age_days))), "
                "only using this dataset's `wb_ratio` and `age_days` columns. Unlike a per-split "
                "fitted version, A=124.7 MPa, B=14.5 (ln B=2.675) and s=0.424 are fixed literal "
                "constants calibrated once by log-linear OLS on 1005 conventional-concrete "
                "records from a separate source database -- never refit on LC3 -- so this term "
                "is static and carries zero LC3 training-data leakage, matching how every known "
                "formula in this app works (one fixed string, evaluated the same way on every "
                "row/split/seed). ln B=2.675 lands close to Abrams' own mass-basis constant "
                "(2.92) and s=0.424 close to fib's slow-hydration class (0.38), both plausible "
                "independent confirmations of the underlying physics. Applied unchanged to LC3 "
                "(no LC3 fitting at all): R²~0.36 on the full database, R² 0.34-0.41 on held-out "
                "test splits at every availability, with a systematic ~20% underprediction "
                "(median predicted/measured ~0.80) -- i.e. mostly a multiplicative scale bias "
                "between mortar (LC3) and the concrete this was calibrated on, not a random "
                "error. Recommended to pair with the multiplicative combine mode "
                "(physics x (1 + GEP(X))) below so GEP's correction term absorbs that scale "
                "offset directly, plus LC3's specific clinker/age/mix effects this formula "
                "deliberately leaves out; with additive mode GEP instead has to reconstruct a "
                "~20%-of-physics-sized term from scratch for every row. Does not use "
                "`clinker_content` or any other mix-design column -- left for GEP to find a "
                "correction with."
            ),
        },
        "ACI 318-19/-25 — Two-way shear stress vc (λs, βc, αs, k_ecc)": {
            "formula": (
                "sqrt(2/(1+0.004*d)) * lambda_conc * sqrt(fc) "
                "* min(0.33, min(0.17*(1+2/beta_c), 0.083*(2+alpha_s_ACI*d/bo))) / k_ecc"
            ),
            "purpose": (
                "Full ACI 318-19/-25 two-way (punching) shear *stress* provision, "
                "vc = V/(bo*d) -- unlike the simpler \"ACI 318 — Two-way (punching) shear "
                "(Vc)\" entry elsewhere in this list (0.33*sqrt(fc)*bo*d/1000, a force, with "
                "no size-effect/eccentricity refinements), this substitutes every lambda and "
                "limiting factor the code specifies, deliberately *uncapped*: lambda_s_ACI = "
                "sqrt(2/(1+0.004*d)) is the size-effect factor for thick slabs/footings (d in "
                "mm) -- no ceiling at 1, so it exceeds 1 (increases capacity) for d < 250mm, "
                "which is 419 of this dataset's 436 rows; `lambda_conc` is this dataset's own "
                "lightweight-concrete factor (1.0 for normalweight); sqrt(fc) is also uncapped "
                "here (no ceiling at 8.3 MPa^0.5 / f'c~69 MPa), affecting 55 rows with higher-"
                "strength concrete; the governing-of-three vc limit is still "
                "min(0.33, 0.17*(1+2/beta_c), 0.083*(2+alpha_s_ACI*d/bo)) (`beta_c` = column "
                "long/short side ratio, `alpha_s_ACI` = 40/30/20 for interior/edge/corner "
                "columns) -- that min is the code's inherent choice-of-failure-mode logic, not "
                "a cap being removed, so it's kept; and the whole stress is divided by "
                "`k_ecc`, this dataset's eccentric-shear stress-concentration factor, "
                "converting the code's uniform nominal stress into the design value against "
                "an eccentrically loaded connection. No fitted parameters -- every constant "
                "(0.33, 0.17, 0.083, 0.004) is the literal code value, applied unchanged. On "
                "this dataset (no refitting): R²≈0.40 vs `target_V_over_bod`, median "
                "predicted/measured≈0.80 -- both noticeably better than the capped version "
                "this replaced (R²≈-0.13, median≈0.67), since most specimens here are thin "
                "enough (d<250mm) that the size-effect cap was binding and discarding real "
                "predictive signal. Still conservative (underpredicting) for the typical "
                "specimen, leaving headroom for GEP's correction term. Uses `d, fc, "
                "lambda_conc, bo, beta_c, alpha_s_ACI, k_ecc`; doesn't use `alpha_s_CSA, c1, "
                "c2`, load angles/gammas, `rho_avg`, or fibre content -- left for GEP to find "
                "a correction with."
            ),
        },
        "CSA A23.3-19/-24 — Two-way shear stress vc (λs, βc, αs, k_ecc)": {
            "formula": (
                "min(1, 1300/(1000+d)) * lambda_conc * min(sqrt(fc), 8) "
                "* min(0.19*(1+2/beta_c), min(0.19+alpha_s_CSA*d/bo, 0.38)) / k_ecc"
            ),
            "purpose": (
                "CSA A23.3-19/-24 two-way (punching) shear *stress* provision, "
                "vc = V/(bo*d), with every lambda/limiting factor substituted in, as the "
                "CSA-side counterpart to the ACI 318-19/-25 entry above (same dataset, "
                "same `target_V_over_bod`). lambda_s_CSA = 1300/(1000+d) for d > 300mm, else "
                "1 -- written here as the equivalent continuous form min(1, 1300/(1000+d)) "
                "(the two pieces agree exactly at d = 300mm, where 1300/(1000+300) = 1, so "
                "this is not an approximation, just the same piecewise rule expressed as one "
                "expression); `lambda_conc` is this dataset's own lightweight-concrete factor; "
                "sqrt(fc) is capped at 8 (SI, MPa^0.5, i.e. f'c effectively capped near 64 "
                "MPa); the governing-of-three vc limit is min(0.19*(1+2/beta_c), "
                "0.19+alpha_s_CSA*d/bo, 0.38) (`beta_c` = column long/short side ratio, "
                "`alpha_s_CSA` = 4/3/2 for interior/edge/corner columns); divided by `k_ecc`, "
                "this dataset's eccentric-shear stress-concentration factor, same role as in "
                "the ACI entry. No fitted parameters. On this dataset (no refitting): "
                "R²≈0.19 vs `target_V_over_bod`, median predicted/measured≈0.77 -- also "
                "conservative, less so than the ACI version here. Uses `d, fc, lambda_conc, "
                "bo, beta_c, alpha_s_CSA, k_ecc`; doesn't use `alpha_s_ACI, c1, c2`, load "
                "angles/gammas, `rho_avg`, or fibre content -- left for GEP to find a "
                "correction with."
            ),
        },
    }
    # Trimmed the "Known formula" dropdown down to the three the user wants
    # front-and-center (beam shear, Waldron and Skillen, Bolomey's cement/
    # water ratio). The rest stay fully defined in KNOWN_FORMULAS above --
    # just filtered out of the options list below -- so they can be brought
    # back later by shortening this set, without re-typing anything.
    HIDDEN_FORMULAS = {
        "ACI 318 — Detailed one-way shear (Vc, w/ ρw & M/V)",
        "ACI 318-19 — Size-effect one-way shear (Vc)",
        "ACI 318 — Two-way (punching) shear (Vc)",
        "Feret's Law — cement/water/air volumes (concrete strength)",
        "Orangun-Jirsa-Breen — bond stress / lap splice (u/sqrt(fc))",
        "ACI 408R-03 (Zuo & Darwin) — splice bar stress at bond failure (f_s,calc)",
        "ACI 318 — Corroded RC flexural capacity (strain-compatibility, Mn)",
    }
    FORMULA_NONE = "(none / custom)"
    formula_options = [FORMULA_NONE] + [
        name for name in KNOWN_FORMULAS if name not in HIDDEN_FORMULAS
    ]

    # Pre-select each demo dataset's own known formula whenever the data
    # source (or, within "built-in demo", which demo) switches TO it -- each
    # demo was chosen/validated against a specific formula -- and reset to
    # "none / custom" when switching to an upload. Only right on that switch
    # (not on every rerun), so it doesn't clobber the user's own later
    # dropdown choice or text edits.
    current_dataset_id = "upload" if source == "Upload a file" else f"demo:{demo_choice}"
    prev_dataset_id = st.session_state.get("_prev_dataset_id")
    if current_dataset_id != prev_dataset_id:
        st.session_state["_prev_dataset_id"] = current_dataset_id
        st.session_state["known_formula_select"] = (
            DEMO_DATASETS[demo_choice]["default_known_formula"]
            if source != "Upload a file" else FORMULA_NONE
        )

    known_choice = st.selectbox(
        "Known formula (optional)", formula_options, key="known_formula_select",
        help="Pick a built-in ACI 318 formula to pre-fill the box below (shown with its "
             "equation and purpose), or leave as \"(none / custom)\" and type your own.",
    )

    # Re-fill the text box's contents only when the DROPDOWN choice itself
    # just changed (not on every rerun), so a hand-edit afterward sticks.
    prev_known_choice = st.session_state.get("_prev_known_choice")
    if known_choice != prev_known_choice:
        st.session_state["_prev_known_choice"] = known_choice
        st.session_state["physics_formula_input"] = (
            KNOWN_FORMULAS[known_choice]["formula"] if known_choice != FORMULA_NONE else ""
        )

    if known_choice != FORMULA_NONE:
        info = KNOWN_FORMULAS[known_choice]
        gep_note(
            f'<span class="gep-label">{known_choice}</span>{_equation_to_html(info["formula"])}<br>'
            f'<em>{info["purpose"]}</em>'
        )

    physics_formula = st.text_input(
        "Physics formula", value="", placeholder="e.g. 0.17*sqrt(fc)*bw*d/1000",
        key="physics_formula_input",
    )
    physics_formula = physics_formula.strip() or None

    combine_mode_label = st.radio(
        "Combination mode",
        ["Additive: physics + GEP(X)", "Multiplicative: physics × (1 + GEP(X))"],
        key="combine_mode_select",
        help="How the physics baseline and GEP's discovered term are combined into a prediction. "
             "Additive (the default) has GEP fit the plain residual (target − physics) -- good when "
             "the error is roughly a fixed offset. Multiplicative has GEP fit a *fractional* "
             "correction instead (target / physics − 1, so 0 still means \"physics is already "
             "correct\") -- often a better fit when the error scales with physics's own prediction "
             "(e.g. a consistent % over/under-estimate). This drives the search itself, so it can't "
             "be changed for a run after it's started -- pick it before running.",
    )
    combine_mode = "mult" if combine_mode_label.startswith("Multiplicative") else "add"

    if physics_formula:
        try:
            validate_formula(physics_formula, feature_names)
            phys_fn = make_physics_fn(physics_formula, feature_names)
            phys_vals = phys_fn(df[feature_names].to_numpy(dtype=float))
            y_vals = df[target_name].to_numpy(dtype=float)
            c1, c2, c3 = st.columns(3)
            c1.metric("physics(X) mean", f"{phys_vals.mean():.3g}")
            c2.metric("target mean", f"{y_vals.mean():.3g}")
            if combine_mode == "mult":
                eps = physics_clip_eps(y_vals)
                phys_vals_clipped, n_clipped = clip_physics_for_mult(phys_vals, eps)
                frac_resid = y_vals / phys_vals_clipped - 1
                c3.metric("fractional correction (target/physics − 1) mean", f"{frac_resid.mean():.3g}")
                st.success("Formula parsed OK — GEP will fit the fractional correction shown above.")
                if n_clipped:
                    st.warning(
                        f"{n_clipped} row(s) here have a physics(X) value very close to zero "
                        f"(within ±{eps:.3g}). Those are clipped away from zero before dividing "
                        f"(both in this preview and during the search itself), since dividing by "
                        f"~0 would otherwise blow up -- results for those rows may be less reliable."
                    )
            else:
                resid = y_vals - phys_vals
                c3.metric("residual (target - physics) mean", f"{resid.mean():.3g}")
                st.success("Formula parsed OK — GEP will fit the residual shown above.")
        except FormulaError as e:
            st.error(f"Formula error: {e}")
            st.stop()
    elif combine_mode == "mult":
        st.warning(
            "No physics formula is entered above. In multiplicative mode, physics(X) defaults to "
            "0, which makes physics × (1 + GEP(X)) always 0 regardless of what GEP discovers -- "
            "enter a physics formula above, or switch back to additive mode."
        )


    # --------------------------------------------------------------------------
    # 4. Equation complexity & search budget
    # --------------------------------------------------------------------------
    st.header("4. Equation complexity & search budget")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.markdown(
                "Head length (max number of operators per gene)",
                help="Bigger = each gene sub-expression can be a deeper/more complex tree.",
            )
            head_length = st.number_input(
                "Head length (max number of operators per gene)", min_value=1, max_value=8, value=2,
                label_visibility="collapsed", width=90,
            )
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.markdown(
                "Number of genes (number of sub terms of length h)",
                help="Bigger = more independent terms can be combined into the final expression.",
            )
            n_genes = st.number_input(
                "Number of genes (number of sub terms of length h)", min_value=1, max_value=6, value=2,
                label_visibility="collapsed", width=90,
            )
    with c2:
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.markdown(
                "Generations per run",
                help="How many rounds of evolving and improving candidate equations to run. "
                     "More generations give the search more chances to refine its answer, but "
                     "take longer to finish.",
            )
            generations = st.number_input(
                "Generations per run", min_value=10, max_value=5000, value=750, step=10,
                label_visibility="collapsed", width=90,
            )
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.markdown(
                "Population size per run",
                help="How many candidate equations are tried side-by-side in each generation. "
                     "A bigger population explores more possibilities at once, but each "
                     "generation takes longer to run.",
            )
            pop_size = st.number_input(
                "Population size per run", min_value=10, max_value=2000, value=100, step=10,
                label_visibility="collapsed", width=90,
            )

    with st.expander("Advanced GEP settings"):
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.markdown(
                "Tournament size",
                help="How many randomly-picked candidate equations compete against each other "
                     "each time GEP chooses a parent for the next generation. Bigger = the best "
                     "current solutions win more often (search converges faster, but explores "
                     "less and can get stuck early).",
            )
            tournsize = st.number_input(
                "Tournament size", min_value=2, max_value=20, value=5,
                label_visibility="collapsed", width=90,
            )
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.markdown(
                "Hall-of-Fame size (solutions recorded per run)",
                help="How many of the best *distinct* equations seen during a run are kept and "
                     "reported afterward, instead of only the single best one -- useful for "
                     "seeing other strong, differently-shaped solutions the search turned up.",
            )
            champs = st.number_input(
                "Hall-of-Fame size (solutions recorded per run)", min_value=1, max_value=10, value=3,
                label_visibility="collapsed", width=90,
            )
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.markdown(
                "Validation fraction of train+val pool",
                help="Of the training-data pool for a given run (not the separate held-out test "
                     "set), what fraction is set aside to score/rank candidate equations during "
                     "the search rather than used to fit them -- guards against picking an "
                     "equation that only looks good on the exact rows it was fit to.",
            )
            val_frac = st.number_input(
                "Validation fraction of train+val pool", min_value=0.05, max_value=0.4, value=0.15,
                step=0.01, format="%.2f", label_visibility="collapsed", width=90,
            )


    # --------------------------------------------------------------------------
    # 5. Sweep design: training-data availability x random seeds
    # --------------------------------------------------------------------------
    st.header("5. Search design")
    c1, c2 = st.columns(2)
    with c1:
        ptrains_str = st.text_input(
            "Training-data-availability levels (fraction of whole dataset in train+val pool)",
            value="0.05, 0.1, 0.15, 0.25, 0.5, 0.75",
            help="Each value is a fraction of the *whole* dataset set aside as that level's "
                 "train+val pool (the rest becomes the held-out test set for scoring it). The "
                 "search is repeated once per level (crossed with every seed below), so you can "
                 "see which features/terms keep showing up even when there's very little "
                 "training data, versus ones that only emerge once plenty is available.",
        )
        try:
            ptrains = parse_float_list(ptrains_str)
            assert ptrains and all(0 < p < 1 for p in ptrains)
        except Exception:
            st.error("Enter comma-separated fractions strictly between 0 and 1, e.g. 0.05, 0.1, 0.5")
            st.stop()
    with c2:
        n_seeds = st.number_input(
            "Number of random seeds", min_value=1, max_value=200, value=10,
            help="How many independent GEP runs (each with a different random seed, so a "
                 "different starting population and evolutionary path) are done at *each* "
                 "training-data-availability level. More seeds means a slower sweep but a more "
                 "reliable answer to \"does this term/feature show up consistently,\" since any "
                 "single run's result is otherwise just one roll of the dice.",
        )
        custom_seeds = st.text_input(
            "...or a custom seed list (overrides count above, e.g. 0-19 or 10,28,50)", value="",
            help="Optional. Lets you pin down the exact seed values used instead of just a count "
                 "-- handy for reproducing a specific earlier run, or picking up where a previous "
                 "sweep left off. Accepts a comma-separated list, a range like `0-19`, or a mix "
                 "of both (e.g. `0-9, 15, 20-24`). Leave blank to just use the count above.",
        )
        seeds = parse_int_list(custom_seeds) if custom_seeds.strip() else list(range(int(n_seeds)))

    total_runs = len(seeds) * len(ptrains)
    st.info(f"This search will run **{len(seeds)} seeds x {len(ptrains)} training levels = {total_runs} GEP evolutions**.")


    # --------------------------------------------------------------------------
    # 6. Run
    # --------------------------------------------------------------------------
    st.header("6. Run")
    run_clicked = st.button("Run search", type="primary")

    if run_clicked:
        progress_bar = st.progress(0.0)
        metrics_ph = st.empty()
        table_ph = st.empty()
        all_records = []
        t_start = time.time()

        for prog in sweep(
            df, feature_names, target_name, seeds, ptrains,
            physics_formula=physics_formula, combine_mode=combine_mode,
            generations=int(generations), pop_size=int(pop_size),
            head_length=int(head_length), n_genes=int(n_genes), tournsize=int(tournsize),
            champs=int(champs), val_frac_of_pool=float(val_frac), verbose=False,
        ):
            all_records.extend(prog["records"])
            done, total = prog["done"], prog["total"]
            elapsed = time.time() - t_start
            eta = elapsed / done * (total - done)
            progress_bar.progress(done / total, text=f"{done}/{total} runs complete")
            with metrics_ph.container():
                m1, m2, m3 = st.columns(3)
                m1.metric("Elapsed", f"{elapsed:.0f}s")
                m2.metric("Avg time/run", f"{elapsed/done:.1f}s")
                m3.metric("ETA remaining", f"{eta:.0f}s")
            best = prog["records"][0]
            with table_ph.container():
                st.caption(
                    f"Last: seed={prog['seed']} ptrain={prog['ptrain']} ({prog['dt']:.1f}s) "
                    f"test_r2={best['test_r2']:.3f} test_mae={best['test_mae']:.2f} — `{best['expr']}`"
                )

        st.session_state["results_df"] = pd.DataFrame(all_records)
        # Freeze everything the results section needs, so later widget changes
        # (e.g. re-selecting features) can't desync the report from what was run.
        st.session_state["results_physics_formula"] = physics_formula
        st.session_state["results_combine_mode"] = combine_mode
        st.session_state["results_known_formula_name"] = (
            known_choice if known_choice != FORMULA_NONE else None
        )
        st.session_state["results_dataset_label"] = dataset_label
        st.session_state["results_feature_names"] = list(feature_names)
        st.session_state["results_target_name"] = target_name
        st.session_state["results_data"] = df[feature_names + [target_name]].copy()
        st.session_state["results_hyperparams"] = {
            "head_length": int(head_length),
            "n_genes": int(n_genes),
            "generations": int(generations),
            "pop_size": int(pop_size),
            "tournsize": int(tournsize),
            "champs": int(champs),
            "val_frac": float(val_frac),
            "ptrains": list(ptrains),
            "seeds": list(seeds),
            "total_runs": total_runs,
            "combine_mode": combine_mode,
        }
        # Which of those same rows were flagged as response-variable outliers
        # in step 2, so the results plots can highlight them -- aligned
        # positionally with "results_data" since neither has been re-indexed
        # since (row order/index is only ever dropped from, never reordered).
        st.session_state["results_outlier_mask"] = outlier_mask.reindex(df.index, fill_value=False).to_numpy()
        # Stash the completion message for the results section (below) to show,
        # rather than displaying it here, since we immediately rerun the app
        # right after this so the now-wide-for-no-reason setup sidebar shrinks
        # back down and the results take over the main area right away.
        st.session_state["results_run_summary"] = f"Done: {total_runs} runs in {time.time()-t_start:.1f}s."
        n_clipped_total = sum(r.get("n_physics_clipped", 0) for r in all_records)
        if combine_mode == "mult" and n_clipped_total:
            st.session_state["results_physics_clipped_warning"] = (
                f"Multiplicative mode: physics(X) was clipped away from zero at {n_clipped_total} "
                f"train/val/test point(s) (summed across all runs) where it was very close to 0, "
                f"to keep the fractional correction from blowing up. Results at those points may be "
                f"less reliable."
            )
        else:
            st.session_state.pop("results_physics_clipped_warning", None)
        st.rerun()


# --------------------------------------------------------------------------
# 7. Results
# --------------------------------------------------------------------------
if "results_df" in st.session_state:
    st.header("7. Emergent feature terms")
    if st.session_state.get("results_run_summary"):
        st.success(st.session_state["results_run_summary"])
    if st.session_state.get("results_physics_clipped_warning"):
        st.warning(st.session_state["results_physics_clipped_warning"])
    results_df = st.session_state["results_df"]
    used_physics_formula = st.session_state.get("results_physics_formula")
    used_combine_mode = st.session_state.get("results_combine_mode", "add")
    used_known_formula_name = st.session_state.get("results_known_formula_name")
    used_dataset_label = st.session_state.get("results_dataset_label")
    used_hyperparams = st.session_state.get("results_hyperparams", {})
    st.caption(
        "Combination mode for this run: **"
        + ("Multiplicative -- physics × (1 + GEP(X))" if used_combine_mode == "mult"
           else "Additive -- physics + GEP(X)")
        + "**"
    )

    # Raw (pre-R²-filter) solution count per training-data-availability level --
    # i.e. seeds x Hall-of-Fame size for that level, since every run logs all of
    # its HOF members regardless of fit quality. Used as the denominator for the
    # pie chart's per-level breakdown below, so that heavy R² filtering at a
    # level shows up as a LOWER percentage (fewer of that level's attempts
    # actually contained the term) rather than silently shrinking the
    # denominator to just the few survivors.
    raw_counts_per_ptrain = (
        results_df.groupby("ptrain").size().to_dict() if "ptrain" in results_df.columns else {}
    )

    # Optional quality filter: only count recorded solutions whose R² on the
    # whole train+val pool (the data each one was fit/scored on during the
    # search -- NOT the separate held-out test set, which stays untouched)
    # clears this bar, before tallying which terms/features recur most often
    # below. Applied per training-data-availability level rather than as one
    # global cutoff, since less training data makes a lower R² unavoidable --
    # a uniform floor would otherwise risk silently wiping out every solution
    # from a low-data level. If *none* of a level's solutions clear the bar,
    # that level's single best solution is kept anyway so it isn't dropped
    # from the analysis entirely.
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.markdown(
            "Minimum R² (train+val pool) to include a solution",
            help="Recorded solutions below this train+val R² are excluded before ranking "
                 "terms/features below, so weak fits don't dilute the results. Uses R² on the "
                 "combined train+val pool (not the held-out test set), so test data is never "
                 "used to decide which solutions to keep. Applied per training-data-availability "
                 "level -- if none of a level's solutions clear the bar, that level's single best "
                 "solution is kept anyway rather than dropping the level entirely. Set very low "
                 "(e.g. -10) to effectively disable this filter.",
        )
        min_trainval_r2 = st.number_input(
            "Minimum R² (train+val pool) to include a solution",
            min_value=-100.0, max_value=1.0, value=0.5, step=0.05, format="%.2f",
            label_visibility="collapsed", width=90, key="min_trainval_r2",
        )

    if "trainval_r2" in results_df.columns and len(results_df):
        # Vectorized per-group floor (rather than groupby().apply()) so the
        # "ptrain" grouping column can't get silently dropped by pandas.
        keep_mask = results_df["trainval_r2"] >= min_trainval_r2
        kept_counts = results_df.loc[keep_mask].groupby("ptrain").size()
        best_idx_per_group = results_df.groupby("ptrain")["trainval_r2"].idxmax()
        for ptrain_val in results_df["ptrain"].unique():
            if kept_counts.get(ptrain_val, 0) == 0:
                keep_mask.loc[best_idx_per_group[ptrain_val]] = True
        ranked_solutions_df = results_df[keep_mask]
        n_dropped = len(results_df) - len(ranked_solutions_df)
        if n_dropped > 0:
            st.caption(
                f"{n_dropped} of {len(results_df)} recorded solution(s) excluded from the term "
                f"ranking below (train+val R² < {min_trainval_r2:.2f})."
            )
    else:
        # Results recorded before this filter existed won't have this column.
        ranked_solutions_df = results_df
        st.caption("(Results predate the train+val R² column -- quality filter unavailable; showing all solutions.)")

    report = parse_results_df(
        ranked_solutions_df, by_ptrain=(ranked_solutions_df["ptrain"].nunique() > 1),
        physics_formula=used_physics_formula,
    )

    if report["parse_failures"]:
        st.warning(f"{report['parse_failures']}/{report['n_solutions']} expressions failed to parse.")

    var_df = report["var_df"]
    term_df = report["term_df"]

    # Compact column widths for the small count/percentage columns in the
    # term-ranking tables below -- they don't need nearly as much horizontal
    # space as the "term"/"example" text columns, so give them a "small" hint
    # and let the dataframe distribute the rest of the width to the text ones.
    _term_table_col_config = {
        "term": st.column_config.TextColumn("term", width="medium"),
        "n_variables": st.column_config.NumberColumn("n_variables", width="small"),
        "n_solutions": st.column_config.NumberColumn("n_solutions", width="small"),
        # Inline mini progress bar + number in one cell, instead of a bare
        # number -- relative prevalence reads at a glance without needing a
        # separate chart alongside the table.
        "pct_of_solutions": st.column_config.ProgressColumn(
            "pct_of_solutions", format="%.1f%%", min_value=0, max_value=100, width="medium",
        ),
        "example": st.column_config.TextColumn("example", width="large"),
    }

    # Dashboard layout for this section: two full-width tabs instead of a
    # side-by-side column split, so each gets the entire pane's width and
    # there's no need to scroll to see both at once -- results load first,
    # then the user clicks over to the equation-building tools.
    tab_results, tab_equation = st.tabs(["Results", "Equation building"])

    plot_feature_names = st.session_state.get("results_feature_names")
    plot_target_name = st.session_state.get("results_target_name")
    plot_data = st.session_state.get("results_data")

    with tab_results:
        tab_plots, tab_more, tab_range = st.tabs(["Plots", "More", "Applicable range"])

        # ----------------------------------------------------------------
        # More tab is filled in FIRST (even though "Plots" is shown first,
        # on the left) because the ranked-term tables and the "include
        # single-variable terms" toggle now live here -- decluttering the
        # top of the Results page -- and the Plots tab below still needs
        # both `interactions` and `include_single_vars`. Streamlit only
        # cares about each tab's index/label, not which `with` block runs
        # first, so this is safe.
        # ----------------------------------------------------------------
        with tab_more:
            # Display-only cosmetic cleanup for these plain (non-HTML)
            # tables: real superscripts + '·' instead of raw '^-1'/'*'.
            # Built once here and reused below for the by-ptrain pivot too.
            # The underlying term_df/interactions/CSVs keep the raw ASCII
            # form (portable, re-parseable) -- only what's shown on screen
            # is prettified.
            _term_sigs_all = report["term_sigs"]
            _label_to_pretty = (
                dict(zip(term_df["term"], (_pretty_term_text(s) for s in _term_sigs_all)))
                if len(term_df) else {}
            )

            def _display_term_table(df_in):
                if not len(df_in):
                    return df_in
                out = df_in.copy()
                out["term"] = out["term"].map(_label_to_pretty).fillna(out["term"])
                if "example" in out.columns:
                    out["example"] = out["example"].map(_prettify_ascii_expr)
                return out

            # Primary result: which terms/interactions GEP kept re-discovering.
            # Side-by-side (rather than stacked) to save vertical space, with
            # a bolded sub-label over each table (rather than a plain
            # caption) so the two are easy to tell apart at a glance, and
            # dividers below separating this from the other, secondary
            # sections (by-level breakdown / main effects / downloads).
            st.subheader("Ranked terms / interactions")
            st.caption("Every recorded term/interaction, ranked by how many solutions contained it.")
            rt1, rt2 = st.columns(2, gap="medium")
            with rt1:
                st.markdown("**All ranked terms**")
                st.dataframe(_display_term_table(term_df), hide_index=True, width="stretch", height=220, column_config=_term_table_col_config)
            with rt2:
                st.markdown("**Multi-variable interactions only**")
                interactions = term_df[term_df["n_variables"] >= 2] if len(term_df) else term_df
                st.dataframe(_display_term_table(interactions), hide_index=True, width="stretch", height=220, column_config=_term_table_col_config)

            # st.toggle over st.checkbox -- same bool return value, no
            # downstream change needed, just a more modern-looking control
            # to sit under the two ranked-term tables above.
            include_single_vars = st.toggle(
                "Include single-variable terms in the 'top 3' (not just multi-variable interactions)",
                value=False, key="top3_include_single_vars",
                help="By default the top 3 are chosen only from multi-variable interaction terms. "
                     "Check this to let single-variable terms (e.g. just `d` or `1/fc`) compete for "
                     "those slots too, ranked purely by how often each term recurs across recorded solutions.",
            )

            if "by_ptrain_df" in report:
                st.divider()
                st.subheader("Term frequency by training-data-availability level")
                st.caption("Share of that level's maximum possible solutions (seeds × Hall-of-Fame size) containing each term. Darker = more prevalent.")
                pivot = report["by_ptrain_df"].pivot_table(
                    index="term", columns="ptrain", values="pct_of_solutions", fill_value=0.0,
                ).sort_values(by=list(report["by_ptrain_df"]["ptrain"].unique())[-1], ascending=False)
                pivot = pivot.rename(index=_label_to_pretty)
                pivot.columns = [f"{c * 100:g}% training data" for c in pivot.columns]
                # Heatmap-style shading (0-100% fixed scale, shared across all
                # cells via axis=None) so a glance at color alone shows which
                # terms are consistently strong across every data level vs.
                # which only show up at one. Formatted with a literal "%" per
                # cell on top of the color, rather than a separate legend.
                pivot_styled = pivot.style.background_gradient(
                    cmap="Blues", vmin=0, vmax=100, axis=None,
                ).format("{:.1f}%")
                st.dataframe(pivot_styled, width="stretch", height=200)

            st.divider()
            st.subheader("Main-effect feature frequency")
            st.caption("How often each feature appears anywhere in the expanded expression, regardless of which term it's in.")
            mt1, mt2 = st.columns(2, gap="medium")
            with mt1:
                st.dataframe(
                    var_df, hide_index=True, width="stretch", height=200,
                    column_config={
                        "feature": st.column_config.TextColumn("feature", width="medium"),
                        "n_solutions": st.column_config.NumberColumn("n_solutions", width="small"),
                        "pct_of_solutions": st.column_config.ProgressColumn(
                            "pct_of_solutions", format="%.1f%%", min_value=0, max_value=100, width="medium",
                        ),
                    },
                )
            with mt2:
                if len(var_df):
                    # Horizontal Plotly bar, sorted descending and colored on
                    # a single continuous scale matching the pivot heatmap
                    # above, instead of the plain default-colored st.bar_chart
                    # -- reads more like a deliberate part of the report.
                    var_sorted = var_df.sort_values("pct_of_solutions", ascending=True)
                    bar_fig = go.Figure(
                        data=[go.Bar(
                            x=var_sorted["pct_of_solutions"],
                            y=var_sorted["feature"],
                            orientation="h",
                            marker=dict(
                                color=var_sorted["pct_of_solutions"],
                                colorscale="Blues", cmin=0, cmax=100,
                            ),
                            text=[f"{p:g}%" for p in var_sorted["pct_of_solutions"]],
                            textposition="outside",
                            hovertemplate="<b>%{y}</b><br>%{x:.1f}% of solutions<extra></extra>",
                        )]
                    )
                    bar_fig.update_layout(
                        margin=dict(l=10, r=30, t=10, b=10), height=200,
                        xaxis=dict(title="% of solutions", range=[0, 100]),
                        yaxis=dict(title=None),
                    )
                    st.plotly_chart(bar_fig, width="stretch", key="main_effects_bar_chart")

            st.divider()
            st.subheader("Downloads")
            with st.container(border=True):
                d1, d2, d3 = st.columns(3, gap="medium")
                d1.download_button(
                    "⬇️ Raw results CSV", results_df.to_csv(index=False), file_name="search_results.csv",
                    mime="text/csv", width="stretch",
                )
                d2.download_button(
                    "⬇️ Main effects ranked CSV", var_df.to_csv(index=False), file_name="main_effects_ranked.csv",
                    mime="text/csv", width="stretch",
                )
                d3.download_button(
                    "⬇️ Terms ranked CSV", term_df.to_csv(index=False), file_name="terms_ranked.csv",
                    mime="text/csv", width="stretch",
                )

        # ----------------------------------------------------------------
        # Plots tab: auto-generated predicted-vs-actual plots for the top
        # few recurring terms, plus physics-only / physics+combo reference
        # plots. Each group is rendered as ONE matplotlib figure with
        # side-by-side subplots (rather than one st.pyplot call per plot)
        # so the whole tab fits in far less vertical space.
        # ----------------------------------------------------------------
        with tab_plots:
            term_pool = term_df if include_single_vars else interactions

            good_terms = []  # (term_str, term_vals, row) for terms that plotted OK, for the combined plot below
            if plot_data is None:
                st.caption("(run a search to see plots)")
            else:
                term_sigs = report["term_sigs"]
                y_actual = plot_data[plot_target_name].to_numpy(dtype=float)
                phys_fn_whole = make_physics_fn(used_physics_formula, plot_feature_names)
                phys_vals_whole = phys_fn_whole(plot_data[plot_feature_names].to_numpy(dtype=float))
                n_phys_clipped_whole = 0
                if used_combine_mode == "mult":
                    _eps_whole = physics_clip_eps(y_actual)
                    phys_vals_whole, n_phys_clipped_whole = clip_physics_for_mult(phys_vals_whole, _eps_whole)
                    if n_phys_clipped_whole:
                        st.caption(
                            f"⚠ {n_phys_clipped_whole} point(s) here have a physics(X) value very "
                            f"close to zero; clipped to ±{_eps_whole:.3g} before computing the "
                            f"multiplicative predictions below."
                        )

                # Rows flagged as response-variable outliers back in step 2
                # (aligned positionally with "results_data"/plot_data), so
                # the scatter plots below can call them out in orange and
                # report R² with vs. without them.
                # Left column holds the compact toggle/control widgets, right
                # column holds the (visually heavier) interactions pie chart
                # below -- side by side rather than stacked, to save vertical
                # space. Created once here and reused later (once more, for
                # the pie chart itself) the same way `term_cols` is reused.
                layout_left, layout_right = st.columns([1, 2], vertical_alignment="top")

                outlier_mask_arr = st.session_state.get("results_outlier_mask")
                has_outliers = (
                    outlier_mask_arr is not None
                    and len(outlier_mask_arr) == len(y_actual)
                    and outlier_mask_arr.any()
                )
                show_outliers = True
                if has_outliers:
                    with layout_left:
                        show_outliers = st.checkbox(
                            "Highlight response-variable outliers (orange) and report R² without them",
                            value=True, key="plots_show_outliers",
                            help="Rows flagged as response-variable outliers in step 2 are shown in "
                                 "orange in the plots below, alongside R² both with and without "
                                 "those points. Uncheck to show every point the normal way, with a "
                                 "single overall R².",
                        )

                # Predicted-value outliers are a separate concept from the
                # response-variable outliers above: instead of using the
                # step-2 flag, each plot below flags its OWN unusually
                # extreme predictions (since the prediction differs per
                # term/physics/combo panel), using the same z-score method.
                PRED_OUTLIER_N_STD = 3.0
                with layout_left:
                    show_pred_outliers = st.checkbox(
                        "Highlight predicted-value outliers (red) and report R² without them",
                        value=True, key="plots_show_pred_outliers",
                        help=f"Flags points whose *predicted* value is unusually extreme for that "
                             f"particular plot (more than {PRED_OUTLIER_N_STD:g} standard deviations "
                             "from the mean prediction) -- computed separately for each plot since the "
                             "prediction differs per term/combo, then shown in red alongside R² without "
                             "them. Uncheck to turn this off.",
                    )

                # Extra vertical room for the (possibly multi-line) R² block
                # under each plot title -- one more line per active toggle,
                # so titles don't get clipped or overlap the plot below.
                _extra_r2_lines = (1 if (show_outliers and has_outliers) else 0) + (1 if show_pred_outliers else 0)
                _term_fig_h = 2.8 + 0.35 * _extra_r2_lines
                _combo_fig_h = 2.6 + 0.35 * _extra_r2_lines

                def _plot_scatter(ax, prediction, title):
                    n_pts = len(y_actual)
                    r2_all = r2_score(y_actual, prediction)

                    resp_active = show_outliers and has_outliers
                    pred_std = prediction.std()
                    pred_outlier_mask = (
                        np.abs(prediction - prediction.mean()) > PRED_OUTLIER_N_STD * pred_std
                        if pred_std and not np.isnan(pred_std) else np.zeros(n_pts, dtype=bool)
                    )
                    pred_active = show_pred_outliers and bool(pred_outlier_mask.any())

                    # 0 = normal (blue), 1 = response outlier (orange), 2 = predicted-value
                    # outlier (red). If a point qualifies for both, red wins (drawn last/on top).
                    cat = np.zeros(n_pts, dtype=int)
                    if resp_active:
                        cat[outlier_mask_arr] = 1
                    if pred_active:
                        cat[pred_outlier_mask] = 2

                    styles = {
                        0: dict(color="tab:blue", label="normal", alpha=0.6, s=12),
                        1: dict(color="orange", label="response outlier", alpha=0.85, s=18),
                        2: dict(color="red", label="predicted-value outlier", alpha=0.85, s=18),
                    }
                    for c in (0, 1, 2):
                        sel = cat == c
                        if sel.any():
                            ax.scatter(y_actual[sel], prediction[sel], edgecolors="none", **styles[c])

                    # Each R² variant gets its OWN line (rather than being
                    # crammed together with " / ") since these titles sit
                    # above small subplots -- a single long line was
                    # overlapping/getting cut off once 2-3 variants were shown.
                    suffix = " (all)" if (resp_active or pred_active) else ""
                    r2_lines = [f"R² = {r2_all:.3f}{suffix}"]
                    if resp_active:
                        inlier = ~outlier_mask_arr
                        r2_lines.append(f"R² = {r2_score(y_actual[inlier], prediction[inlier]):.3f} (no resp. outliers)")
                    if pred_active:
                        inlier = ~pred_outlier_mask
                        r2_lines.append(f"R² = {r2_score(y_actual[inlier], prediction[inlier]):.3f} (no pred. outliers)")
                    if resp_active or pred_active:
                        ax.legend(fontsize=6, framealpha=0.5, loc="best")
                    title = f"{title}\n" + "\n".join(r2_lines)
                    title_fontsize = 9 if len(r2_lines) <= 1 else 7.5

                    lo = min(y_actual.min(), prediction.min())
                    hi = max(y_actual.max(), prediction.max())
                    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
                    ax.set_xlabel(f"Actual {plot_target_name}", fontsize=8)
                    ax.set_ylabel("Predicted", fontsize=8)
                    ax.set_title(title, fontsize=title_fontsize, linespacing=1.25)
                    ax.tick_params(labelsize=7)

                if len(term_pool) == 0:
                    st.caption("(no terms found)" if include_single_vars else "(no multi-variable interactions found)")
                else:
                    # ------------------------------------------------------
                    # Interactions pie chart: relative frequency of the top N
                    # discovered terms, with a per-training-data-availability
                    # breakdown on hover so the user can gauge how robust each
                    # term is (found consistently across ptrain levels vs.
                    # only ever showing up at one). Slices are NOT a strict
                    # partition of "all solutions" -- one solution's expanded
                    # expression can contain several terms at once -- so this
                    # is relative prevalence, not mutually-exclusive shares.
                    # ------------------------------------------------------
                    max_pie_terms = min(30, len(term_pool))
                    default_pie_n = min(15, max_pie_terms)
                    if max_pie_terms > 5:
                        with layout_left:
                            n_pie_terms = st.number_input(
                                "Number of terms to show in the interactions pie chart",
                                min_value=min(5, max_pie_terms), max_value=max_pie_terms,
                                value=default_pie_n, step=1, key="pie_n_terms",
                            )
                    else:
                        n_pie_terms = max_pie_terms
                    pie_rows = term_pool.head(n_pie_terms)

                    # One breakdown row per training-data-availability level
                    # that exists in this run at all (raw_counts_per_ptrain's
                    # keys), in ascending order, even for levels this term
                    # never appeared at (shown as "0 of <max> (0.0%)") -- so
                    # the row count/order is the same for every term slice.
                    # Each row's denominator is that level's MAXIMUM possible
                    # solutions (seeds x Hall-of-Fame size, before the R²
                    # filter), not the post-filter survivor count, so a level
                    # decimated by the R² filter reads as a low percentage
                    # instead of an artificially inflated one.
                    all_ptrains_sorted = sorted(raw_counts_per_ptrain)
                    by_ptrain_df = report.get("by_ptrain_df")
                    show_breakdown = len(all_ptrains_sorted) > 1
                    hover_extra = []
                    for _, prow in pie_rows.iterrows():
                        if not show_breakdown:
                            hover_extra.append("")
                            continue
                        counts_by_pt = {}
                        if by_ptrain_df is not None and len(by_ptrain_df):
                            sub = by_ptrain_df[by_ptrain_df["term"] == prow["term"]]
                            counts_by_pt = dict(zip(sub["ptrain"], sub["n_solutions"]))
                        lines = []
                        for pt in all_ptrains_sorted:
                            n_this = counts_by_pt.get(pt, 0)
                            raw_total = raw_counts_per_ptrain.get(pt, 0)
                            pct = round(100 * n_this / raw_total, 1) if raw_total else 0.0
                            lines.append(f"training data {pt * 100:g}%: {n_this} of {raw_total} ({pct}%)")
                        hover_extra.append("<br>".join(lines))

                    # Slice percentage = this term's share of ALL recorded
                    # solutions (report["n_solutions"], after the R² floor
                    # filter) -- the same "pct_of_solutions" already shown in
                    # the ranked-term tables above -- NOT Plotly's default
                    # "%{percent}", which would only total 100% across
                    # whichever N slices are drawn here.
                    pretty_labels = [_pretty_term_text(term_sigs[i]) for i in pie_rows.index]

                    # Percent sign is baked into the customdata STRING itself
                    # (e.g. "60%") rather than left as literal template text
                    # right after a "%{...}" token -- a literal "%" glued
                    # onto a token close-brace is what caused the per-level
                    # breakdown to render nested inside the overall row's
                    # "(...%)" instead of as separate lines below it.
                    pct_strs = [f"{p:g}%" for p in pie_rows["pct_of_solutions"].tolist()]
                    customdata_arr = list(zip(pct_strs, hover_extra))

                    hovertemplate = (
                        "<b>%{label}</b><br>"
                        f"%{{value}} of {report['n_solutions']} solutions (%{{customdata[0]}})"
                        + ("<br>%{customdata[1]}" if show_breakdown else "")
                        + "<extra></extra>"
                    )
                    pie_fig = go.Figure(
                        data=[go.Pie(
                            labels=pretty_labels,
                            values=pie_rows["n_solutions"].tolist(),
                            customdata=customdata_arr,
                            texttemplate="%{label}<br>%{customdata[0]}",
                            hovertemplate=hovertemplate,
                            textfont=dict(size=15),
                        )]
                    )
                    pie_fig.update_layout(
                        margin=dict(l=10, r=10, t=10, b=10), height=420, showlegend=False,
                        font=dict(size=15), hoverlabel=dict(font_size=14),
                    )
                    with layout_right:
                        st.plotly_chart(pie_fig, key="terms_pie_chart")
                        st.caption(
                            f"Slice % = share of all {report['n_solutions']} recorded solutions (after "
                            "the R² filter above) that contained that term. Per-level hover % = share "
                            "of that level's maximum possible solutions (seeds × Hall-of-Fame size, "
                            "before the R² filter) that contained that term."
                        )

                    # ------------------------------------------------------
                    # Top-3 term panels, each with a dropdown to swap that
                    # slot for any other ranked term in term_pool -- the
                    # dropdown defaults to the rank-i term but remembers an
                    # explicit user choice (by term label, which stays
                    # meaningful across reruns even if rankings shift)
                    # across reruns, resetting only if that chosen term is no
                    # longer present in the current term_pool at all.
                    # ------------------------------------------------------
                    n = min(3, len(term_pool))
                    term_labels_all = term_pool["term"].tolist()
                    label_to_idx = dict(zip(term_pool["term"], term_pool.index))
                    term_cols = st.columns(n)
                    selected_rows = []
                    for i, col in enumerate(term_cols):
                        default_label = term_labels_all[i]
                        slot_key = f"top_term_select_{i}"
                        if slot_key in st.session_state and st.session_state[slot_key] not in term_labels_all:
                            del st.session_state[slot_key]  # chosen term no longer available -- fall back to default
                        with col:
                            chosen_label = st.selectbox(
                                f"Term #{i + 1}",
                                term_labels_all,
                                index=term_labels_all.index(default_label) if slot_key not in st.session_state else 0,
                                key=slot_key,
                                format_func=lambda lbl: _pretty_term_text(term_sigs[label_to_idx[lbl]]),
                                help="Swap this slot for a different discovered term/interaction; "
                                     "the new choice is used for the combined plot below and "
                                     "everywhere else downstream (equation building, report, etc.).",
                            )
                        idx = label_to_idx[chosen_label]
                        selected_rows.append((idx, term_pool.loc[idx]))

                    # One column per term, each holding its own small plot with the
                    # term's description directly underneath it (rather than one
                    # combined figure with all descriptions stacked below).
                    for col, (idx, row) in zip(term_cols, selected_rows):
                        sig = term_sigs[idx]
                        term_str = term_expr_string(sig)
                        with col:
                            fig, ax = plt.subplots(figsize=(3.2, _term_fig_h))
                            try:
                                term_fn = make_physics_fn(term_str, plot_feature_names)
                                term_vals = term_fn(plot_data[plot_feature_names].to_numpy(dtype=float))
                                prediction = combine_physics(phys_vals_whole, term_vals, used_combine_mode)
                                _plot_scatter(ax, prediction, row["term"])
                                fig.tight_layout()
                                st.pyplot(fig)
                                plt.close(fig)
                                formula_bits = _combine_formula_str(used_physics_formula, term_str, used_combine_mode)
                                gep_note(
                                    f'<span class="gep-label">{_equation_to_html(term_str)}</span>'
                                    f'n={row["n_solutions"]} ({row["pct_of_solutions"]}%) '
                                    f"— {_equation_to_html(formula_bits)} — <em>{describe_term(sig)}</em>"
                                )
                                good_terms.append((term_str, term_vals, row))
                            except Exception as e:
                                plt.close(fig)
                                st.caption(f"Could not plot term '{row['term']}': {e}")

                # Two more reference plots, side by side in one figure: (1) the
                # injected physics formula on its own, and (2) physics + the
                # top-3 raw terms, each added exactly as-is (no fitting at all,
                # here or in the panels above). Any fitting happens
                # deliberately, later, in the "Equation building" tab.
                fig2, (ax_phys, ax_combo) = plt.subplots(1, 2, figsize=(5.6, _combo_fig_h))
                _plot_scatter(ax_phys, phys_vals_whole, "Physics only")

                combo_formula = None
                if good_terms:
                    # physics (as given) combined with each raw term's sum --
                    # no fitting; physics is only combined once even though
                    # it's part of each panel's own prediction above.
                    terms_sum = sum(term_vals for _, term_vals, _ in good_terms)
                    combo_pred = combine_physics(phys_vals_whole, terms_sum, used_combine_mode)
                    combo_label = "Physics + top 3" if used_combine_mode != "mult" else "Physics × (1 + top 3)"
                    _plot_scatter(ax_combo, combo_pred, combo_label)
                    terms_bits = " + ".join(
                        term_str for term_str, _, _ in good_terms
                    )
                    combo_formula = _combine_formula_str(used_physics_formula, terms_bits, used_combine_mode)
                else:
                    ax_combo.axis("off")
                    ax_combo.text(0.5, 0.5, "no terms\nto combine", ha="center", va="center", fontsize=8)
                fig2.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)
                if used_physics_formula:
                    gep_equation("physics:", _equation_to_html(used_physics_formula))
                else:
                    st.caption("(no physics formula was used)")
                if combo_formula:
                    gep_equation("combo (unfitted):", _equation_to_html(combo_formula))

        # ----------------------------------------------------------------
        # Applicable range tab: pick any of the same predictions shown in
        # "Plots" (physics only / physics + one of the top-3 terms / the
        # combo) and inspect where its predicted-value outliers sit relative
        # to each feature's distribution -- a rough "domain of applicability"
        # check for whether the model is extrapolating on some feature.
        # Filled after "Plots" (in code, though it's shown third/rightmost)
        # since it reuses phys_vals_whole/good_terms/combo_pred computed
        # there, the same way "Plots" reuses "More"'s include_single_vars.
        # ----------------------------------------------------------------
        with tab_range:
            if plot_data is None:
                st.caption("(run a search to see this)")
            else:
                _combo_join = " × (1 + " if used_combine_mode == "mult" else " + "
                _combo_close = ")" if used_combine_mode == "mult" else ""
                model_predictions = {"Physics only": phys_vals_whole}
                for term_str, term_vals, row in good_terms:
                    model_predictions[f"Physics{_combo_join}{row['term']}{_combo_close}"] = combine_physics(
                        phys_vals_whole, term_vals, used_combine_mode
                    )
                if good_terms:
                    model_predictions[f"Physics{_combo_join}top 3 (combo){_combo_close}"] = combo_pred

                model_choice = st.selectbox(
                    "Model to analyze", list(model_predictions), key="applic_range_model",
                    help="Pick which prediction's outliers to inspect -- these match the panels "
                         "in the \"Plots\" tab. Predicted-value outliers are flagged the same way "
                         f"as there: more than {PRED_OUTLIER_N_STD:g} standard deviations from "
                         "that prediction's own mean.",
                )
                prediction = model_predictions[model_choice]

                pred_std = prediction.std()
                pred_outlier_mask = (
                    np.abs(prediction - prediction.mean()) > PRED_OUTLIER_N_STD * pred_std
                    if pred_std and not np.isnan(pred_std) else np.zeros(len(prediction), dtype=bool)
                )
                n_out = int(pred_outlier_mask.sum())

                if n_out == 0:
                    st.caption(
                        f"No predicted-value outliers for '{model_choice}' (nothing more than "
                        f"{PRED_OUTLIER_N_STD:g} standard deviations from its own mean prediction)."
                    )
                else:
                    st.caption(
                        f"{n_out} of {len(prediction)} row(s) "
                        f"({100 * n_out / len(prediction):.1f}%) flagged as predicted-value "
                        f"outliers for '{model_choice}'. Feature distributions and the "
                        "recommended applicable range below are based on this split."
                    )

                    feat_df = plot_data[plot_feature_names]
                    inlier_mask = ~pred_outlier_mask

                    # One combined figure with a small overlaid histogram per
                    # feature (outlier rows vs. the rest), rather than one
                    # st.pyplot call per feature, to keep this compact.
                    n_feat = len(plot_feature_names)
                    ncols = min(4, n_feat)
                    nrows = -(-n_feat // ncols)  # ceil
                    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.2 * nrows), squeeze=False)
                    axes = axes.flatten()
                    for ax, feat in zip(axes, plot_feature_names):
                        vals_in = feat_df.loc[inlier_mask, feat].to_numpy(dtype=float)
                        vals_out = feat_df.loc[pred_outlier_mask, feat].to_numpy(dtype=float)
                        ax.hist(vals_in, bins=15, color="tab:blue", alpha=0.6, label="normal")
                        ax.hist(vals_out, bins=15, color="red", alpha=0.7, label="outlier")
                        ax.set_title(feat, fontsize=8)
                        ax.tick_params(labelsize=6)
                    for ax in axes[n_feat:]:
                        ax.axis("off")
                    axes[0].legend(fontsize=6, framealpha=0.5)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    # Residual (prediction - actual) vs. each feature value --
                    # a static min/max "applicable range" boundary can't show
                    # HOW badly the model does outside it: error might grow
                    # gradually (safe-ish to nudge past the boundary) or fall
                    # off a cliff right at the edge (risky to extrapolate at
                    # all). Same blue/red inlier/outlier coloring as the
                    # histograms above, plus a dashed zero-residual reference
                    # line, so trends are easy to read at a glance.
                    st.markdown("**Prediction error vs. feature value**")
                    st.caption(
                        "Residual (prediction − actual) plotted against each feature, so you can "
                        "see whether error grows smoothly toward the edges of the data or jumps "
                        "sharply -- more informative than the min/max boundary alone for judging "
                        "how risky it is to extrapolate slightly past it."
                    )
                    residual = prediction - y_actual
                    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.2 * nrows), squeeze=False)
                    axes2 = axes2.flatten()
                    for ax, feat in zip(axes2, plot_feature_names):
                        x_in = feat_df.loc[inlier_mask, feat].to_numpy(dtype=float)
                        x_out = feat_df.loc[pred_outlier_mask, feat].to_numpy(dtype=float)
                        r_in = residual[inlier_mask]
                        r_out = residual[pred_outlier_mask]
                        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
                        ax.scatter(x_in, r_in, s=10, color="tab:blue", alpha=0.6, label="normal")
                        ax.scatter(x_out, r_out, s=14, color="red", alpha=0.8, label="outlier")
                        ax.set_title(feat, fontsize=8)
                        ax.tick_params(labelsize=6)
                    for ax in axes2[n_feat:]:
                        ax.axis("off")
                    axes2[0].legend(fontsize=6, framealpha=0.5)
                    fig2.tight_layout()
                    st.pyplot(fig2)
                    plt.close(fig2)

                    # Recommended applicable range: the span of each feature
                    # among the well-predicted (non-outlier) rows -- both the
                    # literal min/max and the 5th-95th percentile band (a bit
                    # more robust to a single stray point) -- next to the
                    # feature's full dataset range and how many outlier rows
                    # fall outside that well-predicted range.
                    range_rows = []
                    for feat in plot_feature_names:
                        full_vals = feat_df[feat].to_numpy(dtype=float)
                        in_vals = feat_df.loc[inlier_mask, feat].to_numpy(dtype=float)
                        out_vals = feat_df.loc[pred_outlier_mask, feat].to_numpy(dtype=float)
                        if len(in_vals) == 0:
                            continue
                        lo, hi = float(in_vals.min()), float(in_vals.max())
                        p5, p95 = float(np.percentile(in_vals, 5)), float(np.percentile(in_vals, 95))
                        n_out_outside = int(((out_vals < lo) | (out_vals > hi)).sum()) if len(out_vals) else 0
                        range_rows.append({
                            "feature": feat,
                            "full data range": f"{full_vals.min():.3g} to {full_vals.max():.3g}",
                            "applicable range (min-max)": f"{lo:.3g} to {hi:.3g}",
                            "applicable range (5th-95th pct)": f"{p5:.3g} to {p95:.3g}",
                            "outliers outside min-max range": n_out_outside,
                        })
                    st.dataframe(pd.DataFrame(range_rows), hide_index=True, width="stretch")
                    st.caption(
                        "\"Applicable range\" is the span of each feature among the rows this "
                        "model predicted well (not flagged as predicted-value outliers) -- a rough "
                        "guide to where the model can be trusted. \"Outliers outside min-max range\" "
                        "counts how many flagged outliers fall outside that range for a given "
                        "feature -- a higher count suggests the model may be extrapolating on that "
                        "feature for those points."
                    )

    with tab_equation:
        if plot_data is None:
            st.caption("(run a search to see the equation tools)")
        else:
            tab_fit, tab_build, tab_report = st.tabs(
                ["Weighted regression", "Build your own equation", "Report"]
            )

            # ------------------------------------------------------------
            # Weighted regression tab: an honest, held-out-tested version of
            # the combined model shown in the middle pane. Instead of
            # least-squares-fitting all terms against the whole dataset,
            # split into train/test, fit a proper (weighted) linear
            # regression of physics + the top interaction terms on the
            # train split only, then report the weighted equation and
            # evaluate/plot it on data it wasn't fit on.
            # ------------------------------------------------------------
            with tab_fit:
                if not good_terms:
                    st.caption("(no terms available to fit)")
                else:
                    # Same controls-on-the-left, plot-on-the-right layout as
                    # the "Build your own equation" tab: everything that's
                    # computed from the controls lives in controls_col, and
                    # the plot (drawn from those results) lives in plot_col,
                    # placed after controls_col's block so it can see the
                    # variables computed there.
                    controls_col, plot_col = st.columns([1, 1.2], gap="large")

                    with controls_col:
                        train_frac = st.slider(
                            "Training fraction for this regression", min_value=0.1, max_value=0.9, value=0.7, step=0.05,
                            help="Fraction of rows used to fit the weights below; the rest are held out to evaluate the equation.",
                        )
                        method = st.radio(
                            "Weighting algorithm", ["Linear (OLS)", "Ridge (L2-regularized)"], horizontal=True,
                            help=(
                                "OLS is the plain, most interpretable least-squares fit. Ridge shrinks the "
                                "weights (via an L2 penalty) which can help when the physics term and the "
                                "GEP terms are correlated with each other, at a small cost to interpretability."
                            ),
                        )
                        ridge_alpha = st.number_input(
                            "Ridge alpha (regularization strength)", min_value=0.0, value=1.0, step=0.5,
                            disabled=(method != "Ridge (L2-regularized)"),
                        ) if method == "Ridge (L2-regularized)" else None

                        regressors = []
                        regressor_labels = []
                        if used_physics_formula:
                            regressors.append(phys_vals_whole)
                            regressor_labels.append(f"({used_physics_formula})")
                        for term_str, tv, _ in good_terms:
                            regressors.append(tv)
                            regressor_labels.append(f"({term_str})")

                        if used_combine_mode == "mult":
                            # physics × (1 + Σ wi·termi) is bilinear in the weights (expanding
                            # it mixes physics's own weight together with each wi), so ordinary
                            # least squares can't solve for a free physics weight here -- fix it
                            # at a multiplier of 1 and fit only the term weights inside the
                            # (1 + ...) correction. Physics still gets its own adjustable weight
                            # (starting at 1x) below in "Build your own equation".
                            term_labels_fit = [f"({term_str})" for term_str, _, _ in good_terms]
                            term_cols_fit = [tv for _, tv, _ in good_terms]
                            X = np.column_stack(term_cols_fit) if term_cols_fit else np.zeros((len(y_actual), 0))
                            eps_wr = physics_clip_eps(y_actual)
                            phys_clipped_wr, n_clip_wr = clip_physics_for_mult(phys_vals_whole, eps_wr)
                            if n_clip_wr:
                                st.caption(
                                    f"⚠ {n_clip_wr} point(s) have a physics(X) value very close to "
                                    f"zero; clipped to ±{eps_wr:.3g} before dividing to fit the "
                                    f"fractional correction below."
                                )
                            z_target = y_actual / phys_clipped_wr - 1.0
                            (X_train, X_test, z_train, z_test, y_train, y_test,
                             phys_train_wr, phys_test_wr) = train_test_split(
                                X, z_target, y_actual, phys_clipped_wr, train_size=train_frac, random_state=42,
                            )
                            if method == "Ridge (L2-regularized)":
                                reg = Ridge(alpha=ridge_alpha, fit_intercept=False).fit(X_train, z_train)
                            else:
                                reg = LinearRegression(fit_intercept=False).fit(X_train, z_train)
                            pred_train = phys_train_wr * (1 + reg.predict(X_train))
                            pred_test = phys_test_wr * (1 + reg.predict(X_test))
                            train_r2 = r2_score(y_train, pred_train)
                            test_r2 = r2_score(y_test, pred_test)

                            correction_bits = " + ".join(
                                f"{c:.3g}×{lbl}" for c, lbl in zip(reg.coef_, term_labels_fit)
                            ) or "0"
                            equation = _combine_formula_str(used_physics_formula, correction_bits, "mult")
                            build_seed_weights = ([1.0] if used_physics_formula else []) + list(reg.coef_)
                        else:
                            X = np.column_stack(regressors)
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y_actual, train_size=train_frac, random_state=42,
                            )
                            # No intercept: the equation should be pure weights on physics + the
                            # terms, nothing else.
                            if method == "Ridge (L2-regularized)":
                                reg = Ridge(alpha=ridge_alpha, fit_intercept=False).fit(X_train, y_train)
                            else:
                                reg = LinearRegression(fit_intercept=False).fit(X_train, y_train)
                            train_r2 = reg.score(X_train, y_train)
                            test_r2 = reg.score(X_test, y_test)
                            pred_train = reg.predict(X_train)
                            pred_test = reg.predict(X_test)
                            equation = " + ".join(f"{c:.3g}×{lbl}" for c, lbl in zip(reg.coef_, regressor_labels))
                            build_seed_weights = list(reg.coef_)

                        gep_equation(
                            f"Fitted equation ({train_frac:.0%} train split, {method}):<br>",
                            _equation_to_html(equation),
                        )
                        st.caption(f"train R² = {train_r2:.3f} (n={len(y_train)}), test R² = {test_r2:.3f} (n={len(y_test)}, held out)")

                    with plot_col:
                        fig, ax = plt.subplots(figsize=(4, 3.2))
                        ax.scatter(y_train, pred_train, alpha=0.6, edgecolors="none", s=14, label=f"train (R²={train_r2:.3f})")
                        ax.scatter(y_test, pred_test, alpha=0.8, edgecolors="none", marker="^", s=14, label=f"test (R²={test_r2:.3f})")
                        lo = min(y_actual.min(), pred_train.min(), pred_test.min())
                        hi = max(y_actual.max(), pred_train.max(), pred_test.max())
                        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
                        ax.set_xlabel(f"Actual {plot_target_name}", fontsize=8)
                        ax.set_ylabel("Predicted", fontsize=8)
                        ax.set_title(
                            "Weighted physics × (1 + top terms)" if used_combine_mode == "mult"
                            else "Weighted physics + top terms", fontsize=9,
                        )
                        ax.tick_params(labelsize=7)
                        ax.legend(fontsize=7)
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

            # ------------------------------------------------------------
            # Build-your-own-equation tab: let the engineer hand-tune the
            # fitted weights, add a conservatism bias, apply an overall
            # scale factor, drop terms, or round to clean numbers -- then
            # see the resulting equation, R², % conservative, and plot
            # update live. Less-common controls (reset/round shortcuts,
            # auto-bias) are tucked into collapsed expanders to keep the
            # main flow compact.
            # ------------------------------------------------------------
            with tab_build:
                if not good_terms:
                    st.caption("(no terms available to fit)")
                else:
                    weight_keys = [f"eqw_{i}" for i in range(len(regressor_labels))]
                    bias_key, scale_key = "eqw_bias", "eqw_scale"
                    dynamic_key, scale_lo_key, scale_hi_key = "eqw_dynamic_scale", "eqw_scale_lo", "eqw_scale_hi"

                    # Controls on the left, resulting plot on the right.
                    controls_col, plot_col = st.columns([1, 1.2], gap="large")

                    with controls_col:
                        # Buttons that mutate weight/bias/scale state must run (and be
                        # defined) before the widgets bound to those keys are instantiated
                        # further down, so they stay right at the top.
                        with st.expander("Shortcuts: reset / round weights"):
                            btn_reset, btn_round = st.columns(2)
                            if btn_reset.button("Reset to fitted"):
                                # build_seed_weights (from the Weighted regression tab) is one
                                # weight per regressor_labels entry in both combine modes --
                                # physics included at 1.0x in "mult" mode (since the weighted
                                # fit there can't solve for a free physics weight; see the
                                # comment where it's built) -- unlike reg.coef_ alone, which in
                                # "mult" mode only covers the GEP terms and is one short.
                                for k, c in zip(weight_keys, build_seed_weights):
                                    st.session_state[k] = float(c)
                                st.session_state[bias_key] = 0.0
                                st.session_state[scale_key] = 1.0
                                st.session_state[dynamic_key] = False
                                st.session_state[scale_lo_key] = 1.0
                                st.session_state[scale_hi_key] = 1.0
                            if btn_round.button("Round to 2 sig figs"):
                                for k in weight_keys:
                                    st.session_state[k] = float(f"{st.session_state.get(k, 0.0):.2g}")
                                if bias_key in st.session_state and st.session_state[bias_key]:
                                    st.session_state[bias_key] = float(f"{st.session_state[bias_key]:.2g}")
                                for k in (scale_lo_key, scale_hi_key):
                                    if k in st.session_state:
                                        st.session_state[k] = float(f"{st.session_state[k]:.2g}")

                        for k, c in zip(weight_keys, build_seed_weights):
                            st.session_state.setdefault(k, float(c))
                        st.session_state.setdefault(bias_key, 0.0)
                        st.session_state.setdefault(scale_key, 1.0)
                        st.session_state.setdefault(dynamic_key, False)
                        st.session_state.setdefault(scale_lo_key, 1.0)
                        st.session_state.setdefault(scale_hi_key, 1.0)

                        has_physics = bool(used_physics_formula)
                        mult_mode = used_combine_mode == "mult" and has_physics

                        with st.container(border=True):
                            st.write("**Term weights** (uncheck to drop a term):")
                            if mult_mode:
                                st.caption(
                                    "Multiplicative mode: physics's weight is a multiplier out front, "
                                    "and the GEP terms' weights combine inside a (1 + …) fractional "
                                    "correction that then also multiplies physics -- i.e. "
                                    "*physics_weight*×physics × (1 + Σ *term_weight*×term), not a "
                                    "plain sum of all of them."
                                )
                            # Collapse each widget's own label (which would otherwise render on
                            # its own line above the field, roughly doubling the height of every
                            # row) and instead show the term as compact inline text vertically
                            # centered next to the checkbox/input -- one row per term instead of
                            # two, so this section takes noticeably less vertical space.
                            weights = []
                            for lbl, key in zip(regressor_labels, weight_keys):
                                wc1, wc2, wc3 = st.columns([0.5, 2.3, 1.4], gap="small")
                                with wc1:
                                    include = st.checkbox(
                                        "on", value=True, key=f"{key}_incl", label_visibility="collapsed"
                                    )
                                with wc2:
                                    st.markdown(
                                        f'<div style="padding-top:0.55rem; font-size:0.97rem; '
                                        f'white-space:nowrap; overflow:hidden; text-overflow:ellipsis;" '
                                        f'title="{lbl}">{lbl}</div>',
                                        unsafe_allow_html=True,
                                    )
                                with wc3:
                                    w = st.number_input(
                                        lbl, step=0.1, format="%.4g", key=key, disabled=not include,
                                        label_visibility="collapsed",
                                    )
                                weights.append(w if include else 0.0)

                            A_custom = np.column_stack(regressors)
                            weights_arr = np.array(weights)
                            if mult_mode:
                                # physics(X) is regressors[0]/weights[0] (see how `regressors` is
                                # built above, in the Weighted regression tab); it multiplies the
                                # whole thing rather than being summed in with the terms.
                                phys_w = weights_arr[0]
                                term_w = weights_arr[1:]
                                term_vals_matrix = A_custom[:, 1:]
                                raw_pred = phys_w * regressors[0] * (1 + term_vals_matrix @ term_w)
                            else:
                                # Additive mode -- also the fallback for the discouraged "multiplicative
                                # with no physics formula" combo (physics(X) is 0 there, which would
                                # make a true multiplicative combination always 0 too; summing the
                                # (then physics-less) terms directly is more useful than that).
                                raw_pred = A_custom @ weights_arr

                            def _current_scale_array():
                                # The scale factor(s) as currently set, as a plain array (one entry
                                # per row) -- read straight from session_state (mirroring the
                                # scale-factor box further down) so it's available to the refit
                                # buttons and quick auto-set buttons above/below before that box
                                # has actually run this rerun.
                                if st.session_state.get(dynamic_key, False):
                                    lo_ = st.session_state.get(scale_lo_key, 1.0)
                                    hi_ = st.session_state.get(scale_hi_key, 1.0)
                                    rp_min_, rp_max_ = float(raw_pred.min()), float(raw_pred.max())
                                    span_ = max(rp_max_ - rp_min_, 1e-9)
                                    frac_ = (raw_pred - rp_min_) / span_
                                    return lo_ + (hi_ - lo_) * frac_
                                return np.full_like(raw_pred, st.session_state.get(scale_key, 1.0))

                            def _scaled_pred_from_state():
                                return _current_scale_array() * raw_pred

                            def _term_refit_pieces():
                                # Shared setup for the two "refit terms" buttons below: express
                                # custom_pred as offset + multiplier * (on-term-matrix @ term_w),
                                # holding physics's own weight, the scale factor(s), and the bias
                                # exactly as currently set -- so solving for term_w alone (via
                                # ordinary or quantile least squares) refits only the on terms,
                                # without disturbing anything else. Terms unchecked ("off") are
                                # left out of the refit entirely (and out of the resulting
                                # prediction, same as everywhere else in this tab).
                                term_range = range(1, len(regressors)) if has_physics else range(len(regressors))
                                on_idx = [i for i in term_range if st.session_state.get(f"{weight_keys[i]}_incl", True)]
                                if not on_idx:
                                    return None, None, None
                                term_matrix_on = np.column_stack([regressors[i] for i in on_idx])
                                phys_val = regressors[0] if has_physics else np.zeros_like(y_actual)
                                phys_w_now = weights[0] if has_physics else 0.0
                                bias_now = st.session_state.get(bias_key, 0.0)
                                scale_arr = _current_scale_array()
                                offset = scale_arr * phys_w_now * phys_val + bias_now
                                multiplier = scale_arr * phys_w_now * phys_val if mult_mode else scale_arr
                                eff_matrix = multiplier[:, None] * term_matrix_on
                                target = y_actual - offset
                                return on_idx, eff_matrix, target

                            st.write(
                                "Refit just the on term weights above to best match the actual "
                                "values -- physics's own weight, the scale factor(s), and the bias "
                                "stay exactly as set below."
                            )
                            rf1_col, rf2_col = st.columns(2)
                            with rf1_col:
                                if st.button(
                                    "Refit terms: maximize R²", width="stretch",
                                    help="Least-squares refit of the on term weights only (physics's "
                                         "weight, scale, and bias untouched) -- the exact weights "
                                         "that minimize squared error against the actual values.",
                                ):
                                    on_idx, eff_matrix, target = _term_refit_pieces()
                                    if on_idx is None:
                                        st.warning("No terms are currently turned on to refit.")
                                    else:
                                        new_w, *_ = np.linalg.lstsq(eff_matrix, target, rcond=None)
                                        for idx, w in zip(on_idx, new_w):
                                            st.session_state[weight_keys[idx]] = float(w)
                            with rf2_col:
                                if st.button(
                                    "Refit terms: focus on conservatism", width="stretch",
                                    help="Quantile refit of the on term weights only (physics's "
                                         "weight, scale, and bias untouched), targeting the % "
                                         "conservative set in the Target conservatism box below -- "
                                         "instead of the tightest overall fit, this leans toward "
                                         "predictions that stay at or under the actual value that often.",
                                ):
                                    on_idx, eff_matrix, target = _term_refit_pieces()
                                    if on_idx is None:
                                        st.warning("No terms are currently turned on to refit.")
                                    else:
                                        target_pct_now = st.session_state.get("eqw_target_pct", 90)
                                        q = float(np.clip(1 - target_pct_now / 100.0, 0.01, 0.99))
                                        qr = QuantileRegressor(
                                            quantile=q, alpha=0.0, fit_intercept=False, solver="highs",
                                        )
                                        qr.fit(eff_matrix, target)
                                        for idx, w in zip(on_idx, qr.coef_):
                                            st.session_state[weight_keys[idx]] = float(w)

                        # Shared grid-search helper for the auto-balance-scale and quick
                        # auto-set-scale buttons: sweep a range of scale factors (holding the
                        # current bias fixed), and report the resulting % conservative and R²
                        # at each candidate -- vectorized over the whole dataset at once since
                        # the grid is small. (Only depends on raw_pred / session_state, not on
                        # any widget's return value, so it's fine to call this from the quick
                        # auto-set buttons below, before the detailed scale-factor box further
                        # down has actually run this rerun.)
                        def _scale_grid_stats(candidates):
                            bias_now = st.session_state.get(bias_key, 0.0)
                            preds_grid = candidates[:, None] * raw_pred[None, :] + bias_now
                            pct_cons_grid = 100 * np.mean(preds_grid <= y_actual[None, :], axis=1)
                            ss_res_grid = np.sum((y_actual[None, :] - preds_grid) ** 2, axis=1)
                            ss_tot = np.sum((y_actual - y_actual.mean()) ** 2)
                            r2_grid = 1 - ss_res_grid / max(ss_tot, 1e-9)
                            return pct_cons_grid, r2_grid

                        with st.container(border=True):
                            st.write("**Target conservatism level** (used by the auto-set buttons below):")
                            tp_lbl, tp_val = st.columns([2.5, 1.2])
                            with tp_lbl:
                                _label_beside_input("Target: % of points where prediction ≤ actual")
                            with tp_val:
                                target_pct = st.number_input(
                                    "Target: % of points where prediction ≤ actual", min_value=50, max_value=99,
                                    value=90, step=1, key="eqw_target_pct", label_visibility="collapsed",
                                    help="Used by the auto-set scale/bias buttons below to pick a value so the "
                                         "predicted value is at or below the actual value for at least this "
                                         "fraction of the dataset -- a common way to pick a conservative design value.",
                                )

                            as_col, ab_col = st.columns(2)
                            with as_col:
                                if st.button(
                                    "Auto-set scale for this %", width="stretch",
                                    help="Uniformly reweights all terms (via the scale factor below) so this "
                                         "many points come out at or under the actual value.",
                                ):
                                    candidates = np.linspace(0.1, 2.0, 381)
                                    pct_cons_grid, _ = _scale_grid_stats(candidates)
                                    best_idx = int(np.argmin(np.abs(pct_cons_grid - target_pct)))
                                    st.session_state[scale_key] = float(candidates[best_idx])
                            with ab_col:
                                if st.button(
                                    "Auto-set bias for this %", width="stretch",
                                    help="Shifts every prediction up or down by a constant (via the bias below) "
                                         "so this many points come out at or under the actual value.",
                                ):
                                    resid_for_bias = y_actual - _scaled_pred_from_state()
                                    st.session_state[bias_key] = float(
                                        np.percentile(resid_for_bias, 100 - target_pct)
                                    )

                            st.write("Or, ignore the target % above and optimize for pure accuracy:")
                            as2_col, ab2_col = st.columns(2)
                            with as2_col:
                                if st.button(
                                    "Scale to maximize R²", width="stretch",
                                    help="Closed-form best-fit scale factor (holding the current bias fixed) "
                                         "-- the value of the scale factor below that minimizes squared error "
                                         "against the actual values, clamped to its 0.1-2.0 range.",
                                ):
                                    bias_now = st.session_state.get(bias_key, 0.0)
                                    denom = float(np.dot(raw_pred, raw_pred))
                                    if denom > 1e-12:
                                        best_scale = float(np.dot(raw_pred, y_actual - bias_now) / denom)
                                        st.session_state[scale_key] = float(np.clip(best_scale, 0.1, 2.0))
                            with ab2_col:
                                if st.button(
                                    "Bias to maximize R²", width="stretch",
                                    help="Closed-form best-fit bias (holding the current scale fixed) -- the "
                                         "mean remaining error, which is the value of the bias below that "
                                         "minimizes squared error against the actual values.",
                                ):
                                    resid_for_r2 = y_actual - _scaled_pred_from_state()
                                    st.session_state[bias_key] = float(resid_for_r2.mean())

                        st.caption(
                            "The scale-factor and bias sections below are for fine-tuning those "
                            "values by hand (or using the balance/dynamic-scale options) if you "
                            "want something other than the auto-set result above."
                        )

                        # Everything to do with the overall scale factor lives in one bordered
                        # box, in the order: balance R² against conservatism -> the
                        # dynamic-vs-fixed toggle -> the actual value(s).
                        # Read the dynamic-scale toggle's *current* value from session_state
                        # before the checkbox widget itself is drawn (its value doesn't change
                        # again mid-run once a rerun starts, so this is safe) so the balance
                        # controls -- which only apply to the fixed-scale case -- can be shown
                        # above the checkbox that gates them, matching the requested order.
                        with st.container(border=True):
                            st.write("**Scale factor**")
                            dynamic_scale_now = st.session_state.get(dynamic_key, False)

                            if not dynamic_scale_now:
                                st.write("Balance overall accuracy (R²) against conservatism:")
                                bal_lbl, bal_val = st.columns([2.5, 1.2])
                                with bal_lbl:
                                    _label_beside_input("Balance: favor R² (0.0) vs favor conservatism (1.0)")
                                with bal_val:
                                    balance = st.number_input(
                                        "Balance: favor R² (0.0) vs favor conservatism (1.0)",
                                        min_value=0.0, max_value=1.0, value=0.5, step=0.05, format="%.2f",
                                        key="eqw_balance", label_visibility="collapsed",
                                        help="Searches scale factors and picks the one that best trades off "
                                             "overall R² (whole dataset) against % conservative, weighted by "
                                             "this slider -- 0 favors the most accurate fit, 1 favors the most "
                                             "conservative one.",
                                    )
                                if st.button(
                                    "Auto-balance scale (R² vs conservatism)",
                                    help="Sets the scale factor below using the balance slider above, instead "
                                         "of targeting a specific % conservative directly.",
                                ):
                                    candidates = np.linspace(0.1, 2.0, 381)
                                    pct_cons_grid, r2_grid = _scale_grid_stats(candidates)
                                    r2_norm = np.clip(r2_grid / max(float(r2_grid.max()), 1e-9), 0, None)
                                    score = balance * (pct_cons_grid / 100.0) + (1 - balance) * r2_norm
                                    best_idx = int(np.argmax(score))
                                    st.session_state[scale_key] = float(candidates[best_idx])

                            dynamic_scale = st.checkbox(
                                "Dynamic scale factor (varies with predicted magnitude)", key=dynamic_key,
                                help=(
                                    "Instead of one fixed multiplier, linearly interpolate the scale factor "
                                    "between the two values below, based on where each row's raw prediction "
                                    "falls between the lowest and highest predictions in this dataset. Set "
                                    "'high' lower than 'low' to weight larger predictions less (more "
                                    "conservative for bigger capacities), or higher than 'low' to weight them more."
                                ),
                            )
                            if not dynamic_scale:
                                sc_lbl, sc_val = st.columns([2.5, 1.2])
                                with sc_lbl:
                                    _label_beside_input(
                                        "Scale factor (multiplies the whole weighted sum, e.g. a strength-reduction factor)"
                                    )
                                with sc_val:
                                    scale = st.number_input(
                                        "Scale factor (multiplies the whole weighted sum, e.g. a strength-reduction factor)",
                                        min_value=0.1, max_value=2.0, step=0.05, format="%.2f", key=scale_key,
                                        label_visibility="collapsed",
                                        help="Multiplies the whole weighted sum by this single fixed number -- "
                                             "e.g. 0.9 knocks every prediction down 10% (a strength-reduction "
                                             "factor); 1.0 leaves it unchanged. Use the auto-set/auto-balance "
                                             "buttons above to fill this in automatically, or type your own.",
                                    )
                                scaled_pred = scale * raw_pred
                                scale_desc = None
                            else:
                                lo_lbl, lo_val = st.columns([2.5, 1.2])
                                with lo_lbl:
                                    _label_beside_input("Scale @ lowest prediction")
                                with lo_val:
                                    scale_at_low = st.number_input(
                                        "Scale @ lowest prediction", min_value=0.1, max_value=2.0, step=0.05,
                                        format="%.2f", key=scale_lo_key, label_visibility="collapsed",
                                        help="The multiplier applied at the smallest raw prediction in this "
                                             "dataset. Predictions in between are interpolated linearly between "
                                             "this and the 'highest prediction' value below.",
                                    )
                                hi_lbl, hi_val = st.columns([2.5, 1.2])
                                with hi_lbl:
                                    _label_beside_input("Scale @ highest prediction")
                                with hi_val:
                                    scale_at_high = st.number_input(
                                        "Scale @ highest prediction", min_value=0.1, max_value=2.0, step=0.05,
                                        format="%.2f", key=scale_hi_key, label_visibility="collapsed",
                                        help="The multiplier applied at the largest raw prediction in this "
                                             "dataset. Set this lower than 'lowest prediction' above to weight "
                                             "bigger predictions less (more conservative for larger capacities), "
                                             "or higher to weight them more.",
                                    )
                                rp_min, rp_max = float(raw_pred.min()), float(raw_pred.max())
                                span = max(rp_max - rp_min, 1e-9)
                                frac = (raw_pred - rp_min) / span  # 0 at the lowest prediction, 1 at the highest
                                dyn_scale_arr = scale_at_low + (scale_at_high - scale_at_low) * frac
                                scaled_pred = dyn_scale_arr * raw_pred
                                # Express the same interpolation as an affine function of the raw
                                # prediction (scale(x) = A + B·x) so it can be written as one equation.
                                slope_b = (scale_at_high - scale_at_low) / span
                                intercept_a = scale_at_low - slope_b * rp_min
                                scale_desc = (intercept_a, slope_b)

                        # Bias controls get their own bordered box, separate from the scale
                        # options above, so the two groups read as distinct steps.
                        with st.container(border=True):
                            st.write("**Bias / residual shift**")
                            bias_range = max(1.0, float(np.abs(y_actual - scaled_pred).max()))
                            # Clamp in case a previous run's bias (e.g. from auto-set, or before
                            # dropping a term/changing weights) now falls outside this run's range.
                            st.session_state[bias_key] = float(
                                np.clip(st.session_state[bias_key], -bias_range, bias_range)
                            )
                            bias_lbl, bias_val = st.columns([2.5, 1.2])
                            with bias_lbl:
                                _label_beside_input(
                                    "Residual / bias shift (additive -- moves every prediction up or down)"
                                )
                            with bias_val:
                                bias = st.number_input(
                                    "Residual / bias shift (additive -- moves every prediction up or down)",
                                    min_value=-bias_range, max_value=bias_range, step=max(bias_range / 100, 1e-6),
                                    format="%.4g", key=bias_key, label_visibility="collapsed",
                                    help="Added to every prediction after the scale factor is applied -- a "
                                         "negative value shifts everything down (more conservative), a positive "
                                         "value shifts it up. Use the auto-set button above to fill this in "
                                         "automatically, or type your own.",
                                )

                        custom_pred = scaled_pred + bias
                        custom_r2 = r2_score(y_actual, custom_pred)
                        pct_conservative = 100 * float(np.mean(custom_pred <= y_actual))
                        mean_err = float(np.mean(custom_pred - y_actual))

                        if mult_mode:
                            phys_w = weights[0]
                            term_bits = " + ".join(
                                f"{w:.3g}×{lbl}" for w, lbl in zip(weights[1:], regressor_labels[1:]) if w != 0
                            ) or "0"
                            phys_bit = regressor_labels[0] if phys_w == 1 else f"{phys_w:.3g}×{regressor_labels[0]}"
                            core_expr = f"{phys_bit} × (1 + {term_bits})"
                        else:
                            core_expr = " + ".join(
                                f"{w:.3g}×{lbl}" for w, lbl in zip(weights, regressor_labels) if w != 0
                            ) or "0"
                        if scale_desc is not None:
                            intercept_a, slope_b = scale_desc
                            raw_label = f"({core_expr})"
                            scale_fn_str = f"[{intercept_a:.3g} {'+' if slope_b >= 0 else '-'} {abs(slope_b):.3g}×{raw_label}]"
                            custom_equation = f"{scale_fn_str} × {raw_label}"
                        else:
                            custom_equation = f"{scale:.3g} × ({core_expr})" if scale != 1 else core_expr
                        if bias:
                            custom_equation += f" + {bias:.3g}"

                    with plot_col:
                        m1, m2, m3 = st.columns(3)
                        m1.metric("R² (whole dataset)", f"{custom_r2:.3f}")
                        m2.metric("% conservative (pred ≤ actual)", f"{pct_conservative:.0f}%")
                        m3.metric("Mean error (pred − actual)", f"{mean_err:.3g}")

                        fig, ax = plt.subplots(figsize=(4.5, 3.6))
                        conservative_mask = custom_pred <= y_actual
                        ax.scatter(
                            y_actual[conservative_mask], custom_pred[conservative_mask],
                            alpha=0.6, edgecolors="none", s=14, color="tab:green", label="conservative",
                        )
                        ax.scatter(
                            y_actual[~conservative_mask], custom_pred[~conservative_mask],
                            alpha=0.6, edgecolors="none", s=14, color="tab:red", label="unconservative",
                        )
                        lo = min(y_actual.min(), custom_pred.min())
                        hi = max(y_actual.max(), custom_pred.max())
                        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
                        ax.set_xlabel(f"Actual {plot_target_name}", fontsize=8)
                        ax.set_ylabel("Predicted (custom)", fontsize=8)
                        ax.set_title(f"R² = {custom_r2:.3f}, {pct_conservative:.0f}% conservative", fontsize=9)
                        ax.tick_params(labelsize=7)
                        ax.legend(fontsize=7)
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                        gep_equation("Your custom equation:", _equation_to_html(custom_equation))
                        if scale_desc is not None:
                            st.caption(
                                f"Dynamic scale: {scale_at_low:.3g}× at the lowest prediction in this dataset "
                                f"({rp_min:.3g}) to {scale_at_high:.3g}× at the highest ({rp_max:.3g}), interpolated linearly."
                            )

                        st.download_button(
                            "Download custom equation (txt)", custom_equation,
                            file_name="custom_equation.txt", mime="text/plain",
                        )

            # ------------------------------------------------------------
            # Report tab: an on-demand (button-triggered, never automatic)
            # export of a self-contained, downloadable HTML summary --
            # physics formula, custom equation, physics-vs-fitted stats,
            # top-term frequency, and a true-vs-predicted comparison
            # (plot + full data table). Nothing here is computed just from
            # visiting the tab; it's only (re)built when "Generate report"
            # is clicked, and the result is cached in session_state so it
            # survives reruns from other widgets until regenerated.
            # ------------------------------------------------------------
            with tab_report:
                if not good_terms:
                    st.caption("(no terms available to report on)")
                else:
                    st.caption(
                        "Generates a downloadable, self-contained HTML summary of everything found "
                        "above: the physics formula, your custom equation from the 'Build your own "
                        "equation' tab, how often the top terms showed up across recorded solutions, "
                        "physics-only vs. fitted-equation statistics, and a true-vs-predicted "
                        "comparison (plot + full data table). Nothing is generated until you click "
                        "the button below."
                    )
                    if st.button("Generate report", type="primary", key="report_generate_btn"):
                        phys_stats = _regression_stats(y_actual, phys_vals_whole)
                        fit_stats = _regression_stats(y_actual, custom_pred)

                        fig_r, (ax_p, ax_f) = plt.subplots(1, 2, figsize=(8, 3.6))
                        ax_p.scatter(y_actual, phys_vals_whole, alpha=0.6, edgecolors="none", s=14, color="tab:blue")
                        lo = min(y_actual.min(), phys_vals_whole.min())
                        hi = max(y_actual.max(), phys_vals_whole.max())
                        ax_p.plot([lo, hi], [lo, hi], "k--", linewidth=1)
                        ax_p.set_xlabel(f"Actual {plot_target_name}", fontsize=9)
                        ax_p.set_ylabel("Predicted", fontsize=9)
                        ax_p.set_title(f"Physics only\nR² = {phys_stats['R2']:.3f}", fontsize=10)
                        ax_p.tick_params(labelsize=8)

                        cons_mask_r = custom_pred <= y_actual
                        ax_f.scatter(
                            y_actual[cons_mask_r], custom_pred[cons_mask_r],
                            alpha=0.6, edgecolors="none", s=14, color="tab:green", label="conservative",
                        )
                        ax_f.scatter(
                            y_actual[~cons_mask_r], custom_pred[~cons_mask_r],
                            alpha=0.6, edgecolors="none", s=14, color="tab:red", label="unconservative",
                        )
                        lo2 = min(y_actual.min(), custom_pred.min())
                        hi2 = max(y_actual.max(), custom_pred.max())
                        ax_f.plot([lo2, hi2], [lo2, hi2], "k--", linewidth=1)
                        ax_f.set_xlabel(f"Actual {plot_target_name}", fontsize=9)
                        ax_f.set_ylabel("Predicted (custom)", fontsize=9)
                        ax_f.set_title(f"Fitted equation\nR² = {fit_stats['R2']:.3f}", fontsize=10)
                        ax_f.tick_params(labelsize=8)
                        ax_f.legend(fontsize=8)
                        fig_r.tight_layout()
                        plot_b64 = _fig_to_base64(fig_r)
                        plt.close(fig_r)

                        top_terms_info = []
                        for term_str, _, row in good_terms:
                            sig = term_sigs[row.name]
                            top_terms_info.append({
                                "term": row["term"],
                                "n_solutions": int(row["n_solutions"]),
                                "pct_of_solutions": row["pct_of_solutions"],
                                "description": describe_term(sig),
                            })

                        # One standalone plot + data table per individual top
                        # term, i.e. "physics + just this one term" on its own
                        # (not combined with the others, unlike the fitted
                        # equation above) -- so a reader can see each term's
                        # contribution in isolation.
                        term_plots_info = []
                        for term_str, term_vals, row in good_terms:
                            sig = term_sigs[row.name]
                            term_pred = phys_vals_whole + term_vals
                            stats_t = _regression_stats(y_actual, term_pred)

                            fig_t, ax_t = plt.subplots(figsize=(4, 3.6))
                            ax_t.scatter(y_actual, term_pred, alpha=0.6, edgecolors="none", s=14, color="tab:purple")
                            lo_t = min(y_actual.min(), term_pred.min())
                            hi_t = max(y_actual.max(), term_pred.max())
                            ax_t.plot([lo_t, hi_t], [lo_t, hi_t], "k--", linewidth=1)
                            ax_t.set_xlabel(f"Actual {plot_target_name}", fontsize=9)
                            ax_t.set_ylabel("Predicted", fontsize=9)
                            ax_t.set_title(f"Physics + {row['term']}\nR² = {stats_t['R2']:.3f}", fontsize=10)
                            ax_t.tick_params(labelsize=8)
                            fig_t.tight_layout()
                            term_plot_b64 = _fig_to_base64(fig_t)
                            plt.close(fig_t)

                            term_data_table = pd.DataFrame({
                                f"Actual ({plot_target_name})": y_actual,
                                f"Physics + {row['term']} prediction": term_pred,
                                "Residual (prediction - actual)": term_pred - y_actual,
                            })

                            term_plots_info.append({
                                "term": row["term"],
                                "description": describe_term(sig),
                                "stats": stats_t,
                                "plot_b64": term_plot_b64,
                                "data_table_html": term_data_table.to_html(
                                    index=False, float_format=lambda x: f"{x:.4g}"
                                ),
                            })

                        data_table = pd.DataFrame({
                            f"Actual ({plot_target_name})": y_actual,
                            "Physics-only prediction": phys_vals_whole,
                            "Fitted-equation prediction": custom_pred,
                            "Residual (fitted - actual)": custom_pred - y_actual,
                        })

                        report_html = _build_report_html(
                            physics_formula=used_physics_formula,
                            known_formula_name=used_known_formula_name,
                            combo_formula=combo_formula,
                            custom_equation=custom_equation,
                            phys_stats=phys_stats,
                            fit_stats=fit_stats,
                            plot_b64=plot_b64,
                            top_terms_info=top_terms_info,
                            term_plots_info=term_plots_info,
                            data_table=data_table,
                            target_name=plot_target_name,
                            dataset_label=used_dataset_label,
                            feature_names=plot_feature_names,
                            hyperparams=used_hyperparams,
                        )
                        st.session_state["report_html"] = report_html
                        st.session_state["report_filename"] = (
                            f"gep_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                        )

                    if "report_html" in st.session_state:
                        st.success("Report ready.")
                        st.download_button(
                            "Download report (HTML)",
                            data=st.session_state["report_html"],
                            file_name=st.session_state["report_filename"],
                            mime="text/html",
                            key="report_download_btn",
                        )
