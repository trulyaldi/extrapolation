"""
utils.py
--------
Pure-Python utility functions used by app.py.

ARCHITECTURE NOTE:
  - No Streamlit imports here.  This keeps utilities testable independently.
  - All plotting functions return matplotlib Figure objects so the caller
    (app.py) decides how to render them (st.pyplot / st.image / etc.).
"""

import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Data I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_dataframe(uploaded_file) -> pd.DataFrame:
    """
    Load a CSV or Excel file from a Streamlit UploadedFile object.
    Returns a DataFrame or raises ValueError with a descriptive message.
    """
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif name.endswith((".xls", ".xlsx")):
            return pd.read_excel(uploaded_file)
        else:
            raise ValueError(f"Unsupported file type: {uploaded_file.name}")
    except Exception as exc:
        raise ValueError(f"Could not parse '{uploaded_file.name}': {exc}") from exc


def results_to_dataframe(results: dict, y_min: float, y_range: float) -> pd.DataFrame:
    """
    Convert the results dict from VarProIRLS.fit_irls() to a tidy DataFrame
    suitable for CSV download.
    """
    rows = []
    for model, res in results.items():
        C_scaled = res["C"]
        C_orig   = y_min + y_range * C_scaled
        sigma_mc = y_range * res.get("sigma_mc", np.nan)
        rows.append({
            "model":          model,
            "B":              res["B"],
            "A_scaled":       res["A"],
            "C_scaled":       C_scaled,
            "C_original":     C_orig,
            "sigma_mc":       sigma_mc,
            "sigma_C_lower":  res.get("sigma_C_lower_unscaled", np.nan),
            "sigma_C_upper":  res.get("sigma_C_upper_unscaled", np.nan),
            "SSR":            res["ssr"],
            "sigma_noise":    res["sigma_noise"],
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Figure helpers  (return Figure objects – no st.* calls)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_COLORS = {
    "exponential":      "#2196F3",   # blue
    "sqrt_exponential": "#FF9800",   # orange
    "power_law":        "#4CAF50",   # green
}

MODEL_LABELS = {
    "exponential":      r"Exp $e^{-Bx}$",
    "sqrt_exponential": r"SqrtExp $e^{-B\sqrt{x}}$",
    "power_law":        r"Power $x^{-B}$",
}


def _unscale(y_scaled, y_min, y_range):
    return y_min + y_range * np.asarray(y_scaled, dtype=float)


def fig_live_iteration(info: dict, history: dict, truth_val=None) -> plt.Figure:
    """
    Build a 2×2 figure summarising the current IRLS state.

    Parameters
    ----------
    info    : callback dict from VarProIRLS (single iteration snapshot)
    history : running dict  {model -> {'B': [], 'C': [], 'A': [], 'iter': []}}
    truth_val : optional ground-truth asymptote (original scale)
    """
    model   = info["model"]
    y_min   = info["y_min"]
    y_range = info["y_range"]
    raw_x   = info["raw_x"]
    raw_y   = info["raw_y"]
    x_max   = info["x_max"]
    B, C, A = info["B"], info["C"], info["A"]
    weights = info["weights"]

    color = MODEL_COLORS.get(model, "blue")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Model: {model} — Iteration {info['iteration']}  "
        f"(rel_obj={info['rel_obj']:.2e}, rel_w={info['rel_w']:.2e})",
        fontsize=12,
    )

    # ── Top-left: fitted curve vs data ───────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(raw_x, _unscale(raw_y, y_min, y_range), "ko",
            ms=5, zorder=5, label="Data")

    x_plot = np.linspace(raw_x.min(), raw_x.max() * 1.5, 200)
    t_plot = x_plot / x_max
    phi    = _basis(model, B, t_plot)
    y_plot = _unscale(C + A * phi, y_min, y_range)
    ax.plot(x_plot, y_plot, "-", color=color, lw=2, label=MODEL_LABELS.get(model, model))

    C_orig = float(_unscale(C, y_min, y_range))
    ax.axhline(C_orig, color=color, ls="--", alpha=0.5, label=f"C={C_orig:.5g}")
    if truth_val is not None:
        ax.axhline(truth_val, color="red", ls=":", lw=1.5, label=f"Truth={truth_val:.5g}")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Fitted Curve")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Top-right: weight distribution ───────────────────────────────────────
    ax = axes[0, 1]
    idx_sort = np.argsort(raw_x)
    ax.bar(np.arange(len(raw_x)), weights[idx_sort], color=color, alpha=0.7)
    ax.set_xlabel("Data point (sorted by x)")
    ax.set_ylabel("Weight")
    ax.set_title("IRLS Weights")
    ax.grid(True, axis="y", alpha=0.3)

    # ── Bottom-left: B history ────────────────────────────────────────────────
    ax = axes[1, 0]
    if model in history and history[model]["iter"]:
        iters = history[model]["iter"]
        ax.plot(iters, history[model]["B"], color=color, marker="o", ms=3)
    ax.scatter([info["iteration"]], [B], color="red", zorder=5, s=40)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("B (decay)")
    ax.set_title("B Evolution")
    ax.grid(True, alpha=0.3)

    # ── Bottom-right: C history (original scale) ──────────────────────────────
    ax = axes[1, 1]
    if model in history and history[model]["iter"]:
        iters  = history[model]["iter"]
        C_hist = [_unscale(c, y_min, y_range) for c in history[model]["C"]]
        ax.plot(iters, C_hist, color=color, marker="o", ms=3)
    ax.scatter([info["iteration"]], [C_orig], color="red", zorder=5, s=40)
    if truth_val is not None:
        ax.axhline(truth_val, color="red", ls=":", lw=1.2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("C (asymptote)")
    ax.set_title("C Evolution")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def fig_final_results(results: dict, raw_x, raw_y, y_min, y_range,
                      x_max, truth_val=None, err_df=None, y_col=None) -> plt.Figure:
    """Two-panel final result figure (full view + zoomed tail)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, ax in enumerate(axes):
        ax.plot(raw_x, _unscale(raw_y, y_min, y_range), "ko",
                ms=6, zorder=5, label="Data")

        zoom = ax_idx == 1
        x_start = raw_x.min() + 0.6 * (raw_x.max() - raw_x.min()) if zoom else raw_x.min()
        x_end   = x_max * 1.5
        x_plot  = np.linspace(raw_x.min(), x_end, 300)
        t_plot  = x_plot / x_max

        for model, res in results.items():
            color  = MODEL_COLORS.get(model, "gray")
            label  = MODEL_LABELS.get(model, model)
            B, C, A = res["B"], res["C"], res["A"]
            sigma  = y_range * res.get("sigma_mc", 0.0)
            C_orig = float(_unscale(C, y_min, y_range))

            phi   = _basis(model, B, t_plot)
            y_p   = _unscale(C + A * phi, y_min, y_range)
            ax.plot(x_plot, y_p, "-", color=color, lw=2, alpha=0.85, label=label)
            ax.axhline(C_orig, color=color, ls="--", alpha=0.35)
            ax.fill_between([raw_x.min(), x_end],
                            C_orig - sigma, C_orig + sigma,
                            color=color, alpha=0.08)

        if truth_val is not None:
            ax.axhline(truth_val, color="red", ls=":", lw=2,
                       label=f"Truth {truth_val:.6g}")
            if err_df is not None and y_col and y_col in err_df.columns:
                try:
                    te = float(err_df[y_col].values[-1])
                    ax.fill_between([raw_x.min(), x_end],
                                    truth_val - te, truth_val + te,
                                    color="red", alpha=0.1, label=f"Truth ±{te:.1e}")
                except Exception:
                    pass

        if zoom:
            ax.set_xlim(x_start, x_end)
        ax.set_title("Full View" if not zoom else "Zoomed Tail & Extrapolation")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    return fig


def fig_weights(results: dict, raw_x) -> plt.Figure:
    """Bar charts of final IRLS weights for each model."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]
    idx_sort = np.argsort(raw_x)
    for ax, (model, res) in zip(axes, results.items()):
        w = res.get("final_weights", np.ones(len(raw_x)))
        ax.bar(np.arange(len(raw_x)), w[idx_sort],
               color=MODEL_COLORS.get(model, "blue"), alpha=0.75)
        ax.set_title(f"Weights – {model}")
        ax.set_xlabel("Data point (sorted by x)")
        ax.set_ylabel("Weight")
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper: compute basis function values (mirrors VarProIRLS logic)
# ─────────────────────────────────────────────────────────────────────────────

def _basis(model: str, B: float, t: np.ndarray) -> np.ndarray:
    """Return the nonlinear basis vector φ(t; B) for the given model."""
    if model == "exponential":
        return np.exp(-B * t)
    if model == "sqrt_exponential":
        return np.exp(-B * np.sqrt(t))
    if model == "power_law":
        return np.power(t, -B)
    raise ValueError(f"Unknown model: {model}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure → bytes  (for st.image fast rendering)
# ─────────────────────────────────────────────────────────────────────────────

def fig_to_png_bytes(fig: plt.Figure, dpi: int = 100) -> bytes:
    """Render a matplotlib Figure to PNG bytes without saving to disk."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()