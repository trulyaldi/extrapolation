import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from extrap import extraplus
import re

def upload_df(file_path, start_basis_size = 99):
  df = pd.read_csv(file_path)
  df['Basis Size'] = df['Basis Size'].astype(int)

  for i in range(df['Basis Size'].values[0],start_basis_size + 1,100):
    df.drop(df[df['Basis Size'] == i].index, inplace = True)

  return df

def upload_basis(file_path):
  df = pd.read_csv(file_path)
  df['basis size'] = df['basis size'].astype(int)

  return df[:-1]

def upload_error(file_path):
  df = pd.read_csv(file_path)
  df['basis size'] = df['basis size'].astype(int)

  return df.tail(1)

def graph(df: pd.DataFrame, n_cols: int = 4):
    df_plot = df.copy()
    df_plot.columns = [col.lower() for col in df_plot.columns]

    if 'basis size' not in df_plot.columns:
        raise ValueError("Input DataFrame must contain a 'basis size' column.")

    df_plot['basis size'] = df_plot['basis size'].astype(int)
    features = sorted([col for col in df_plot.columns if col != 'basis size'])
    n_features = len(features)

    if n_features == 0:
        return

    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7 * n_cols, 5.5 * n_rows)
    )
    
    fig.suptitle(r'Be($^3D^e$)', fontsize=22, y=1.05, fontweight='bold')

    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Define the data points and the limit
    basis_sizes = df_plot['basis size'].values
    # This is the x-value of the point just before the last one
    x_limit = basis_sizes[-2] 

    for i, feature in enumerate(features):
        ax = axes[i]
        y = df_plot[feature].values

        # Plot points up until the second to last
        ax.scatter(basis_sizes[:-1], y[:-1], marker='o', s=50, edgecolors='royalblue', alpha=0.8)

        # Horizontal dashed line at the very last value
        ax.axhline(y[-1], color='red', linestyle='--', linewidth=2, alpha=0.9)

        ax.set_title(feature, fontsize=15, pad=10)
        ax.set_xlabel('Basis Size', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        
        # --- KEY FIX: SET X-AXIS LIMIT ---
        # This cuts the plot off right at the last scatter point
        ax.set_xlim(left=basis_sizes[0], right=x_limit)
        
        # Clean up ticks so they don't overlap
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
        ax.ticklabel_format(style='plain', axis='x')
        ax.grid(True, linestyle='--', alpha=0.3)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=3.0)
    plt.show()

def fit_all_irls(main_df, main_inf_df=None, main_df_err=None):
    x_col = "basis size" if "basis size" in main_df.columns else main_df.columns[0]
    y_cols = [c for c in main_df.columns if c != x_col]

    def truth_val_for(col):
        if main_inf_df is None:
            return None
        if col not in main_inf_df.columns:
            return None
        vals = np.asarray(main_inf_df[col], dtype=float)
        vals = vals[np.isfinite(vals)]
        return float(vals[-1]) if vals.size else None

    def truth_err_for(col):
        if main_df_err is None:
            return None
        if col not in main_df_err.columns:
            return None
        vals = np.asarray(main_df_err[col], dtype=float)
        vals = vals[np.isfinite(vals)]
        return float(vals[-1]) if vals.size else None

    for y_col in y_cols:
        solver = extraplus(main_df, x_col, y_col,
                            err_df=main_df_err,
                            inf_df=main_inf_df)
        solver.fit_irls(verbose=False)

        tv = truth_val_for(y_col)
        te = truth_err_for(y_col)

        if te is not None:
            solver.err_df = solver.err_df.copy()
            solver.err_df[y_col] = te

        solver.plot(truth_val=tv)

def fit_all_log(main_df, main_inf_df=None, main_df_err=None):
    x_col = "basis size" if "basis size" in main_df.columns else main_df.columns[0]
    y_cols = [c for c in main_df.columns if c != x_col]

    def truth_val_for(col):
        if main_inf_df is None:
            return None
        if col not in main_inf_df.columns:
            return None
        vals = np.asarray(main_inf_df[col], dtype=float)
        vals = vals[np.isfinite(vals)]
        return float(vals[-1]) if vals.size else None

    def truth_err_for(col):
        if main_df_err is None:
            return None
        if col not in main_df_err.columns:
            return None
        vals = np.asarray(main_df_err[col], dtype=float)
        vals = vals[np.isfinite(vals)]
        return float(vals[-1]) if vals.size else None

    for y_col in y_cols:
        solver = extraplus(main_df, x_col, y_col,
                            err_df=main_df_err,
                            inf_df=main_inf_df,
                            use_energy_b=True)
        solver.fit_linearized(verbose=False)

        tv = truth_val_for(y_col)
        te = truth_err_for(y_col)

        if te is not None:
            solver.err_df = solver.err_df.copy()
            solver.err_df[y_col] = te

        solver.plot(truth_val=tv)

import matplotlib
matplotlib.use('Agg')

from matplotlib.backends.backend_pdf import PdfPages


class CBSFitPlotter:
    """
    Fits and plots CBS-extrapolation expectation values for one or many systems
    with a single, uniform API.

    Three ways to provide data (all optional — you can also use .add_system later):

        # 1) single system
        CBSFitPlotter(df, 'mu_16O_4_1De').run(save_pdf='one.pdf')

        # 2) several systems as {name: df}
        CBSFitPlotter({
            'mu_16O_4_1De': df1,
            'mu_16O_4_1Po': df2,
        }).run(save_pdf='many.pdf')

        # 3) several systems as a list of spec dicts (allows per-system overrides
        #    of err_df / inf_df / skip_cols / n_fit / x_col)
        CBSFitPlotter([
            {'df': df1, 'system_name': 'mu_16O_4_1De'},
            {'df': df2, 'system_name': 'mu_16O_4_1Po', 'inf_df': inf2},
        ]).run(save_pdf='many.pdf')

        # 4) build incrementally
        p = CBSFitPlotter(x_col='basis size')
        p.add_system(df1, 'mu_16O_4_1De')
        p.add_system(df2, 'mu_16O_4_1Po')
        p.run(save_pdf='many.pdf')

    NOTE: this class is save-only. It never calls plt.show() and never
    renders inline in a notebook — every system is fitted, drawn onto its
    own page, written to the PDF, and the figure is closed immediately
    afterward. This keeps memory and render time flat no matter how many
    systems you run (the old "build one giant Figure with everyone's rows
    in it" approach is what caused notebook lag/crashes).

    The keyword args on __init__ (x_col, err_df, inf_df, skip_cols, n_fit) act as
    defaults applied to every system that doesn't override them itself.
    """

    # ---- styling shared across all plots ----
    COLORS = {
        'exponential':      'blue',
        'sqrt_exponential': 'orange',
        'power_law':        'green',
    }
    LABELS = {
        'exponential':      r'Exp($e^{-Bx}$)',
        'sqrt_exponential': r'SqrtExp($e^{-B\sqrt{x}}$)',
        'power_law':        r'Power($x^{-B}$)',
    }
    X_LABELS_LOG = {
        'exponential':      r'$x \;/\; x_{\max}$',
        'sqrt_exponential': r'$\sqrt{x \;/\; x_{\max}}$',
        'power_law':        r'$\ln(x \;/\; x_{\max})$',
    }
    TITLES_LOG = {
        'exponential':      r'Exponential: $\ln(y-C)$ vs $x/x_{max}$',
        'sqrt_exponential': r'Sqrt-Exp: $\ln(y-C)$ vs $\sqrt{x/x_{max}}$',
        'power_law':        r'Power-Law: $\ln(y-C)$ vs $\ln(x/x_{max})$',
    }
    MODELS = ['exponential', 'sqrt_exponential', 'power_law']

    def __init__(self, data=None, system_name=None, *,
                 x_col='basis size', err_df=None, inf_df=None,
                 skip_cols=None, n_fit=None):
        self.default_x_col = x_col
        self.default_err_df = err_df
        self.default_inf_df = inf_df
        self.default_skip_cols = skip_cols
        self.default_n_fit = n_fit

        self.systems = []        # list of normalized spec dicts

        if data is not None:
            self._ingest(data, system_name)

    # =====================================================================
    # Input handling
    # =====================================================================
    def _ingest(self, data, system_name):
        if isinstance(data, pd.DataFrame):
            if system_name is None:
                raise ValueError("A single DataFrame also needs a system_name, "
                                 "e.g. CBSFitPlotter(df, 'mu_16O_4_1De').")
            self.add_system(data, system_name)
        elif isinstance(data, dict):
            for name, df in data.items():
                self.add_system(df, name)
        elif isinstance(data, (list, tuple)):
            for spec in data:
                if not isinstance(spec, dict) or 'df' not in spec or 'system_name' not in spec:
                    raise ValueError("Each item in a list must be a dict with at "
                                     "least 'df' and 'system_name' keys.")
                self.add_system(
                    spec['df'], spec['system_name'],
                    x_col=spec.get('x_col'),
                    err_df=spec.get('err_df'),
                    inf_df=spec.get('inf_df'),
                    skip_cols=spec.get('skip_cols'),
                    n_fit=spec.get('n_fit'),
                )
        else:
            raise TypeError("`data` must be a DataFrame, a {name: df} dict, or a "
                            "list of spec dicts.")

    def add_system(self, df, system_name, *, x_col=None, err_df=None, inf_df=None,
                   skip_cols=None, n_fit=None):
        """
        Register one system. Any arg left as None falls back to the default given
        to __init__. Returns self so calls can be chained.
        """
        self.systems.append({
            'df':          df,
            'system_name': system_name,
            'x_col':       x_col if x_col is not None else self.default_x_col,
            'err_df':      err_df if err_df is not None else self.default_err_df,
            'inf_df':      inf_df if inf_df is not None else self.default_inf_df,
            'skip_cols':   (skip_cols if skip_cols is not None else self.default_skip_cols) or [],
            'n_fit':       n_fit if n_fit is not None else self.default_n_fit,
        })
        return self

    # =====================================================================
    # Name formatting
    # =====================================================================
    @staticmethod
    def _format_system_name(name):
        """
        Parses names like 'mu_16O_4_1De' or 'mu_infO_7_2Se' into LaTeX:
            mu_16O_4_1De   -> $\\mu\\,{}^{16}\\mathrm{O}^{4+}\\,{}^{1}D^{e}$
            mu_infO_7_2Se  -> $\\mu\\,{}^{\\infty}\\mathrm{O}^{7+}\\,{}^{2}S^{e}$
        Falls back to the raw name for anything that doesn't match this shape.
        """
        parts = name.split('_')
        if len(parts) != 4 or parts[0] != 'mu':
            return name

        _, nucleus_part, n_electrons, term = parts

        m = re.match(r'^(\d+|inf)([A-Za-z]+)$', nucleus_part)
        if not m:
            return name
        mass_str, element_sym = m.groups()
        mass_latex = r'\infty' if mass_str == 'inf' else mass_str
        element_sym = element_sym.capitalize()

        if len(term) < 2:
            return name
        mult = term[0]
        L = term[1].upper()
        parity = term[2] if len(term) > 2 else ''

        ion = rf"{{}}^{{{mass_latex}}}\mathrm{{{element_sym}}}^{{{n_electrons}+}}"
        term_latex = rf"{{}}^{{{mult}}}{L}" + (rf"^{{{parity}}}" if parity else "")

        return rf"$\mu\,{ion}\,{term_latex}$"

    # =====================================================================
    # Fitting (one system at a time, called lazily from run())
    # =====================================================================
    def _fit_system(self, spec, verbose=True):
        """Fit one system and return its block dict, or None if nothing to fit."""
        df = spec['df']
        x_col = spec['x_col']
        skip_cols = spec['skip_cols']

        y_cols = [c for c in df.columns if c != x_col and c not in skip_cols]
        if not y_cols:
            print(f"No columns found to fit for '{spec['system_name']}', skipping.")
            return None

        formatted_name = self._format_system_name(spec['system_name'])
        if verbose:
            print(f"Fitting {len(y_cols)} columns for {formatted_name}...")

        fitters = []
        for y_col in y_cols:
            fitter = extraplus(
                df=df, x_col=x_col, y_col=y_col,
                err_df=spec['err_df'], inf_df=spec['inf_df'], n_fit=spec['n_fit'],
                use_energy_b=True
            )
            fitter.fit_linearized(compute_uq=True, verbose=False)
            fitters.append(fitter)

        return {
            'system_name':    spec['system_name'],
            'formatted_name': formatted_name,
            'fitters':        fitters,
            'err_df':         spec['err_df'],
            'inf_df':         spec['inf_df'],
        }

    # =====================================================================
    # Public entry points — SAVE ONLY, never shown inline
    # =====================================================================
    def run(self, save_pdf='cbs_fits.pdf', verbose=True):
        """
        Fit and plot every registered system, one page per system, written
        directly to `save_pdf`. Nothing is displayed in the notebook; each
        page's figure is closed right after it's saved so memory and render
        time stay flat regardless of how many systems are registered.

        Returns save_pdf (the path written).
        """
        if not self.systems:
            raise RuntimeError("No systems registered. Use add_system(...) or pass "
                               "data to the constructor first.")

        n_written = 0
        with PdfPages(save_pdf) as pdf:
            for spec in self.systems:
                block = self._fit_system(spec, verbose=verbose)
                if block is None:
                    continue

                fig = self._build_system_page(block)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)          # frees this system's axes immediately
                n_written += 1
                del block, fig          # drop fitted arrays before the next system

        print(f"Saved {n_written} system page(s) to '{save_pdf}'")
        return save_pdf

    # kept as an alias so existing call sites (`.plot(...)`) keep working
    def plot(self, save_pdf='cbs_fits.pdf', verbose=True):
        return self.run(save_pdf=save_pdf, verbose=verbose)

    # ------------------------------------------------------------------
    # one system = one self-contained page/figure
    # ------------------------------------------------------------------
    def _build_system_page(self, block):
        fitters = block['fitters']
        N = len(fitters)
        fig, axes = plt.subplots(nrows=N, ncols=5, figsize=(28, 5 * N), squeeze=False)
        fig.suptitle(block['formatted_name'], fontsize=24, fontweight='bold')

        for idx, fitter in enumerate(fitters):
            axes_row = {
                'full': axes[idx, 0],
                'zoom': axes[idx, 1],
                'exponential':      axes[idx, 2],
                'sqrt_exponential': axes[idx, 3],
                'power_law':        axes[idx, 4],
            }
            self._draw_data_row(axes_row, block, fitter)

        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.subplots_adjust(top=0.95, hspace=0.3)
        return fig

    # ------------------------------------------------------------------
    # one expectation-value row: unchanged plotting logic, reads axes
    # from the passed-in dict
    # ------------------------------------------------------------------
    def _draw_data_row(self, axes_row, block, fitter):
        colors = self.COLORS
        labels = self.LABELS
        err_df = block['err_df']
        inf_df = block['inf_df']
        y_col = fitter.y_col

        ax_full = axes_row['full']
        ax_zoom = axes_row['zoom']
        ax_log_axes = {
            'exponential':      axes_row['exponential'],
            'sqrt_exponential': axes_row['sqrt_exponential'],
            'power_law':        axes_row['power_law'],
        }

        if not fitter.results:
            ax_full.text(0.5, 0.5, f"Fit failed for {y_col}", ha='center', va='center')
            return

        truth_val = fitter.truth_val

        def unscale(y_sc):
            return fitter.y_min + fitter.y_range * np.asarray(y_sc, dtype=float)

        y_data = unscale(fitter.raw_y)

        # ---------------- Full View ----------------
        ax_full.plot(fitter.raw_x, y_data, 'ko', label='Data', zorder=5, markersize=6)
        for model, res in fitter.results.items():
            fitter.model_type = model
            C_sc   = res['C']
            sig_sc = res.get('sigma_mc', 0.0)
            C      = float(unscale(C_sc))
            sigma  = float(fitter.y_range * sig_sc)

            x_plot = np.linspace(fitter.x_min, fitter.x_max * 1.5, 200)
            t_plot = x_plot / fitter.x_max
            phi_p  = fitter._compute_basis(res['B'], t_plot)
            y_plot = unscale(C_sc + res['A'] * phi_p[:, 1])

            ax_full.plot(x_plot, y_plot, '-', color=colors[model], linewidth=2, alpha=0.8, label=labels[model])
            ax_full.axhline(C, color=colors[model], linestyle='--', alpha=0.3)
            ax_full.fill_between([fitter.x_min, fitter.x_max * 1.5],
                                 C - sigma, C + sigma,
                                 color=colors[model], alpha=0.1)

        if inf_df is not None and y_col in inf_df.columns:
            inf_val = float(inf_df[y_col].iloc[0])
            err_val = float(err_df[y_col].iloc[0]) if (err_df is not None and y_col in err_df.columns) else 0.0
            x_band  = [fitter.x_min, fitter.x_max * 1.5]
            ax_full.axhline(inf_val, color='red', linestyle='--', linewidth=1.5,
                            alpha=0.7, label='CBS limit', zorder=6)
            if err_val:
                ax_full.fill_between(x_band,
                                     inf_val - err_val, inf_val + err_val,
                                     color='red', alpha=0.15, label='CBS ± err', zorder=5)

        if truth_val is not None:
            ax_full.axhline(truth_val, color='r', linestyle=':', linewidth=2, label='Exact')

        ax_full.set_title(f"{y_col} - Full View")
        ax_full.set_xlabel("Basis Size")
        ax_full.set_ylabel(y_col)
        ax_full.grid(True, alpha=0.3)
        ax_full.legend(fontsize=9)

        # ---------------- Zoomed Tail ----------------
        ax_zoom.plot(fitter.raw_x, y_data, 'ko', label='Data', zorder=5, markersize=6)

        zoom_start = fitter.x_min + 0.6 * fitter.range_x
        zoom_end   = fitter.x_max * 1.5
        y_min_z, y_max_z = np.inf, -np.inf

        mask = fitter.raw_x >= zoom_start
        if np.any(mask):
            y_min_z = min(y_min_z, np.min(y_data[mask]))
            y_max_z = max(y_max_z, np.max(y_data[mask]))

        for model, res in fitter.results.items():
            fitter.model_type = model
            C_sc   = res['C']
            sig_sc = res.get('sigma_mc', 0.0)
            C      = float(unscale(C_sc))
            sigma  = float(fitter.y_range * sig_sc)

            x_plot = np.linspace(fitter.x_min, fitter.x_max * 1.5, 200)
            t_plot = x_plot / fitter.x_max
            phi_p  = fitter._compute_basis(res['B'], t_plot)
            y_plot = unscale(C_sc + res['A'] * phi_p[:, 1])

            ax_zoom.plot(x_plot, y_plot, '-', color=colors[model], linewidth=2, alpha=0.8, label=labels[model])
            ax_zoom.axhline(C, color=colors[model], linestyle='--', alpha=0.3)
            ax_zoom.fill_between([fitter.x_min, fitter.x_max * 1.5],
                                 C - sigma, C + sigma,
                                 color=colors[model], alpha=0.1)

            y_min_z = min(y_min_z, C - sigma)
            y_max_z = max(y_max_z, C + sigma)

            mask_p = (x_plot >= zoom_start) & (x_plot <= zoom_end)
            if np.any(mask_p):
                y_min_z = min(y_min_z, np.min(y_plot[mask_p]))
                y_max_z = max(y_max_z, np.max(y_plot[mask_p]))

        if inf_df is not None and y_col in inf_df.columns:
            inf_val = float(inf_df[y_col].iloc[0])
            err_val = float(err_df[y_col].iloc[0]) if (err_df is not None and y_col in err_df.columns) else 0.0
            x_band  = [fitter.x_min, fitter.x_max * 1.5]
            ax_zoom.axhline(inf_val, color='red', linestyle='-', linewidth=1.5,
                            alpha=0.7, label='CBS limit', zorder=6)
            if err_val:
                ax_zoom.fill_between(x_band,
                                     inf_val - err_val, inf_val + err_val,
                                     color='red', alpha=0.15, label='CBS ± err', zorder=5)
            y_min_z = min(y_min_z, inf_val - err_val)
            y_max_z = max(y_max_z, inf_val + err_val)

        if truth_val is not None:
            ax_zoom.axhline(truth_val, color='r', linestyle='--', linewidth=2, label='Exact')
            y_min_z = min(y_min_z, truth_val)
            y_max_z = max(y_max_z, truth_val)

        ax_zoom.set_xlim(zoom_start, zoom_end)
        if not (np.isinf(y_min_z) or np.isinf(y_max_z)):
            span = y_max_z - y_min_z
            if span == 0:
                span = abs(y_min_z) * 0.01 + 1e-10
            ax_zoom.set_ylim(y_min_z - 0.1 * span, y_max_z + 0.1 * span)

        ax_zoom.set_title(f"{y_col} - Zoomed Tail")
        ax_zoom.set_xlabel("Basis Size")
        ax_zoom.grid(True, alpha=0.3)

        # ---------------- Linearized panels ----------------
        for model in self.MODELS:
            ax_log = ax_log_axes[model]
            if model not in fitter.results:
                continue

            res = fitter.results[model]
            fitter.model_type = model
            color = colors[model]

            B, C, A = res['B'], res['C'], res['A']
            tx = fitter._make_tx(model)

            if fitter.is_increasing:
                diff = C - fitter.raw_y
            else:
                diff = fitter.raw_y - C

            valid = diff > 0
            tx_v      = tx[valid]
            ln_diff_v = np.log(diff[valid])

            n_pts = len(tx_v)
            ax_log.scatter(tx_v, ln_diff_v, color=color, s=40, zorder=4, label=f'Data ({n_pts} pts)')

            n_invalid = int(np.sum(~valid))
            if n_invalid:
                ax_log.scatter(tx[~valid],
                               np.full(n_invalid, ln_diff_v.min() if len(ln_diff_v) else 0),
                               color='red', marker='x', s=50, zorder=5,
                               label=f'Invalid (y≤C): {n_invalid}')

            ln_A  = np.log(abs(A)) if abs(A) > 1e-15 else 0.0
            slope = -B
            tx_line = np.linspace(tx_v.min() if n_pts else 0, tx_v.max() if n_pts else 1, 200)
            ln_line = ln_A + slope * tx_line

            r2 = res.get('r2_linearized', float('nan'))
            ax_log.plot(tx_line, ln_line, '-', color=color, linewidth=2,
                        label=f'Fit (R²={r2:.5f})\nB={B:.4f}')

            if len(ln_diff_v) > 3:
                from scipy import stats

                df_log     = n_pts - 2
                t_crit     = stats.t.ppf(0.975, df_log)
                resid_log  = ln_diff_v - (ln_A + slope * tx_v)
                sigma2_res = np.sum(resid_log ** 2) / df_log
                mean_tx    = np.mean(tx_v)
                ss_tx      = np.sum((tx_v - mean_tx) ** 2)
                if ss_tx == 0:
                    ss_tx = 1e-15

                var_line = sigma2_res * (1.0 / n_pts + (tx_line - mean_tx) ** 2 / ss_tx)
                var_pred = sigma2_res * (1.0 + 1.0 / n_pts + (tx_line - mean_tx) ** 2 / ss_tx)
                se_line  = np.sqrt(var_line)
                se_pred  = np.sqrt(var_pred)

                ax_log.fill_between(tx_line,
                                    ln_line - t_crit * se_pred,
                                    ln_line + t_crit * se_pred,
                                    color=color, alpha=0.08, label='95% PI Band')
                ax_log.fill_between(tx_line,
                                    ln_line - t_crit * se_line,
                                    ln_line + t_crit * se_line,
                                    color=color, alpha=0.2, label='95% CI Band')

            ax_log.set_title(self.TITLES_LOG[model], fontsize=11)
            ax_log.set_xlabel(self.X_LABELS_LOG[model], fontsize=10)
            ax_log.set_ylabel(r'$\ln(y - C)$', fontsize=10)
            ax_log.legend(fontsize=8)
            ax_log.grid(True, alpha=0.3)





def generate_and_fit_synthetic(
    system_name='synth_random',
    n_values=18,
    n_basis=60,
    basis_start=100,
    basis_step=100,
    cbs_basis=16000,
    convergence_model='mixed',   # 'mixed' | 'exponential' | 'sqrt_exponential' | 'power_law'
    probs=None,                  # shape probabilities; default = uniform over the 4 kinds
    annotate_kind=False,         # if True, appends kind to column name: obs_03_bump
    seed=42,
    save_pdf=None,               # path string -> save plot as PDF; None -> skip
    plot=True,                   # False -> build data only, skip fitter/plot
):
    """Generate a randomised synthetic convergence dataset and fit/plot it with
    ``fit_and_plot_system``.

    Purpose: stress-test ``VarProLinearized`` on many varied but realistic
    convergence curves where the ground truth (asymptote C, amplitude, decay
    model, shape) is known, so recovered asymptotes and uncertainties can be
    validated against it.

    Each of ``n_values`` columns converges (in the long run) toward a CBS limit
    ``C`` via one of three decay families (exponential / sqrt-exponential /
    power-law) and takes one of four shapes:

      * 'inc'  -- pure monotone increasing  (approaches C from below)
      * 'dec'  -- pure monotone decreasing  (approaches C from above)
      * 'bump' -- long-run DECREASING, but rises in the early region then falls
      * 'dip'  -- long-run INCREASING, but dips in the early region then rises

    All per-column parameters (CBS limit, amplitude, decay rate, noise level,
    bump geometry, convergence speed) are drawn from physically motivated
    distributions internally so no tuning knobs are exposed to the caller.

    Returns ``(df_init, df_inf, df_err, df_truth)`` where ``df_truth`` holds
    per-column ground truth (C, amp, kind, model, B, r6, noise levels) for
    downstream recovery checks.
    """
    rng = np.random.default_rng(seed)

    # ── shape probabilities ────────────────────────────────────────────────────
    if probs is None:
        probs = {'inc': 0.25, 'dec': 0.25, 'bump': 0.25, 'dip': 0.25}
    kinds_pool = list(probs.keys())
    p = np.asarray([probs[k] for k in kinds_pool], dtype=float)
    p /= p.sum()

    # ── basis grid ─────────────────────────────────────────────────────────────
    basis_sizes = np.arange(
        basis_start, basis_start + n_basis * basis_step, basis_step, dtype=float
    )
    x_max = float(basis_sizes[-1])
    t     = basis_sizes / x_max
    t0    = float(t[0])
    t_win = 1.0
    t_inf = float(cbs_basis) / x_max

    models = ['exponential', 'sqrt_exponential', 'power_law']

    # ── helpers ────────────────────────────────────────────────────────────────
    def solve_B(model, r6):
        """Decay rate so the remaining fraction at window-end == r6."""
        r6 = float(np.clip(r6, 1e-6, 0.5))
        if model == 'exponential':
            return -np.log(r6) / (t_win - t0)
        if model == 'sqrt_exponential':
            return -np.log(r6) / (np.sqrt(t_win) - np.sqrt(t0))
        return np.log(r6) / np.log(t0 / t_win)   # power-law

    def decay(model, B, tt):
        """Basis function: g(tt) -> 0 as tt -> inf."""
        if model == 'exponential':
            return np.exp(-B * tt)
        if model == 'sqrt_exponential':
            return np.exp(-B * np.sqrt(tt))
        return tt ** (-B)

    def hnorm(model, B, tt):
        """Normalised so h(t0) == 1, making |amp| the first-basis offset."""
        return decay(model, B, tt) / decay(model, B, t0)

    # ── per-column generation ──────────────────────────────────────────────────
    kinds = rng.choice(kinds_pool, size=n_values, p=p)

    data      = {'basis size': basis_sizes.astype(int)}
    inf_data  = {'basis size': [int(cbs_basis)]}
    err_data  = {}
    truth_rows = []

    for i in range(n_values):
        kind = str(kinds[i])

        # --- physical parameters (all drawn from the rng) ---------------------

        # CBS limit: span covers typical atomic-observable magnitudes;
        # avoid |C| < 0.5 so relative amplitude is well-defined.
        C = float(rng.uniform(-20.0, 20.0))
        while abs(C) < 0.5:
            C = float(rng.uniform(-20.0, 20.0))

        # Amplitude: 0.3 %..30 % of |C| (log-uniform), matching real data spread.
        amp = max(abs(C), 1.0) * 10.0 ** rng.uniform(-2.5, -0.5)

        # Decay family.
        model = rng.choice(models) if convergence_model == 'mixed' else convergence_model

        # Convergence speed: fraction of amplitude still remaining at window-end.
        # Log-uniform over [5e-5, 5e-3] keeps signal above the tail noise floor,
        # matching the 99.7–99.999 % converged regime seen in real Be/Li data.
        r6 = 10.0 ** rng.uniform(np.log10(5e-5), np.log10(5e-3))

        # Bump/dip geometry: excursion decays 1.5x–3x faster than the main term
        # and covers 70 %–95 % of the main amplitude so A still wins long-run.
        bump_rate_mult = rng.uniform(1.5, 3.0)
        bump_frac      = rng.uniform(0.70, 0.95)

        # Noise: head std = 5 %–20 % of amp, decaying with a random rate so the
        # tail is always clean (noise_decay drawn from 3–6, so exp(-noise_decay)
        # gives tail/head ratio ~ 0.002–5e-3 at t=1).
        noise_frac0 = rng.uniform(0.05, 0.07)
        noise_decay = rng.uniform(3.0, 4.0)
        

        # --- build the clean curve --------------------------------------------
        B      = solve_B(model, r6)
        B_fast = B * bump_rate_mult

        h_main = hnorm(model, B,      t)
        h_exc  = hnorm(model, B_fast, t)

        if kind == 'inc':
            clean    = C - amp * h_main
            true_dir = 'increasing'
        elif kind == 'dec':
            clean    = C + amp * h_main
            true_dir = 'decreasing'
        elif kind == 'bump':
            clean    = C + amp * h_main - amp * bump_frac * h_exc
            true_dir = 'decreasing'
        else:  # dip
            clean    = C - amp * h_main + amp * bump_frac * h_exc
            true_dir = 'increasing'

        # --- add decaying measurement noise -----------------------------------
        sigma_t = noise_frac0 * amp * np.exp(-noise_decay * t)
        y = clean + rng.normal(0.0, sigma_t, size=len(basis_sizes))

        # --- CBS reference point at basis == cbs_basis ------------------------
        h_inf      = hnorm(model, B,      t_inf)
        h_inf_fast = hnorm(model, B_fast, t_inf)

        if kind == 'inc':
            clean_inf = C - amp * h_inf
        elif kind == 'dec':
            clean_inf = C + amp * h_inf
        elif kind == 'bump':
            clean_inf = C + amp * h_inf - amp * bump_frac * h_inf_fast
        else:
            clean_inf = C - amp * h_inf + amp * bump_frac * h_inf_fast

        sigma_tail = noise_frac0 * amp * np.exp(-noise_decay * t_win)
        cbs_val    = clean_inf + rng.normal(0.0, sigma_tail)
        err_val    = max(sigma_tail, amp * 1e-9)

        # --- store ------------------------------------------------------------
        name = f'obs_{i + 1:02d}' + (f'_{kind}' if annotate_kind else '')

        data[name]     = y
        inf_data[name] = [cbs_val]
        err_data[name] = [err_val]
        truth_rows.append({
            'column': name, 'kind': kind, 'true_dir': true_dir, 'model': model,
            'C': C, 'amp': amp, 'A_first_offset': float(clean[0] - C),
            'B': float(B), 'r6': float(r6),
            'bump_frac': float(bump_frac), 'bump_rate_mult': float(bump_rate_mult),
            'noise_frac0': float(noise_frac0), 'noise_decay': float(noise_decay),
            'sigma_head': float(sigma_t[0]), 'sigma_tail': float(sigma_tail),
        })

    df_init  = pd.DataFrame(data)
    df_inf   = pd.DataFrame(inf_data)
    df_err   = pd.DataFrame(err_data)
    df_truth = pd.DataFrame(truth_rows)

    print(f"[{system_name}] generated {n_values} columns  "
          f"basis {int(basis_sizes[0])}..{int(basis_sizes[-1])}  "
          f"(+CBS @ {int(cbs_basis)})")
    print(f"  shapes : {df_truth['kind'].value_counts().to_dict()}")
    print(f"  models : {df_truth['model'].value_counts().to_dict()}")

    if plot:
        fit_and_plot_system(
            df=df_init,
            system_name=system_name,
            x_col='basis size',
            err_df=df_err,
            inf_df=df_inf,
            save_pdf=save_pdf,
        )

    return df_init, df_inf, df_err, df_truth


def fit_and_plot_system(df, system_name, x_col='basis size', err_df=None, inf_df=None, skip_cols=None, n_fit=None, save_pdf=None):
    if skip_cols is None:
        skip_cols = []
        
    y_cols = [col for col in df.columns if col != x_col and col not in skip_cols]
    N = len(y_cols)
    
    if N == 0:
        print("No columns found to fit!")
        return

    def format_system_name(name):
        parts = name.split('_')
        if len(parts) != 2:
            return name
        element = parts[0].capitalize()
        term = parts[1]
        if len(term) >= 2:
            multiplicity = term[0]
            L = term[1].upper()
            parity = f"^{term[2]}" if len(term) > 2 else ""
            return f"{element}($^{multiplicity}{L}{parity}$)"
        return f"{element}({term})"

    formatted_name = format_system_name(system_name)

    fitters = []
    print(f"Fitting {N} columns for {formatted_name}...")
    for y_col in y_cols:
        fitter = extraplus(
            df=df, x_col=x_col, y_col=y_col,
            err_df=err_df, inf_df=inf_df, n_fit=n_fit,
            use_energy_b=True
        )
        fitter.fit_linearized(compute_uq=True, verbose=False)
        fitters.append(fitter)

    colors = {
        'exponential':      'blue',
        'sqrt_exponential': 'orange',
        'power_law':        'green',
    }
    labels = {
        'exponential':      r'Exp($e^{-Bx}$)',
        'sqrt_exponential': r'SqrtExp($e^{-B\sqrt{x}}$)',
        'power_law':        r'Power($x^{-B}$)',
    }
    x_labels_log = {
        'exponential':      r'$x \;/\; x_{\max}$',
        'sqrt_exponential': r'$\sqrt{x \;/\; x_{\max}}$',
        'power_law':        r'$\ln(x \;/\; x_{\max})$',
    }
    titles_log = {
        'exponential':      r'Exponential: $\ln(y-C)$ vs $x/x_{max}$',
        'sqrt_exponential': r'Sqrt-Exp: $\ln(y-C)$ vs $\sqrt{x/x_{max}}$',
        'power_law':        r'Power-Law: $\ln(y-C)$ vs $\ln(x/x_{max})$',
    }

    fig, axes = plt.subplots(nrows=N, ncols=5, figsize=(28, 5 * N), squeeze=False)
    fig.suptitle(formatted_name, fontsize=24, fontweight='bold')

    for idx, fitter in enumerate(fitters):  # ← everything below is inside this loop
        y_col = fitter.y_col
        ax_full = axes[idx, 0]
        ax_zoom = axes[idx, 1]
        ax_log_axes = {
            'exponential':      axes[idx, 2],
            'sqrt_exponential': axes[idx, 3],
            'power_law':        axes[idx, 4],
        }

        if not fitter.results:
            ax_full.text(0.5, 0.5, f"Fit failed for {y_col}", ha='center', va='center')
            continue

        truth_val = fitter.truth_val

        def unscale(y_sc):
            return fitter.y_min + fitter.y_range * np.asarray(y_sc, dtype=float)

        y_data = unscale(fitter.raw_y)

        # =======================================================
        # Left Panel (Col 0): Full View
        # =======================================================
        ax_full.plot(fitter.raw_x, y_data, 'ko', label='Data', zorder=5, markersize=6)
        for model, res in fitter.results.items():
            fitter.model_type = model
            C_sc   = res['C']
            sig_sc = res.get('sigma_mc', 0.0)
            C      = float(unscale(C_sc))
            sigma  = float(fitter.y_range * sig_sc)

            x_plot = np.linspace(fitter.x_min, fitter.x_max * 1.5, 200)
            t_plot = x_plot / fitter.x_max
            phi_p  = fitter._compute_basis(res['B'], t_plot)
            y_plot = unscale(C_sc + res['A'] * phi_p[:, 1])

            ax_full.plot(x_plot, y_plot, '-', color=colors[model], linewidth=2, alpha=0.8, label=labels[model])
            ax_full.axhline(C, color=colors[model], linestyle='--', alpha=0.3)
            ax_full.fill_between([fitter.x_min, fitter.x_max * 1.5],
                                 C - sigma, C + sigma,
                                 color=colors[model], alpha=0.1)

        if inf_df is not None and y_col in inf_df.columns:
            inf_val = float(inf_df[y_col].iloc[0])
            err_val = float(err_df[y_col].iloc[0]) if (err_df is not None and y_col in err_df.columns) else 0.0
            x_band  = [fitter.x_min, fitter.x_max * 1.5]
            ax_full.axhline(inf_val, color='red', linestyle='--', linewidth=1.5,
                            alpha=0.7, label='CBS limit', zorder=6)
            if err_val:
                ax_full.fill_between(x_band,
                                     inf_val - err_val, inf_val + err_val,
                                     color='red', alpha=0.15, label='CBS ± err', zorder=5)

        if truth_val is not None:
            ax_full.axhline(truth_val, color='r', linestyle=':', linewidth=2, label='Exact')

        ax_full.set_title(f"{y_col} - Full View")
        ax_full.set_xlabel("Basis Size")
        ax_full.set_ylabel(y_col)
        ax_full.grid(True, alpha=0.3)
        ax_full.legend(fontsize=9)

        # =======================================================
        # Middle Panel (Col 1): Zoomed Tail
        # =======================================================
        ax_zoom.plot(fitter.raw_x, y_data, 'ko', label='Data', zorder=5, markersize=6)

        zoom_start = fitter.x_min + 0.6 * fitter.range_x
        zoom_end   = fitter.x_max * 1.5
        y_min_z, y_max_z = np.inf, -np.inf

        mask = fitter.raw_x >= zoom_start
        if np.any(mask):
            y_min_z = min(y_min_z, np.min(y_data[mask]))
            y_max_z = max(y_max_z, np.max(y_data[mask]))

        for model, res in fitter.results.items():
            fitter.model_type = model
            C_sc   = res['C']
            sig_sc = res.get('sigma_mc', 0.0)
            C      = float(unscale(C_sc))
            sigma  = float(fitter.y_range * sig_sc)

            x_plot = np.linspace(fitter.x_min, fitter.x_max * 1.5, 200)
            t_plot = x_plot / fitter.x_max
            phi_p  = fitter._compute_basis(res['B'], t_plot)
            y_plot = unscale(C_sc + res['A'] * phi_p[:, 1])

            ax_zoom.plot(x_plot, y_plot, '-', color=colors[model], linewidth=2, alpha=0.8, label=labels[model])
            ax_zoom.axhline(C, color=colors[model], linestyle='--', alpha=0.3)
            ax_zoom.fill_between([fitter.x_min, fitter.x_max * 1.5],
                                 C - sigma, C + sigma,
                                 color=colors[model], alpha=0.1)

            y_min_z = min(y_min_z, C - sigma)
            y_max_z = max(y_max_z, C + sigma)

            mask_p = (x_plot >= zoom_start) & (x_plot <= zoom_end)
            if np.any(mask_p):
                y_min_z = min(y_min_z, np.min(y_plot[mask_p]))
                y_max_z = max(y_max_z, np.max(y_plot[mask_p]))

        if inf_df is not None and y_col in inf_df.columns:
            inf_val = float(inf_df[y_col].iloc[0])
            err_val = float(err_df[y_col].iloc[0]) if (err_df is not None and y_col in err_df.columns) else 0.0
            x_band  = [fitter.x_min, fitter.x_max * 1.5]
            ax_zoom.axhline(inf_val, color='red', linestyle='-', linewidth=1.5,
                            alpha=0.7, label='CBS limit', zorder=6)
            if err_val:
                ax_zoom.fill_between(x_band,
                                     inf_val - err_val, inf_val + err_val,
                                     color='red', alpha=0.15, label='CBS ± err', zorder=5)
            y_min_z = min(y_min_z, inf_val - err_val)
            y_max_z = max(y_max_z, inf_val + err_val)

        if truth_val is not None:
            ax_zoom.axhline(truth_val, color='r', linestyle='--', linewidth=2, label='Exact')
            y_min_z = min(y_min_z, truth_val)
            y_max_z = max(y_max_z, truth_val)

        ax_zoom.set_xlim(zoom_start, zoom_end)
        if not (np.isinf(y_min_z) or np.isinf(y_max_z)):
            span = y_max_z - y_min_z
            if span == 0: span = abs(y_min_z) * 0.01 + 1e-10
            ax_zoom.set_ylim(y_min_z - 0.1 * span, y_max_z + 0.1 * span)

        ax_zoom.set_title(f"{y_col} - Zoomed Tail")
        ax_zoom.set_xlabel("Basis Size")
        ax_zoom.grid(True, alpha=0.3)

        # =======================================================
        # Right Panels (Cols 2, 3, 4): Linearized Space
        # =======================================================
        for model in ['exponential', 'sqrt_exponential', 'power_law']:  # ← sibling of Full/Zoom, NOT inside if
            ax_log = ax_log_axes[model]
            if model not in fitter.results:
                continue

            res = fitter.results[model]
            fitter.model_type = model
            color = colors[model]

            B, C, A = res['B'], res['C'], res['A']
            tx = fitter._make_tx(model)

            if fitter.is_increasing:
                diff = C - fitter.raw_y
            else:
                diff = fitter.raw_y - C

            valid = diff > 0
            tx_v      = tx[valid]
            ln_diff_v = np.log(diff[valid])

            n_pts = len(tx_v)
            ax_log.scatter(tx_v, ln_diff_v, color=color, s=40, zorder=4, label=f'Data ({n_pts} pts)')

            n_invalid = int(np.sum(~valid))
            if n_invalid:
                ax_log.scatter(tx[~valid],
                               np.full(n_invalid, ln_diff_v.min() if len(ln_diff_v) else 0),
                               color='red', marker='x', s=50, zorder=5,
                               label=f'Invalid (y≤C): {n_invalid}')

            ln_A  = np.log(abs(A)) if abs(A) > 1e-15 else 0.0
            slope = -B
            tx_line = np.linspace(tx_v.min() if n_pts else 0, tx_v.max() if n_pts else 1, 200)
            ln_line = ln_A + slope * tx_line

            r2 = res.get('r2_linearized', float('nan'))
            ax_log.plot(tx_line, ln_line, '-', color=color, linewidth=2,
                        label=f'Fit (R²={r2:.5f})\nB={B:.4f}')

            if len(ln_diff_v) > 3:
                from scipy import stats

                df_log     = n_pts - 2
                t_crit     = stats.t.ppf(0.975, df_log)
                resid_log  = ln_diff_v - (ln_A + slope * tx_v)
                sigma2_res = np.sum(resid_log ** 2) / df_log
                mean_tx    = np.mean(tx_v)
                ss_tx      = np.sum((tx_v - mean_tx) ** 2)
                if ss_tx == 0:
                    ss_tx = 1e-15

                var_line = sigma2_res * (1.0 / n_pts + (tx_line - mean_tx) ** 2 / ss_tx)
                var_pred = sigma2_res * (1.0 + 1.0 / n_pts + (tx_line - mean_tx) ** 2 / ss_tx)
                se_line  = np.sqrt(var_line)
                se_pred  = np.sqrt(var_pred)

                ax_log.fill_between(tx_line,
                                    ln_line - t_crit * se_pred,
                                    ln_line + t_crit * se_pred,
                                    color=color, alpha=0.08, label='95% PI Band')
                ax_log.fill_between(tx_line,
                                    ln_line - t_crit * se_line,
                                    ln_line + t_crit * se_line,
                                    color=color, alpha=0.2, label='95% CI Band')

            ax_log.set_title(titles_log[model], fontsize=11)
            ax_log.set_xlabel(x_labels_log[model], fontsize=10)
            ax_log.set_ylabel(r'$\ln(y - C)$', fontsize=10)
            ax_log.legend(fontsize=8)
            ax_log.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.subplots_adjust(top=0.95, hspace=0.3)

    if save_pdf:
        filename = save_pdf if isinstance(save_pdf, str) else f"{system_name}_fits.pdf"
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        print(f"Saved plot to {filename}")

    plt.show()


import time

def fit_system_summary(df, system_name, x_col='basis size', err_df=None, inf_df=None,
                        skip_cols=None, n_fit=None):
    if skip_cols is None:
        skip_cols = []

    y_cols = [c for c in df.columns if c != x_col and c not in skip_cols]
    if not y_cols:
        print("No columns found to fit!")
        return pd.DataFrame()

    rows = []
    t_start = time.perf_counter()

    for y_col in y_cols:
        t0 = time.perf_counter()
        fitter = extraplus(
            df=df, x_col=x_col, y_col=y_col,
            err_df=err_df, inf_df=inf_df, n_fit=n_fit,
            use_energy_b=True
        )
        fitter.fit_linearized(compute_uq=True, verbose=False)
        fit_time = time.perf_counter() - t0

        cbs_ref = cbs_err = np.nan
        if inf_df is not None and y_col in inf_df.columns:
            cbs_ref = float(inf_df[y_col].iloc[0])
            if err_df is not None and y_col in err_df.columns:
                cbs_err = float(err_df[y_col].iloc[0])

        if not fitter.results:
            rows.append({
                'system': system_name, 'column': y_col, 'model': None,
                'A': np.nan, 'B': np.nan, 'C': np.nan, 'sigma_C': np.nan, 'R2': np.nan,
                'exact': fitter.truth_val, 'cbs_ref': cbs_ref, 'cbs_err': cbs_err,
                'C_minus_exact': np.nan, 'z_score': np.nan,
                'fit_time_s': fit_time, 'status': 'fit_failed',
            })
            continue

        for model, res in fitter.results.items():
            C_sc    = res['C']
            sig_sc  = res.get('sigma_mc', 0.0)
            C       = fitter.y_min + fitter.y_range * float(C_sc)
            sigma_C = float(fitter.y_range * sig_sc)
            A_phys  = float(fitter.y_range * res['A'])

            diff_exact = (C - fitter.truth_val) if fitter.truth_val is not None else np.nan
            z = diff_exact / sigma_C if (sigma_C and not np.isnan(diff_exact)) else np.nan

            rows.append({
                'system':        system_name,
                'column':        y_col,
                'model':         model,
                'A':             A_phys,
                'B':             res['B'],
                'C':             C,
                'sigma_C':       sigma_C,
                'R2':            res.get('r2_linearized', np.nan),
                'exact':         fitter.truth_val,
                'cbs_ref':       cbs_ref,
                'cbs_err':       cbs_err,
                'C_minus_exact': diff_exact,
                'z_score':       z,
                'fit_time_s':    fit_time,
                'status':        'ok',
            })

    total_time = time.perf_counter() - t_start
    summary = pd.DataFrame(rows)
    print(f"Fitted {len(y_cols)} columns for {system_name} in {total_time:.3f}s "
          f"({1000*total_time/max(len(y_cols),1):.1f} ms/col)")
    return summary