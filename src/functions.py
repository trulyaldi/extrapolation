import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from extrapolator import VarProIRLS
from extrap import VarProLinearized

def upload_df(file_path, start_basis_size = 900):
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
        solver = VarProLinearized(main_df, x_col, y_col,
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
        solver = VarProLinearized(main_df, x_col, y_col,
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
        fitter = VarProLinearized(
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