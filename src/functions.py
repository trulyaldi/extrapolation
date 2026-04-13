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
                            inf_df=main_inf_df)
        solver.fit_linearized(verbose=False)

        tv = truth_val_for(y_col)
        te = truth_err_for(y_col)

        if te is not None:
            solver.err_df = solver.err_df.copy()
            solver.err_df[y_col] = te

        solver.plot(truth_val=tv)



def fit_and_plot_system(df, system_name, x_col='basis', err_df=None, inf_df=None, skip_cols=None, n_fit=None):
    """
    Fits all expectation value columns in a dataframe using VarProLinearized 
    and plots an N x 2 grid.
    
    Parameters:
    -----------
    df          : pd.DataFrame containing the data.
    system_name : str, name of the system (e.g., 'be_1de', 'li_2po').
    x_col       : str, the name of the basis size column.
    err_df      : pd.DataFrame, optional error dataframe.
    inf_df      : pd.DataFrame, optional truth value dataframe.
    skip_cols   : list, optional list of columns to ignore (e.g. ['Energy']).
    n_fit       : int, optional number of points to fit.
    """
    if skip_cols is None:
        skip_cols = []
        
    # Find all expectation value columns (exclude the x_axis and any skipped columns)
    y_cols = [col for col in df.columns if col != x_col and col not in skip_cols]
    N = len(y_cols)
    
    if N == 0:
        print("No columns found to fit!")
        return

    # -----------------------------------------------------------
    # 1. Formatting the Title
    # -----------------------------------------------------------
    def format_system_name(name):
        """Converts 'be_1de' to 'Be($^1D^e$)'"""
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

    # -----------------------------------------------------------
    # 2. Fit all columns
    # -----------------------------------------------------------
    fitters = []
    print(f"Fitting {N} columns for {formatted_name}...")
    for y_col in y_cols:
        # EXPLICITLY USING VarProLinearized
        fitter = VarProLinearized(
            df=df, x_col=x_col, y_col=y_col, 
            err_df=err_df, inf_df=inf_df, n_fit=n_fit
        )
        # compute_uq=False as requested
        fitter.fit_linearized(compute_uq=False, verbose=False)
        fitters.append(fitter)

    # -----------------------------------------------------------
    # 3. Setup the N x 2 Plot
    # -----------------------------------------------------------
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

    fig, axes = plt.subplots(nrows=N, ncols=2, figsize=(14, 5 * N), squeeze=False)
    
    # Set the bold overarching title exactly as requested
    fig.suptitle(formatted_name, fontsize=22, fontweight='bold')

    for idx, fitter in enumerate(fitters):
        y_col = fitter.y_col
        ax_full = axes[idx, 0]
        ax_zoom = axes[idx, 1]
        
        if not fitter.results:
            ax_full.text(0.5, 0.5, f"Fit failed for {y_col}", ha='center', va='center')
            continue

        truth_val = fitter.truth_val

        def unscale(y_sc):
            return fitter.y_min + fitter.y_range * np.asarray(y_sc, dtype=float)

        y_data = unscale(fitter.raw_y)
        
        # =======================================================
        # Left Panel: Full View
        # =======================================================
        ax_full.plot(fitter.raw_x, y_data, 'ko', label='Data', zorder=5, markersize=6)

        for model, res in fitter.results.items():
            fitter.model_type = model
            C_sc   = res['C']
            sig_sc = res.get('sigma_mc', 0.0)
            C      = float(unscale(C_sc))
            sigma  = float(fitter.y_range * sig_sc)

            x_plot  = np.linspace(fitter.x_min, fitter.x_max * 1.5, 200)
            t_plot  = x_plot / fitter.x_max
            phi_p   = fitter._compute_basis(res['B'], t_plot)
            y_plot  = unscale(C_sc + res['A'] * phi_p[:, 1])

            ax_full.plot(x_plot, y_plot, '-', color=colors[model], linewidth=2,
                         alpha=0.8, label=labels[model])
            if sigma > 0:
                ax_full.fill_between([fitter.x_min, fitter.x_max * 1.5],
                                     C - sigma, C + sigma,
                                     color=colors[model], alpha=0.1)
            ax_full.axhline(C, color=colors[model], linestyle='--', alpha=0.3)

        if truth_val is not None:
            ax_full.axhline(truth_val, color='r', linestyle=':', linewidth=2,
                            label=f'Truth ({truth_val:.8f})')
            if fitter.err_df is not None and y_col in fitter.err_df.columns:
                try:
                    te = fitter.err_df[y_col].values[-1]
                    ax_full.fill_between([fitter.x_min, fitter.x_max * 1.5],
                                         truth_val - te, truth_val + te,
                                         color='r', alpha=0.15, zorder=0,
                                         label=f'Ref Error (±{te:.1e})')
                except IndexError:
                    pass

        ax_full.set_xlabel("Basis Size", fontsize=12)
        ax_full.set_ylabel(y_col, fontsize=12)
        ax_full.grid(True, alpha=0.3)
        ax_full.legend()
        
        # =======================================================
        # Right Panel: Zoomed Tail
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

            ax_zoom.plot(x_plot, y_plot, '-', color=colors[model], linewidth=2,
                         alpha=0.8, label=labels[model])
            if sigma > 0:
                ax_zoom.fill_between([fitter.x_min, fitter.x_max * 1.5],
                                     C - sigma, C + sigma,
                                     color=colors[model], alpha=0.1)
            ax_zoom.axhline(C, color=colors[model], linestyle='--', alpha=0.3)

            y_min_z = min(y_min_z, C - sigma)
            y_max_z = max(y_max_z, C + sigma)

            mask_p = (x_plot >= zoom_start) & (x_plot <= zoom_end)
            if np.any(mask_p):
                y_min_z = min(y_min_z, np.min(y_plot[mask_p]))
                y_max_z = max(y_max_z, np.max(y_plot[mask_p]))

        if truth_val is not None:
            ax_zoom.axhline(truth_val, color='r', linestyle=':', linewidth=2,
                            label=f'Truth ({truth_val:.8f})')
            y_min_z = min(y_min_z, truth_val)
            y_max_z = max(y_max_z, truth_val)
            if fitter.err_df is not None and y_col in fitter.err_df.columns:
                try:
                    te = fitter.err_df[y_col].values[-1]
                    ax_zoom.fill_between([fitter.x_min, fitter.x_max * 1.5],
                                         truth_val - te, truth_val + te,
                                         color='r', alpha=0.15, zorder=0,
                                         label=f'Ref Error (±{te:.1e})')
                    y_min_z = min(y_min_z, truth_val - te)
                    y_max_z = max(y_max_z, truth_val + te)
                except IndexError:
                    pass

        ax_zoom.set_xlim(zoom_start, zoom_end)
        if not (np.isinf(y_min_z) or np.isinf(y_max_z)):
            span = y_max_z - y_min_z
            if span == 0:
                span = abs(y_min_z) * 0.01 + 1e-10
            ax_zoom.set_ylim(y_min_z - 0.1 * span, y_max_z + 0.1 * span)

        ax_zoom.set_xlabel("Basis Size", fontsize=12)
        ax_zoom.set_ylabel(y_col, fontsize=12)
        ax_zoom.grid(True, alpha=0.3)
        ax_zoom.legend()

    # Fixed whitespace issue here: Use a stable top margin that doesn't collapse on large N
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.subplots_adjust(top=0.95, hspace=0.3)
    plt.show()