from scipy.optimize import least_squares, lsq_linear
from sklearn.linear_model import HuberRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class VarProLinearized:
    """
    Curve fitter for models of the form y = A * f(x, B) + C using a
    linearization approach.

    Strategy
    --------
    For a fixed candidate asymptote C the model collapses to a linear
    relationship in log-space:

        Exponential      : ln(y - C) = ln(A)  - B * (x / x_max)
        Sqrt-Exponential : ln(y - C) = ln(A)  - B * sqrt(x / x_max)
        Power-Law        : ln(y - C) = ln(A)  - B * ln(x / x_max)

    The algorithm searches over a range of C values and picks the one that
    maximises the R² of the resulting linear fit, then extracts A and B from
    that best linear model.

    The results dictionary format is identical to VarProIRLS so that the
    `plot`, `plot_final_weights`, and `compute_uncertainty` methods are
    directly reusable.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, df, x_col, y_col, err_df=None, inf_df=None,
                 b_init=None, n_fit=None, use_energy_b=False):
        """
        Parameters
        ----------
        df       : DataFrame with the data.
        x_col    : Name of the independent-variable column.
        y_col    : Name of the dependent-variable column.
        err_df   : Optional DataFrame with per-column error estimates
                   (used only for plotting the reference error band).
        inf_df   : Optional DataFrame whose last row holds "truth" values
                   (used only for plotting / printing the reference line).
        b_init       : float or None.
                       If provided, B is **fixed** to this value for every
                       model and is never optimised.  The algorithm will only
                       search for the best asymptote C (and the resulting
                       amplitude A) given that fixed B.
                       If None (default), B is determined freely from the
                       best linear fit in log-space.
                       Ignored when use_energy_b=True and y_col != 'Energy'.
        n_fit        : Use only the first n_fit rows for fitting; scaling
                       uses all rows so the result is directly comparable.
        use_energy_b : bool, default False.
                       When True **and** y_col is not 'Energy', the fitter
                       first fits the 'Energy' column from the same df with a
                       free B, then pins that per-model B value when fitting
                       the current y_col.  This is useful when the decay rate
                       is most reliably determined from the energy and should
                       be shared across all other observables.
                       When y_col == 'Energy', or when False, the behaviour
                       is identical to the standard free/fixed-B logic.
        """
        x_all = df[x_col].values.astype(float)
        y_all = df[y_col].values.astype(float)

        self.y_col  = y_col
        self.x_col  = x_col
        self.err_df = err_df

        # ── truth value from inf_df ──────────────────────────────────────
        if inf_df is not None and y_col in inf_df.columns:
            self.truth_val = float(inf_df[y_col].iloc[-1])
        else:
            self.truth_val = None

        # ── y / x scaling uses ALL rows (consistent with VarProIRLS) ────
        self.y_min   = float(np.min(y_all))
        self.y_max   = float(np.max(y_all))
        self.y_range = self.y_max - self.y_min
        if self.y_range == 0:
            self.y_range = 1.0

        self.x_min   = float(x_all.min())
        self.x_max   = float(x_all.max())
        self.range_x = self.x_max - self.x_min

        # ── select fitting rows ──────────────────────────────────────────
        if n_fit is not None and n_fit < len(df):
            x_fit = x_all[:n_fit]
            y_fit = y_all[:n_fit]
        else:
            x_fit = x_all
            y_fit = y_all

        self.raw_x = x_fit
        self.raw_y = (y_fit - self.y_min) / self.y_range

        # ── trend detection (Huber regression on scaled data) ───────────
        huber  = HuberRegressor()
        denom  = self.raw_x.max() - self.raw_x.min()
        x_norm = (self.raw_x - self.raw_x.min()) / (denom if denom > 0 else 1.0)
        huber.fit(x_norm.reshape(-1, 1), self.raw_y)
        self.is_increasing = huber.coef_[0] > 0

        # ── fixed B (None = free optimisation) ───────────────────────────
        self.fixed_b      = float(b_init) if b_init is not None else None
        self.use_energy_b = bool(use_energy_b)

        # ── keep a reference to raw inputs so Energy can be re-fitted ────
        # (only used when use_energy_b=True and y_col != 'Energy')
        self._df    = df
        self._n_fit = n_fit

        # ── state ────────────────────────────────────────────────────────
        self.model_type = None   # set before each call to _compute_basis
        self.results    = {}

    # ------------------------------------------------------------------
    # Linearized predictor variable (depends on model)
    # ------------------------------------------------------------------

    def _make_tx(self, model_type):
        """
        Return the predictor variable used in the linearized regression.

        For basis phi(t) = f(t, B) with t = x / x_max:

            exponential      : f = exp(-B * t)          → tx = t
            sqrt_exponential : f = exp(-B * sqrt(t))    → tx = sqrt(t)
            power_law        : f = t^(-B)               → tx = ln(t)

        In each case  ln(y - C) = ln(A) + slope * tx  with slope = -B.
        """
        t = self.raw_x / self.x_max
        if model_type == 'exponential':
            return t
        elif model_type == 'sqrt_exponential':
            return np.sqrt(np.maximum(t, 0.0))
        elif model_type == 'power_law':
            return np.log(np.maximum(t, 1e-12))
        else:
            raise ValueError(f"Unknown model_type '{model_type}'")

    # ------------------------------------------------------------------
    # Basis matrix (identical to VarProIRLS)
    # ------------------------------------------------------------------

    def _compute_basis(self, B, t):
        """Design matrix [1, f(t, B)] with t = raw_x / x_max."""
        phi = np.zeros((len(t), 2))
        phi[:, 0] = 1.0
        if self.model_type == 'exponential':
            phi[:, 1] = np.exp(-B * t)
        elif self.model_type == 'sqrt_exponential':
            phi[:, 1] = np.exp(-B * np.sqrt(np.maximum(t, 0.0)))
        elif self.model_type == 'power_law':
            phi[:, 1] = np.power(np.maximum(t, 1e-12), -B)
        return phi

    # ------------------------------------------------------------------
    # Bounds helpers (identical to VarProIRLS – needed for compute_uncertainty)
    # ------------------------------------------------------------------

    def _get_A_bounds(self):
        if self.is_increasing:
            return [-np.inf, -np.inf], [np.inf, 0.0]
        else:
            return [-np.inf, 0.0],    [np.inf, np.inf]

    def _get_b_bounds(self):
        t_scaled  = self.raw_x / self.x_max
        t_min     = t_scaled.min()
        eps_basis = 1e-3

        if self.model_type == 'exponential':
            b_max = -np.log(eps_basis) / t_min
        elif self.model_type == 'sqrt_exponential':
            b_max = -np.log(eps_basis) / np.sqrt(max(t_min, 1e-12))
        elif self.model_type == 'power_law':
            b_max = np.log(1.0 / eps_basis) / np.log(1.0 / t_min + 1e-12)
        else:
            b_max = 20.0

        b_max = min(b_max, 20.0)
        b_max = max(b_max, 2.0)
        return 1.0, b_max

    # ------------------------------------------------------------------
    # Single-C linear fit
    # ------------------------------------------------------------------

    def _fit_for_C(self, C, tx, fixed_b=None):
        """
        Given a candidate asymptote C (in scaled y-space) and the
        linearised predictor tx, fit

            ln(|y - C|) = intercept + slope * tx

        When *fixed_b* is supplied the slope is pinned to ``-fixed_b`` and
        only the intercept is fitted (1-parameter OLS).

        Returns
        -------
        r2        : float  – R² of the linear fit (-inf if infeasible)
        intercept : float  – ln(A) or ln(-A)
        slope     : float  – equals -B (or -fixed_b when B is fixed)
        valid     : bool
        """
        y = self.raw_y

        # The sign convention depends on whether A is positive or negative.
        if self.is_increasing:
            diff = C - y          # C > y  →  A < 0
        else:
            diff = y - C          # y > C  →  A > 0

        if np.any(diff <= 0.0):
            return -np.inf, 0.0, 0.0, False

        ln_diff = np.log(diff)

        if fixed_b is not None:
            # ── fixed-slope OLS: only fit intercept ──────────────────────
            # ln(y-C) - (-fixed_b)*tx = intercept  →  intercept = mean(...)
            slope     = -fixed_b
            adjusted  = ln_diff - slope * tx
            intercept = float(adjusted.mean())
            ln_pred   = intercept + slope * tx
        else:
            # ── free OLS: fit both intercept and slope ───────────────────
            X = np.column_stack([np.ones(len(tx)), tx])
            try:
                coeffs, *_ = np.linalg.lstsq(X, ln_diff, rcond=None)
            except np.linalg.LinAlgError:
                return -np.inf, 0.0, 0.0, False
            intercept, slope = float(coeffs[0]), float(coeffs[1])
            ln_pred = X @ coeffs

        ss_res = float(np.sum((ln_diff - ln_pred) ** 2))
        ss_tot = float(np.sum((ln_diff - ln_diff.mean()) ** 2))
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

        return r2, intercept, slope, True

    # ------------------------------------------------------------------
    # C search
    # ------------------------------------------------------------------

    def _search_C(self, tx, fixed_b=None, n_coarse=400, n_fine=400):
        """
        Two-stage grid search: coarse then fine around the best coarse point.

        When *fixed_b* is not None it is forwarded to ``_fit_for_C`` so that
        the slope is pinned and only the intercept (amplitude A) is fitted.

        Returns best (r2, C, intercept, slope).
        """
        y      = self.raw_y
        y_min  = float(y.min())
        y_max  = float(y.max())
        margin = max(y_max - y_min, 1e-6)

        if self.is_increasing:
            C_lo = y_max + 1e-5 * margin
            C_hi = y_max + 0.25 * margin
        else:
            C_lo = y_min - 0.25  * margin
            C_hi = y_min - 1e-5 * margin

        # ── coarse pass ──────────────────────────────────────────────────
        C_grid  = np.linspace(C_lo, C_hi, n_coarse)
        best    = (-np.inf, C_grid[0], 0.0, 0.0)
        r2_grid = np.full(n_coarse, -np.inf)

        for i, C_cand in enumerate(C_grid):
            r2, intercept, slope, valid = self._fit_for_C(C_cand, tx, fixed_b)
            if not valid:
                continue
            r2_grid[i] = r2
            if r2 > best[0]:
                best = (r2, C_cand, intercept, slope)

        # ── fine pass around best coarse point ───────────────────────────
        best_idx = int(np.argmax(r2_grid))
        lo_idx   = max(0, best_idx - 3)
        hi_idx   = min(n_coarse - 1, best_idx + 3)
        C_fine   = np.linspace(C_grid[lo_idx], C_grid[hi_idx], n_fine)

        for C_cand in C_fine:
            r2, intercept, slope, valid = self._fit_for_C(C_cand, tx, fixed_b)
            if valid and r2 > best[0]:
                best = (r2, C_cand, intercept, slope)

        return best   # (r2, C, intercept, slope)

    # ------------------------------------------------------------------
    # Main fitting method
    # ------------------------------------------------------------------

    def fit_linearized(self, models=None, verbose=False, on_iteration=None):
        """
        Fit the specified models using the linearization approach.

        For each model type the algorithm:
          1. Searches over candidate C values.
          2. Transforms y → ln(|y - C|) and fits a linear model.
          3. Picks the C that maximises R² of the linear fit.
          4. Extracts A and B from the best linear model.

        Parameters
        ----------
        models       : list of str, default all three model types.
        verbose      : bool – print per-model summary.
        on_iteration : callable(dict) – called once per fitted model with a
                       results dict (same keys as VarProIRLS's on_iteration).

        Returns
        -------
        self.results : dict keyed by model name (same schema as VarProIRLS).
        """
        if models is None:
            models = ['exponential', 'sqrt_exponential', 'power_law']

        # ── Resolve per-model B from Energy fit if requested ─────────────
        # energy_b_map: dict[model_type -> float | None]
        # None means "use the normal free/fixed_b logic for this model".
        energy_b_map: dict = {m: None for m in models}

        should_use_energy_b = (
            self.use_energy_b
            and self.y_col != 'Energy'
            and 'Energy' in self._df.columns
        )

        if should_use_energy_b:
            if verbose:
                print(f"[use_energy_b] Fitting 'Energy' first to derive B "
                      f"for '{self.y_col}' ...")

            # Fit Energy with a free B (b_init=None, use_energy_b=False to
            # avoid infinite recursion) using the same x_col and n_fit.
            _energy_fitter = VarProLinearized(
                df          = self._df,
                x_col       = self.x_col,
                y_col       = 'Energy',
                err_df      = None,
                inf_df      = None,
                b_init      = None,        # always free for Energy anchor
                n_fit       = self._n_fit,
                use_energy_b= False,       # prevent recursion
            )
            _energy_results = _energy_fitter.fit_linearized(
                models  = models,
                verbose = verbose,
            )

            for m in models:
                if m in _energy_results:
                    energy_b_map[m] = float(_energy_results[m]['B'])
                    if verbose:
                        print(f"[use_energy_b]   {m:<22} B from Energy = "
                              f"{energy_b_map[m]:.6f}")

        # ── mode label for verbose output ────────────────────────────────
        if verbose:
            if should_use_energy_b:
                print(f"  [B anchored to Energy fit]")
            elif self.fixed_b is not None:
                print(f"  [B fixed = {self.fixed_b}]")
            else:
                print(f"  [B free]")

        for model_type in models:
            self.model_type = model_type
            tx = self._make_tx(model_type)

            # ── Determine which B source applies for this model ──────────
            # Priority: energy_b_map > fixed_b > free
            if energy_b_map.get(model_type) is not None:
                effective_fixed_b = energy_b_map[model_type]
            else:
                effective_fixed_b = self.fixed_b

            # ── 1. Find best C (pass effective_fixed_b so slope may be pinned)
            best_r2, best_C, intercept, slope = self._search_C(
                tx, fixed_b=effective_fixed_b
            )

            # ── 2. Extract parameters ────────────────────────────────────
            if effective_fixed_b is not None:
                B = effective_fixed_b
            else:
                B = max(-slope, 1e-6)

            # intercept = ln(A) for decreasing, ln(-A) for increasing
            if self.is_increasing:
                A = -np.exp(intercept)   # A < 0
            else:
                A = np.exp(intercept)    # A > 0

            C = best_C

            # ── 3. Compute predictions and goodness-of-fit in y-space ────
            t_scaled = self.raw_x / self.x_max
            phi      = self._compute_basis(B, t_scaled)
            y_pred   = C + A * phi[:, 1]

            resid  = self.raw_y - y_pred
            ssr    = float(np.sum(resid ** 2))
            dof    = max(1, len(self.raw_y) - 3)

            # ── 4. Store results (compatible with VarProIRLS schema) ─────
            self.results[model_type] = {
                'B':              B,
                'C':              C,
                'A':              A,
                'ssr':            ssr,
                'r2_linearized':  best_r2,
                't_scaled':       t_scaled.copy(),
                'sigma_noise':    float(np.sqrt(ssr / dof)),
                'y_pred':         y_pred.copy(),
                # Uniform weights (linearization does not produce IRLS weights)
                'final_weights':  np.ones(len(self.raw_x)),
            }

            if verbose:
                C_unscaled = self.y_min + self.y_range * C
                A_unscaled = self.y_range * A
                if energy_b_map.get(model_type) is not None:
                    b_tag = "from Energy"
                elif self.fixed_b is not None:
                    b_tag = "fixed"
                else:
                    b_tag = "free"
                print(
                    f"[{model_type:<20}] "
                    f"B={B:12.6f} ({b_tag})  "
                    f"C_scaled={C:12.8f}  C={C_unscaled:12.8f}  "
                    f"A={A_unscaled:12.6f}  "
                    f"R²(log)={best_r2:.6f}  "
                    f"SSR={ssr:.4e}"
                )

            if on_iteration is not None:
                on_iteration({
                    'model':     model_type,
                    'B':         B,
                    'C':         C,
                    'A':         A,
                    'r2':        best_r2,
                    'ssr':       ssr,
                    'y_pred':    y_pred.copy(),
                    'raw_x':     self.raw_x.copy(),
                    'raw_y':     self.raw_y.copy(),
                    't_scaled':  t_scaled.copy(),
                })

        return self.results

    # ------------------------------------------------------------------
    # Uncertainty quantification (parametric Monte Carlo – same as VarProIRLS)
    # ------------------------------------------------------------------

    def compute_uncertainty(self, n_bootstrap=80, confidence_level=80.0):
        """
        Parametric Monte Carlo uncertainty for the asymptote C.

        Adds 'sigma_mc', 'sigma_C_lower_unscaled', 'sigma_C_upper_unscaled'
        to each model's results dict.  Because the linearized fitter produces
        uniform IRLS weights, noise is assumed homoscedastic.
        """
        if not self.results:
            raise RuntimeError("No results found. Run fit_linearized() first.")

        lower_p = (100.0 - confidence_level) / 2.0
        upper_p = 100.0 - lower_p

        for model, res in self.results.items():
            self.model_type = model

            B_best   = res['B']
            y_fit    = res['y_pred']
            weights  = res['final_weights']
            t_scaled = res['t_scaled']

            lb, ub = self._get_A_bounds()

            # homoscedastic noise estimate (weights are all 1 here)
            resid     = self.raw_y - y_fit
            ssr_w     = float(np.sum(weights * resid ** 2))
            dof       = max(1, len(self.raw_y) - 3)
            sigma_base = np.sqrt(ssr_w / dof)
            eps        = 1e-12
            sigma_i    = sigma_base / np.sqrt(weights + eps)

            boot_C_values = []

            for _ in range(n_bootstrap):
                noise   = np.random.normal(0.0, sigma_i)
                y_synth = y_fit + noise

                def boot_residual(alpha_curr):
                    b_val  = alpha_curr[0]
                    phi    = self._compute_basis(b_val, t_scaled)
                    phi_w  = phi * np.sqrt(weights)[:, np.newaxis]
                    y_sw   = y_synth * np.sqrt(weights)
                    res_lin = lsq_linear(phi_w, y_sw, bounds=(lb, ub),
                                         method='bvls')
                    y_model = np.dot(phi, res_lin.x)
                    return (y_synth - y_model) * np.sqrt(weights)

                try:
                    b_lo, b_hi = self._get_b_bounds()
                    res_opt    = least_squares(boot_residual, x0=[B_best],
                                               bounds=(b_lo, b_hi),
                                               method='trf', loss='linear')
                    B_boot   = res_opt.x[0]
                    phi_b    = self._compute_basis(B_boot, t_scaled)
                    phi_bw   = phi_b * np.sqrt(weights)[:, np.newaxis]
                    y_sw     = y_synth * np.sqrt(weights)
                    final_lin = lsq_linear(phi_bw, y_sw, bounds=(lb, ub),
                                           method='bvls')
                    boot_C_values.append(float(final_lin.x[0]))
                except Exception:
                    continue

            if boot_C_values:
                arr          = np.array(boot_C_values)
                C_low_s      = np.percentile(arr, lower_p)
                C_high_s     = np.percentile(arr, upper_p)
                C_best_u     = self.y_min + self.y_range * res['C']
                C_low_u      = self.y_min + self.y_range * C_low_s
                C_high_u     = self.y_min + self.y_range * C_high_s
                res['sigma_C_lower_unscaled'] = abs(C_best_u - C_low_u)
                res['sigma_C_upper_unscaled'] = abs(C_high_u - C_best_u)
                max_err_s    = max(abs(res['C'] - C_low_s),
                                   abs(res['C'] - C_high_s))
                res['sigma_mc'] = max_err_s
            else:
                res['sigma_mc'] = 0.0

        return self.results

    # ------------------------------------------------------------------
    # Plotting (identical to VarProIRLS)
    # ------------------------------------------------------------------

    def plot(self, truth_val=None):
        if truth_val is None:
            truth_val = self.truth_val
        if not self.results:
            return

        def unscale(y_sc):
            return self.y_min + self.y_range * np.asarray(y_sc, dtype=float)

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

        y_data = unscale(self.raw_y)

        print("\n" + "=" * 85)
        print(f"{'Model':<22} | {'C (Asymptote)':<17} | "
              f"{'MC Uncertainty':<20} | {'Diff from Reference'}")
        print("=" * 85)

        for model, res in self.results.items():
            C_sc    = res['C']
            sig_sc  = res.get('sigma_mc', 0.0)
            C       = float(unscale(C_sc))
            sigma   = float(self.y_range * sig_sc)
            diff_str = f"{abs(C - truth_val):.2e}" if truth_val is not None else "-"
            print(f"{model:<22} | {C:<17.9f} | {sigma:<20.2e} | {diff_str}")

        # ---- Plot 1: full view ----------------------------------------
        plt.figure(figsize=(12, 7))
        plt.plot(self.raw_x, y_data, 'ko', label='Data', zorder=5,
                 markersize=6)

        for model, res in self.results.items():
            self.model_type = model
            C_sc   = res['C']
            sig_sc = res.get('sigma_mc', 0.0)
            C      = float(unscale(C_sc))
            sigma  = float(self.y_range * sig_sc)

            x_plot  = np.linspace(self.x_min, self.x_max * 1.5, 200)
            t_plot  = x_plot / self.x_max
            phi_p   = self._compute_basis(res['B'], t_plot)
            y_plot  = unscale(C_sc + res['A'] * phi_p[:, 1])

            plt.plot(x_plot, y_plot, '-', color=colors[model], linewidth=2,
                     alpha=0.8, label=labels[model])
            plt.fill_between([self.x_min, self.x_max * 1.5],
                             C - sigma, C + sigma,
                             color=colors[model], alpha=0.1)
            plt.axhline(C, color=colors[model], linestyle='--', alpha=0.3)

        if truth_val is not None:
            plt.axhline(truth_val, color='r', linestyle=':', linewidth=2,
                        label=f'Truth ({truth_val:.8f})')
            if self.err_df is not None and self.y_col in self.err_df.columns:
                try:
                    te = self.err_df[self.y_col].values[-1]
                    plt.fill_between([self.x_min, self.x_max * 1.5],
                                     truth_val - te, truth_val + te,
                                     color='r', alpha=0.15, zorder=0,
                                     label=f'Reference Error (±{te:.1e})')
                except IndexError:
                    pass

        plt.title(f"{self.y_col} – Full View  [Linearized Fitter]")
        plt.xlabel("Basis Size")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # ---- Plot 2: zoomed tail & extrapolation ----------------------
        plt.figure(figsize=(12, 7))
        plt.plot(self.raw_x, y_data, 'ko', label='Data', zorder=5,
                 markersize=6)

        zoom_start = self.x_min + 0.6 * self.range_x
        zoom_end   = self.x_max * 1.5
        y_min_z, y_max_z = np.inf, -np.inf

        mask = self.raw_x >= zoom_start
        if np.any(mask):
            y_min_z = min(y_min_z, np.min(y_data[mask]))
            y_max_z = max(y_max_z, np.max(y_data[mask]))

        for model, res in self.results.items():
            self.model_type = model
            C_sc   = res['C']
            sig_sc = res.get('sigma_mc', 0.0)
            C      = float(unscale(C_sc))
            sigma  = float(self.y_range * sig_sc)

            x_plot = np.linspace(self.x_min, self.x_max * 1.5, 200)
            t_plot = x_plot / self.x_max
            phi_p  = self._compute_basis(res['B'], t_plot)
            y_plot = unscale(C_sc + res['A'] * phi_p[:, 1])

            plt.plot(x_plot, y_plot, '-', color=colors[model], linewidth=2,
                     alpha=0.8, label=labels[model])
            plt.fill_between([self.x_min, self.x_max * 1.5],
                             C - sigma, C + sigma,
                             color=colors[model], alpha=0.1)
            plt.axhline(C, color=colors[model], linestyle='--', alpha=0.3)

            y_min_z = min(y_min_z, C - sigma)
            y_max_z = max(y_max_z, C + sigma)

            mask_p = (x_plot >= zoom_start) & (x_plot <= zoom_end)
            if np.any(mask_p):
                y_min_z = min(y_min_z, np.min(y_plot[mask_p]))
                y_max_z = max(y_max_z, np.max(y_plot[mask_p]))

        if truth_val is not None:
            plt.axhline(truth_val, color='r', linestyle=':', linewidth=2,
                        label=f'Truth ({truth_val:.8f})')
            y_min_z = min(y_min_z, truth_val)
            y_max_z = max(y_max_z, truth_val)
            if self.err_df is not None and self.y_col in self.err_df.columns:
                try:
                    te = self.err_df[self.y_col].values[-1]
                    plt.fill_between([self.x_min, self.x_max * 1.5],
                                     truth_val - te, truth_val + te,
                                     color='r', alpha=0.15, zorder=0,
                                     label=f'Reference Error (±{te:.1e})')
                    y_min_z = min(y_min_z, truth_val - te)
                    y_max_z = max(y_max_z, truth_val + te)
                except IndexError:
                    pass

        plt.xlim(zoom_start, zoom_end)
        if not (np.isinf(y_min_z) or np.isinf(y_max_z)):
            span = y_max_z - y_min_z
            if span == 0:
                span = abs(y_min_z) * 0.01 + 1e-10
            plt.ylim(y_min_z - 0.1 * span, y_max_z + 0.1 * span)

        plt.title(f"{self.y_col} – Zoomed Tail & Extrapolation  [Linearized Fitter]")
        plt.xlabel("Basis Size")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_log(self):
        """
        Plot the linearized (log-transformed) space for each fitted model.

        For each model type the data are transformed as:
            Exponential      : ln(y - C)  vs  (x / x_max)
            Sqrt-Exponential : ln(y - C)  vs  sqrt(x / x_max)
            Power-Law        : ln(y - C)  vs  ln(x / x_max)

        The fitted line  intercept + slope * tx  is overlaid.  In a perfect
        fit all points fall exactly on the line; deviations reveal where the
        model struggles.
        """
        if not self.results:
            raise RuntimeError("No results found. Run fit_linearized() first.")

        colors = {
            'exponential':      'blue',
            'sqrt_exponential': 'orange',
            'power_law':        'green',
        }
        x_labels = {
            'exponential':      r'$x \;/\; x_{\max}$',
            'sqrt_exponential': r'$\sqrt{x \;/\; x_{\max}}$',
            'power_law':        r'$\ln(x \;/\; x_{\max})$',
        }
        y_label_template = r'$\ln(y - C)$'
        titles = {
            'exponential':      r'Exponential: $\ln(y-C)$ vs $x/x_{max}$',
            'sqrt_exponential': r'Sqrt-Exp: $\ln(y-C)$ vs $\sqrt{x/x_{max}}$',
            'power_law':        r'Power-Law: $\ln(y-C)$ vs $\ln(x/x_{max})$',
        }

        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models,
                                 figsize=(6 * n_models, 5),
                                 squeeze=False)

        for ax, (model, res) in zip(axes[0], self.results.items()):
            self.model_type = model
            color = colors[model]

            B = res['B']
            C = res['C']
            A = res['A']

            tx = self._make_tx(model)

            # ── transform data ───────────────────────────────────────────
            if self.is_increasing:
                diff = C - self.raw_y
            else:
                diff = self.raw_y - C

            valid = diff > 0
            tx_v      = tx[valid]
            ln_diff_v = np.log(diff[valid])

            # ── fitted line parameters ───────────────────────────────────
            # ln(y - C) = ln(|A|) - B * tx
            ln_A     = np.log(abs(A)) if abs(A) > 1e-15 else 0.0
            slope    = -B
            tx_line  = np.linspace(tx_v.min(), tx_v.max(), 200)
            ln_line  = ln_A + slope * tx_line

            # ── scatter: colour valid vs clipped points ──────────────────
            n_invalid = int(np.sum(~valid))
            ax.scatter(tx_v, ln_diff_v, color=color, s=40, zorder=4,
                       label=f'Data ({len(tx_v)} pts)')
            if n_invalid:
                ax.scatter(tx[~valid],
                           np.full(n_invalid, ln_diff_v.min() if len(ln_diff_v) else 0),
                           color='red', marker='x', s=50, zorder=5,
                           label=f'Invalid (y≤C): {n_invalid}')

            # ── fitted line ──────────────────────────────────────────────
            r2 = res.get('r2_linearized', float('nan'))
            ax.plot(tx_line, ln_line, '-', color=color, linewidth=2,
                    label=f'Fit  (R²={r2:.5f})\nB={B:.4f}')

            # ── residual band (±1 σ in log-space) ────────────────────────
            if len(ln_diff_v) > 3:
                resid_log = ln_diff_v - (ln_A + slope * tx_v)
                sigma_log = float(np.std(resid_log))
                ax.fill_between(tx_line,
                                ln_line - sigma_log,
                                ln_line + sigma_log,
                                color=color, alpha=0.12,
                                label=f'±1σ (log) = {sigma_log:.3f}')

            ax.set_title(titles[model], fontsize=11)
            ax.set_xlabel(x_labels[model], fontsize=10)
            ax.set_ylabel(y_label_template, fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"{self.y_col} – Linearized Space  [Linearized Fitter]",
            fontsize=13, y=1.02
        )
        plt.tight_layout()
        plt.show()

    def plot_final_weights(self, model_type='exponential', sort_by_x=True,
                           normalize=False):
        """
        For the linearized fitter all weights are uniform (= 1), so this
        plot primarily serves as a visual sanity check.
        """
        if not self.results:
            raise RuntimeError("No results found. Run fit_linearized() first.")
        if model_type not in self.results:
            raise ValueError(
                f"Model '{model_type}' not found. "
                f"Available: {list(self.results.keys())}"
            )

        w = np.asarray(self.results[model_type]['final_weights'], dtype=float)
        x = np.asarray(self.raw_x, dtype=float)

        if normalize:
            s = w.sum()
            if s > 0:
                w = w / s

        idx    = np.argsort(x) if sort_by_x else np.arange(len(x))
        x_plot = x[idx]
        w_plot = w[idx]

        plt.figure(figsize=(12, 5))
        plt.bar(np.arange(len(x_plot)), w_plot)
        plt.title(
            f"Final Weights – {model_type}  "
            f"[Linearized Fitter: uniform weights]"
        )
        plt.xlabel(
            "Data point index"
            + (" (sorted by x)" if sort_by_x else " (original order)")
        )
        plt.ylabel("Weight" + (" (normalized)" if normalize else ""))
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()