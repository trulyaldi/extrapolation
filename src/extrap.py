from scipy.optimize import least_squares, lsq_linear
from sklearn.linear_model import HuberRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class VarProLinearized:

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, df, x_col, y_col, err_df=None, inf_df=None,
                 b_init=None, n_fit=None, use_energy_b=False):

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

       
        huber = HuberRegressor()
        denom = self.raw_x.max() - self.raw_x.min()
        x_norm = (self.raw_x - self.raw_x.min()) / (denom if denom > 0 else 1.0)
        
        # Create weights based on basis size (x^2 strongly prefers the asymptote)
        weights = self.raw_x ** 3
        weights = weights / weights.max()  # Normalize weights to avoid numerical overflow
        
        # Fit using all points, but with physical weighting
        huber.fit(x_norm.reshape(-1, 1), self.raw_y, sample_weight=weights)
        self.is_increasing = huber.coef_[0] > 0

        # ── fixed B (None = free optimisation) ───────────────────────────
        self.fixed_b      = float(b_init) if b_init is not None else None
        self.use_energy_b = bool(use_energy_b)

     
        self._df    = df
        self._n_fit = n_fit

        # ── state ────────────────────────────────────────────────────────
        self.model_type = None   
        self.results    = {}
        print(f' this trend is: {self.is_increasing}')

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
        y = self.raw_y

        y_range = float(np.max(y) - np.min(y))
        n_tail  = max(3, len(y) // 5)
        sigma   = max(float(np.std(y[-n_tail:])), np.finfo(float).eps * float(np.abs(y).max()))
        snr     = y_range / sigma

        if snr <= 1.0:
            raise ValueError(f"Data is noise-dominated: SNR={snr:.3f} <= 1.0, cannot constrain b.")

        eps_basis = 1.0 / snr

        tx          = self._make_tx(self.model_type)
        tx_min      = float(tx.min())
        tx_max      = float(tx.max())
        tx_span     = tx_max - tx_min

        if tx_span <= 0:
            raise ValueError("tx domain has zero span, cannot constrain b.")

        tx_sorted   = np.sort(tx)
        dtx         = np.diff(tx_sorted)
        dtx_positive = dtx[dtx > tx_max * np.finfo(float).eps ** 0.5]
        delta_tx    = float(dtx_positive.min()) if len(dtx_positive) > 0 else tx_span

        b_max = min(
            -np.log(eps_basis) / delta_tx,
            -np.log(np.finfo(float).eps) / delta_tx
        )
        b_min = -np.log1p(-1.0 / snr) / tx_span

        if b_max <= b_min:
            raise ValueError(
                f"b bounds degenerate: b_min={b_min:.3e} >= b_max={b_max:.3e}. "
                f"SNR={snr:.1f}, delta_tx={delta_tx:.3e}"
            )

        return b_min, b_max

    # ------------------------------------------------------------------
    # Single-C linear fit
    # ------------------------------------------------------------------

    def _fit_for_C(self, C, tx, fixed_b=None):
        y = self.raw_y
        diff = (C - y) if self.is_increasing else (y - C)

        # 1. GEOMETRIC FILTER: Drop points that physically cross the candidate asymptote
        valid_mask = diff > 0
        diff_v     = diff[valid_mask]
        tx_v       = tx[valid_mask]

        # 2. SPAN VALIDATION: Ensure surviving points still describe the curve
        tx_full_span  = tx.max() - tx.min()
        tx_valid_span = tx_v.max() - tx_v.min() if len(tx_v) > 1 else 0.0
        
        if tx_valid_span < 0.5 * tx_full_span:
            return -np.inf, 0.0, 0.0, False

        # 3. LOG-SPACE REGRESSION 
        # By doing this unweighted, we inherently force the optimizer to care about 
        # the tail, because tiny absolute differences become massive log differences.
        ln_diff = np.log(diff_v)

        if fixed_b is not None:
            slope     = -fixed_b
            intercept = float((ln_diff - slope * tx_v).mean())
            ln_pred   = intercept + slope * tx_v
        else:
            X = np.column_stack([np.ones(len(tx_v)), tx_v])
            try:
                coeffs, *_ = np.linalg.lstsq(X, ln_diff, rcond=None)
            except np.linalg.LinAlgError:
                return -np.inf, 0.0, 0.0, False
            intercept, slope = float(coeffs[0]), float(coeffs[1])
            ln_pred = X @ coeffs

        if fixed_b is None:
            tx_range  = max(tx_v.max() - tx_v.min(), 1e-12) 
            min_slope = -1e-3 / tx_range 
            if slope > min_slope:
                return -np.inf, 0.0, 0.0, False

        # 4. LOG-SPACE R^2 EVALUATION
        # We MUST evaluate candidates in log-space so the grid search ranks them 
        # based on tail-convergence accuracy rather than just head-magnitude accuracy.
        ss_res = float(np.sum((ln_diff - ln_pred) ** 2))
        ss_tot = float(np.sum((ln_diff - ln_diff.mean()) ** 2))
        r2_log = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

        return r2_log, intercept, slope, True


    def _search_C(self, tx, fixed_b=None, n_coarse=1000, n_fine=500, window=3, r2_epsilon=0.01):
        y = self.raw_y

        # --- THE FIX: Tail-Extreme Anchoring ---
        # Anchor to the extreme (max/min) of the chronological tail instead of the median.
        # This guarantees the candidate asymptote C starts strictly OUTSIDE the final 
        # data points, meaning C - y > 0 for all tail points and none are dropped.
        n_tail    = max(3, len(y) // 5)
        tail_vals = y[-n_tail:]
        
        if self.is_increasing:
            y_bound = float(np.max(tail_vals))
        else:
            y_bound = float(np.min(tail_vals))

        y_range    = max(float(y.max() - y.min()), 1e-10)
        tail_noise = max(float(np.std(tail_vals)), 1e-10)

        # Added a 1e-12 floor to dist_min to prevent log(0) domain errors 
        # on perfectly smooth synthetic data where tail_noise might approach 0.
        dist_min = max(tail_noise * 1e-3, 1e-12)
        
        # Expanded upper bound to 10x to avoid truncating slowly decaying curves
        dist_max = y_range * 10.0

        # log_dists spans ~4 decades. n_coarse=1000 yields ~250 pts/decade.
        log_dists = np.logspace(np.log10(dist_min), np.log10(dist_max), n_coarse)

        if self.is_increasing:
            C_grid = y_bound + log_dists
        else:
            C_grid = y_bound - log_dists

        r2_grid = np.full(n_coarse, -np.inf)

        # ── 1. Coarse Pass ──────────────────────────────────────────────
        for i, C_cand in enumerate(C_grid):
            r2, intercept, slope, valid = self._fit_for_C(C_cand, tx, fixed_b)
            if valid:
                r2_grid[i] = r2

        # ── 2. Find and Filter Local Maxima ────────────────────────────
        local_maxima = [
            i for i in range(1, n_coarse - 1)
            if r2_grid[i] > r2_grid[i - 1]
            and r2_grid[i] > r2_grid[i + 1]
            and r2_grid[i] > -np.inf
        ]

        if local_maxima:
            local_maxima = sorted(local_maxima, key=lambda idx: r2_grid[idx], reverse=True)[:3]
        elif np.any(r2_grid > -np.inf):
            local_maxima = [int(np.argmax(r2_grid))]
        else:
            return (-np.inf, C_grid[0], 0.0, 0.0)

        best = (-np.inf, C_grid[0], 0.0, 0.0)
        evaluated_intervals = []

        # ── 3. Fine Pass ────────────────────────────────────────────────
        for idx in local_maxima:
            peak_r2 = r2_grid[idx]
            
            lo_idx = idx
            last_good = idx
            patience = 2
            while lo_idx > 0:
                lo_idx -= 1
                val = r2_grid[lo_idx]
                if val >= peak_r2 - r2_epsilon:
                    last_good = lo_idx
                    patience = 2
                elif val == -np.inf and patience > 0:
                    patience -= 1
                else:
                    lo_idx = last_good
                    break
            else:
                lo_idx = last_good
                    
            hi_idx = idx
            last_good = idx
            patience = 2
            while hi_idx < n_coarse - 1:
                hi_idx += 1
                val = r2_grid[hi_idx]
                if val >= peak_r2 - r2_epsilon:
                    last_good = hi_idx
                    patience = 2
                elif val == -np.inf and patience > 0:
                    patience -= 1
                else:
                    hi_idx = last_good
                    break
            else:
                hi_idx = last_good

            lo_idx = min(lo_idx, max(0, idx - window))
            hi_idx = max(hi_idx, min(n_coarse - 1, idx + window))

            if any(lo <= hi_idx and lo_idx <= hi for lo, hi in evaluated_intervals):
                continue
                
            evaluated_intervals.append((lo_idx, hi_idx))

            fine_dists = np.logspace(
                np.log10(log_dists[lo_idx]),
                np.log10(log_dists[hi_idx]),
                n_fine
            )
            
            if self.is_increasing:
                C_fine = y_bound + fine_dists
            else:
                C_fine = y_bound - fine_dists

            for C_cand in C_fine:
                r2, intercept, slope, valid = self._fit_for_C(C_cand, tx, fixed_b)
                if valid and r2 > best[0]:
                    best = (r2, C_cand, intercept, slope)

        return best
    # ------------------------------------------------------------------
    # Helper to classify property types for B vs B/2 scaling
    # ------------------------------------------------------------------
    def _is_singular_property(self, col_name):
        # 1. Force drach_MV to be singular
        if col_name == 'drach_MV':
            return True
            
        # 2. Make all other drach_ properties nonsingular 

        if 'drach_' in col_name:
            return False
            
        # 3. Standard singular keywords
        singular_keywords = ['MV', 'delta', 'prval']
        for kw in singular_keywords:
            if kw in col_name:
                return True
                
        # 4. Everything else is nonsingular
        return False

    # ------------------------------------------------------------------
    # Main fitting method
    # ------------------------------------------------------------------

    def fit_linearized(self, models=None, verbose=False, compute_uq=True, on_iteration=None):

        if models is None:
            models = ['exponential', 'sqrt_exponential', 'power_law']

        # ── Resolve per-model B from Energy fit if requested ─────────────
        # energy_b_map: dict[model_type -> float | None]
        energy_b_map: dict = {m: None for m in models}

        should_use_energy_b = (
            self.use_energy_b
            and self.y_col != 'Energy'
            and 'Energy' in self._df.columns
        )
        
        # Check if the current column is a singular property (needs B/2)
        is_singular = self._is_singular_property(self.y_col)

        if should_use_energy_b:
            if verbose:
                prop_type = "singular (using B/2)" if is_singular else "nonsingular (using B)"
                print(f"[use_energy_b] Fitting 'Energy' first to derive B "
                      f"for '{self.y_col}' [{prop_type}] ...")

            _energy_fitter = VarProLinearized(
                df          = self._df,
                x_col       = self.x_col,
                y_col       = 'Energy',
                err_df      = None,
                inf_df      = None,
                b_init      = None,        
                n_fit       = self._n_fit,
                use_energy_b= False,       
            )
            _energy_results = _energy_fitter.fit_linearized(
                models  = models,
                verbose = verbose,
            )

            for m in models:
                if m in _energy_results:
                    energy_b = float(_energy_results[m]['B'])
                    
                    # Apply the scaling here: B/2 for singular, B for nonsingular
                    energy_b_map[m] = energy_b / 2.0 if is_singular else energy_b
                    
                    if verbose:
                        scaling_str = "B/2" if is_singular else "B"
                        print(f"[use_energy_b]   {m:<22} {scaling_str} from Energy = "
                              f"{energy_b_map[m]:.6f}")

        if verbose:
            if should_use_energy_b:
                print(f"  [B anchored to Energy fit]")
            elif self.fixed_b is not None:
                print(f"  [B fixed = {self.fixed_b}]")
            else:
                print(f"  [B free]")

        # ── Start Fitting (Retry loop removed) ───────────────────────────
        self.results = {}

        for model_type in models:
            self.model_type = model_type
            tx = self._make_tx(model_type)

            if energy_b_map.get(model_type) is not None:
                effective_fixed_b = energy_b_map[model_type]
            else:
                effective_fixed_b = self.fixed_b

            # ── 1. Find best C ───────────────────────────────────────
            best_r2, best_C, intercept, slope = self._search_C(
                tx, fixed_b=effective_fixed_b
            )

            # ── 2. Extract parameters ────────────────────────────────
            if effective_fixed_b is not None:
                B = effective_fixed_b
            else:
                B = max(-slope, 1e-6)

            if self.is_increasing:
                A = -np.exp(intercept)   # A < 0
            else:
                A = np.exp(intercept)    # A > 0

            C = best_C

            # ── 3. Compute predictions and goodness-of-fit ───────────
            t_scaled = self.raw_x / self.x_max
            phi      = self._compute_basis(B, t_scaled)
            y_pred   = C + A * phi[:, 1]

            resid  = self.raw_y - y_pred
            ssr    = float(np.sum(resid ** 2))
            dof    = max(1, len(self.raw_y) - 3)

            # ── 4. Store results ─────────────────────────────────────
            self.results[model_type] = {
                'B':              B,
                'C':              C,
                'A':              A,
                'ssr':            ssr,
                'r2_linearized':  best_r2,
                't_scaled':       t_scaled.copy(),
                'sigma_noise':    float(np.sqrt(ssr / dof)),
                'y_pred':         y_pred.copy(),
                'final_weights':  np.ones(len(self.raw_x)),
            }

            if verbose:
                C_unscaled = self.y_min + self.y_range * C
                A_unscaled = self.y_range * A
                
                if energy_b_map.get(model_type) is not None:
                    b_tag = f"from Energy ({'B/2' if is_singular else 'B'})"
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

        if compute_uq and self.results:
            self.compute_uncertainty()
            
        return self.results

    # ------------------------------------------------------------------
    # Uncertainty quantification (B-Boundary Error Propagation)
    # ------------------------------------------------------------------

    def compute_uncertainty(self):
        if not self.results:
            raise RuntimeError("No results found. Run fit_linearized() first.")

        eps = np.finfo(float).eps

        for model, res in self.results.items():
            self.model_type = model

            B = res['B']
            A = res['A']
            C = res['C']

            tx = self._make_tx(model)

            if self.is_increasing:
                diff = C - self.raw_y
            else:
                diff = self.raw_y - C

            valid = diff > 0
            tx_v  = tx[valid]
            diff_v = diff[valid]
            z_v   = np.log(diff_v)
            n     = len(tx_v)

            if n < 3:
                res['sigma_mc'] = float(np.max(np.abs(np.diff(self.raw_y))))
                continue

            # ── Log-space residuals and noise variance ──────────────────────
            ln_A      = np.log(abs(A) + eps)
            z_pred    = ln_A - B * tx_v
            log_resid = z_v - z_pred
            dof       = max(1, n - 3)   # 3 parameters: C, ln|A|, B
            sigma_log_sq = float(np.sum(log_resid ** 2) / dof)

            # ── NLS Jacobian of r_i = ln(y_i - C) - ln|A| + B*t_i ─────────
            # dr/dC     = -1 / (y_i - C)    [sign: increasing flips diff]
            # dr/dln|A| = -1
            # dr/dB     =  t_i
            #
            # Shape: (n, 3) with columns [C, ln|A|, B]
            sign = -1.0 if self.is_increasing else 1.0
            J = np.column_stack([
                sign / diff_v,          # dr/dC  (= -1/(y_i - C) with correct sign)
                -np.ones(n),            # dr/dln|A|
                tx_v                    # dr/dB
            ])

            # ── Joint covariance: Cov(θ) = σ² (JᵀJ)⁻¹ ─────────────────────
            JTJ = J.T @ J
            try:
                JTJ_inv = np.linalg.inv(JTJ)
            except np.linalg.LinAlgError:
                res['sigma_mc'] = float(np.max(np.abs(np.diff(self.raw_y))))
                continue

            # σ_C is the square root of the (0,0) entry → variance of C
            var_C = sigma_log_sq * JTJ_inv[0, 0]

            if var_C <= 0.0 or np.isnan(var_C):
                res['sigma_mc'] = float(np.max(np.abs(np.diff(self.raw_y))))
                continue

            sigma_C_scaled = float(np.sqrt(var_C))
            res['sigma_mc'] = sigma_C_scaled

            # Store auxiliary diagnostics if useful
            res['sigma_B']   = float(np.sqrt(sigma_log_sq * JTJ_inv[2, 2]))
            res['corr_CB']   = (JTJ_inv[0, 2]
                                / (np.sqrt(JTJ_inv[0, 0]) * np.sqrt(JTJ_inv[2, 2]) + eps))

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
              f"{'Uncertainty':<20} | {'Diff from Reference'}")
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
