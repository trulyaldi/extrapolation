from scipy.optimize import least_squares, lsq_linear
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from scipy.optimize import minimize_scalar, brentq


class VarProLinearized:

    def _determine_direction(self):

        tau, _ = kendalltau(self.raw_x, self.raw_y)
        return tau > 0

    def __init__(self, df, x_col, y_col, err_df=None, inf_df=None,
                 b_init=None, n_fit=None, use_energy_b=True):

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


        self.is_increasing = self._determine_direction()

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


    def _search_C(self, tx, fixed_b=None, n_coarse=1250, xtol=1e-15, polish=True):

        y = self.raw_y
    
        # --- anchor on the chronological-tail extreme (unchanged) --------------
        n_tail    = max(3, len(y) // 5)
        tail_vals = y[-n_tail:]
        y_bound   = (float(np.max(tail_vals)) if self.is_increasing
                    else float(np.min(tail_vals)))
    
        y_range    = max(float(y.max() - y.min()), 1e-10)
        tail_noise = max(float(np.std(tail_vals)), 1e-10)
        dist_min   = max(tail_noise * 1e-3, 1e-12)
        dist_max   = y_range
    
        dists  = np.logspace(np.log10(dist_min), np.log10(dist_max), n_coarse)
        s_grid = np.log(dists)                        # natural-log distance coord.
    
        def C_of_s(s):
            d = np.exp(s)
            return (y_bound + d) if self.is_increasing else (y_bound - d)
    
        # live "best ever" tracker: (r2, C, intercept, slope) -- the safety net
        self._best_C_track = (-np.inf, C_of_s(s_grid[0]), 0.0, 0.0)
        PENALTY = 1e30
    
        def neg_r2(s):                                # scalar objective for Brent
            r2, intercept, slope, valid = self._fit_for_C(C_of_s(s), tx, fixed_b)
            if not valid:
                return PENALTY
            if r2 > self._best_C_track[0]:
                self._best_C_track = (r2, C_of_s(s), intercept, slope)
            return -r2
    
        # --- 1. coarse pass: ONE vectorised evaluation over the whole grid -----
        C_grid       = C_of_s(s_grid)
        tx_full_span = float(tx.max() - tx.min())
        r2_grid      = self._coarse_scan(C_grid, tx, fixed_b, tx_full_span)
    
        # seed the tracker with the coarse-best point (exact lstsq coeffs)
        if np.any(np.isfinite(r2_grid)):
            bi = int(np.argmax(np.where(np.isfinite(r2_grid), r2_grid, -np.inf)))
            r2b, icb, slb, vb = self._fit_for_C(C_grid[bi], tx, fixed_b)
            if vb:
                self._best_C_track = (r2b, C_grid[bi], icb, slb)
    
        # --- 2. rank up to 3 strongest local maxima ----------------------------
        local_maxima = [
            i for i in range(1, n_coarse - 1)
            if r2_grid[i] > r2_grid[i - 1]
            and r2_grid[i] > r2_grid[i + 1]
            and np.isfinite(r2_grid[i])
        ]
        if local_maxima:
            local_maxima = sorted(local_maxima, key=lambda j: r2_grid[j],
                                reverse=True)[:3]
        elif np.any(np.isfinite(r2_grid)):
            finite = np.where(np.isfinite(r2_grid), r2_grid, -np.inf)
            local_maxima = [int(np.argmax(finite))]
        else:
            return self._best_C_track            # nothing valid anywhere
    
        # --- 3. Brent refinement (+ optional derivative polish) per peak --------
        for idx in local_maxima:
            lo = max(idx - 1, 0)
            hi = min(idx + 1, n_coarse - 1)
            s_lo, s_mid, s_hi = s_grid[lo], s_grid[idx], s_grid[hi]
    
            if not (s_lo < s_mid < s_hi):        # boundary/degenerate bracket
                neg_r2(s_mid)
                continue
    
            try:
                res = minimize_scalar(
                    neg_r2,
                    bracket=(s_lo, s_mid, s_hi),
                    method='brent',
                    options={'xtol': xtol, 'maxiter': 500},
                )
                s_star = float(res.x)
            except Exception:
                s_star = s_mid
    
            if polish:
                s_star = self._polish_s(neg_r2, s_star, s_lo, s_hi)
    
            neg_r2(s_star)
    
        return self._best_C_track
    
    
    def _polish_s(self, neg_r2, s0, s_lo, s_hi):

        f = lambda s: -neg_r2(s)                 # = +r2 (the quantity to maximise)
        h = max(abs(s0), 1.0) * (np.finfo(float).eps ** (1.0 / 3.0))
    
        def g(s):                                # dr2/ds via central difference
            fp, fm = f(s + h), f(s - h)
            if fp <= -1e29 or fm <= -1e29:       # stepped into an invalid region
                return np.nan
            return (fp - fm) / (2.0 * h)
    
        a = max(s0 - 4.0 * h, s_lo)
        b = min(s0 + 4.0 * h, s_hi)
        if not (a < b):
            return s0
        try:
            ga, gb = g(a), g(b)
            if np.isfinite(ga) and np.isfinite(gb) and ga * gb < 0.0:
                return float(brentq(g, a, b, xtol=1e-15, maxiter=200))
        except Exception:
            pass
        return s0
    
    def _coarse_scan(self, C_grid, tx, fixed_b, tx_full_span):

        y = self.raw_y                                       # (N,)
        M = C_grid.shape[0]
    
        if self.is_increasing:
            diff = C_grid[:, None] - y[None, :]              # (M, N)
        else:
            diff = y[None, :] - C_grid[:, None]
    
        valid = diff > 0.0
        w     = valid.astype(np.float64)
        n     = w.sum(axis=1)                                # valid count per row
        txb   = np.broadcast_to(tx, diff.shape)              # (M, N) view
    
        # per-row valid tx-span (span guard + min-slope range)
        tx_hi = np.where(valid, txb, -np.inf).max(axis=1)
        tx_lo = np.where(valid, txb,  np.inf).min(axis=1)
        tx_valid_span = np.where(n > 1, tx_hi - tx_lo, 0.0)
    
        # ln(diff) on valid entries only; 0 elsewhere (weight removes them)
        ld = np.log(np.where(valid, diff, 1.0))
        ld = np.where(valid, ld, 0.0)
    
        # masked regression sums
        Sx  = (txb * w).sum(axis=1)
        Sxx = (txb * txb * w).sum(axis=1)
        Sy  = ld.sum(axis=1)
        Sxy = (txb * ld).sum(axis=1)
        Syy = (ld * ld).sum(axis=1)
    
        nsafe = np.where(n > 0, n, 1.0)
        Sxx_c = Sxx - Sx * Sx / nsafe                        # Σ(tx-mean)^2 (valid)
        Syy_c = Syy - Sy * Sy / nsafe                        # Σ(ld-mean)^2 (valid)
        Sxy_c = Sxy - Sx * Sy / nsafe
    
        ok = (n >= 2) & (tx_valid_span >= 0.5 * tx_full_span) & (Sxx_c > 0.0)
    
        if fixed_b is None:
            slope     = np.divide(Sxy_c, Sxx_c, out=np.zeros(M), where=ok)
            tx_range  = np.maximum(tx_hi - tx_lo, 1e-12)
            min_slope = -1e-3 / tx_range
            ok        = ok & (slope <= min_slope)            # free-b slope guard
            good      = ok & (Syy_c > 1e-15)
            r2_vals   = np.divide(Sxy_c * Sxy_c, Sxx_c * Syy_c,
                                out=np.zeros(M), where=good)
            r2 = np.where(ok, np.where(Syy_c > 1e-15, r2_vals, 0.0), -np.inf)
        else:
            slope     = -float(fixed_b)
            intercept = np.divide(Sy - slope * Sx, nsafe, out=np.zeros(M), where=ok)
            # SS_res = Σ(ld - intercept - slope*tx)^2  expanded via the masked sums
            ss_res = (Syy + n * intercept * intercept + slope * slope * Sxx
                    - 2.0 * intercept * Sy - 2.0 * slope * Sxy
                    + 2.0 * intercept * slope * Sx)
            good    = ok & (Syy_c > 1e-15)
            r2_vals = 1.0 - np.divide(ss_res, Syy_c, out=np.ones(M), where=good)
            r2 = np.where(ok, np.where(Syy_c > 1e-15, r2_vals, 0.0), -np.inf)
    
        return r2
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
                use_energy_b= True,       
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
    def compute_uncertainty(self, tail_frac=0.5, k_cov=1.0,
                                q=0.84, floor_frac=0.3,
                                use_PI=True, robust_scale=True):
            if not self.results:
                raise RuntimeError("No results found. Run fit_linearized() first.")

            for model, res in self.results.items():
                self.model_type = model
                B, A, C = res['B'], res['A'], res['C']
                tx = self._make_tx(model)

                diff  = (C - self.raw_y) if self.is_increasing else (self.raw_y - C)
                valid = diff > 0
                tx_v, diff_v = tx[valid], diff[valid]
                Nv = len(tx_v)

                if Nv < 3:
                    fb = float(np.max(np.abs(np.diff(self.raw_y))))
                    # FIXED: Added missing keys to prevent downstream KeyErrors
                    res.update(dict(sigma_C=fb, sigma_mc=fb,
                                    sigma_C_plus=fb, sigma_C_minus=fb, 
                                    sigma_B=0.0, sigma_ln_A=0.0, 
                                    sigma_log=0.0, corr_CB=0.0))
                    continue

                ln_diff_v   = np.log(diff_v)
                ln_pred_v   = np.log(np.abs(A)) - B * tx_v
                residuals_v = ln_diff_v - ln_pred_v

                # (Channel 2) robust log-residual scale: pre-asymptotic bias -> not noise
                if robust_scale:
                    mad = np.median(np.abs(residuals_v - np.median(residuals_v)))
                    sigma2_res = (1.4826 * mad) ** 2
                else:
                    sigma2_res = np.sum(residuals_v ** 2) / (Nv - 2)

                X_v      = np.column_stack((np.ones(Nv), -tx_v))
                # FIXED: Using pinv adds protection against near-singular matrix errors
                cov_beta = sigma2_res * np.linalg.pinv(X_v.T @ X_v)
                var_ln_A, var_B, cov_lnA_B = cov_beta[0,0], cov_beta[1,1], cov_beta[0,1]

                # confidence (mean) variance + (use_PI) prediction term
                var_fit = var_ln_A + (tx_v**2)*var_B - 2.0*tx_v*cov_lnA_B
                var_z   = var_fit + (sigma2_res if use_PI else 0.0)
                se_z    = np.sqrt(np.maximum(var_z, 0.0))

                P_nom   = np.exp(np.log(np.abs(A)) - B * tx_v)
                d_plus  = P_nom * (1.0 - np.exp(-k_cov * se_z))   # Lower curve error boundary delta
                d_minus = P_nom * (np.exp( k_cov * se_z) - 1.0)   # Upper curve error boundary delta

                # (Channel 1) aggregate over the ASYMPTOTIC region only (largest tx)
                order  = np.argsort(tx_v)
                n_tail = max(3, int(np.ceil(tail_frac * Nv)))
                tail   = order[-n_tail:]
                
                # FIXED: Added direction-aware mapping logic
                if self.is_increasing:
                    # y = C - P => C = y + P. Upper fluctuation (d_minus) pushes C upwards.
                    sigma_C_plus  = float(np.quantile(d_minus[tail], q))
                    sigma_C_minus = float(np.quantile(d_plus[tail],  q))
                else:
                    # y = C + P => C = y - P. Upper fluctuation (d_minus) pulls C downwards.
                    sigma_C_plus  = float(np.quantile(d_plus[tail],  q))
                    sigma_C_minus = float(np.quantile(d_minus[tail], q))

                # conservative side
                sigma_C = max(sigma_C_plus, sigma_C_minus)

                # floor at a fraction of the remaining extrapolation correction
                i_conv    = int(np.argmax(tx))                # most-converged data point
                remaining = abs(C - self.raw_y[i_conv])
                sigma_C   = max(sigma_C, floor_frac * remaining)

                res.update(dict(
                    sigma_C_plus=sigma_C_plus, sigma_C_minus=sigma_C_minus,
                    sigma_C=sigma_C, sigma_mc=sigma_C,
                    sigma_B=float(np.sqrt(var_B)),
                    sigma_ln_A=float(np.sqrt(var_ln_A)),
                    sigma_log=float(np.sqrt(sigma2_res)),
                    # FIXED: Added np.sqrt around var_B to maintain mathematical validity
                    corr_CB=float(cov_lnA_B / (np.sqrt(var_ln_A) * np.sqrt(var_B)) if var_ln_A > 0 and var_B > 0 else 0.0),
                ))
            return self.results

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
                        label=f'Exact ({truth_val:.8f})')
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
                        label=f'Exact ({truth_val:.8f})')
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

                # ── residual band (95% CI and 95% PI) ────────────────────────
                if len(ln_diff_v) > 3:
                    from scipy import stats
                    
                    N_pts = len(tx_v)
                    df = N_pts - 2
                    
                    # Critical t-value for 95% interval (two-tailed)
                    t_crit = stats.t.ppf(0.975, df)
                    
                    # Calculate residual variance (sigma^2)
                    resid_log = ln_diff_v - (ln_A + slope * tx_v)
                    sigma2_res = np.sum(resid_log ** 2) / df
                    
                    # Calculate mean and sum of squared differences for x
                    mean_tx = np.mean(tx_v)
                    ss_tx = np.sum((tx_v - mean_tx) ** 2)
                    
                    # Prevent division by zero if all x values are identical
                    if ss_tx == 0:
                        ss_tx = 1e-15
                        
                    # 1. Confidence Interval Variance (Uncertainty of the Mean Line)
                    var_line = sigma2_res * (1.0 / N_pts + (tx_line - mean_tx) ** 2 / ss_tx)
                    se_line = np.sqrt(var_line)
                    
                    # 2. Prediction Interval Variance (Uncertainty of Future Points)
                    # Adds +1.0 to account for the inherent variance of individual points
                    var_pred = sigma2_res * (1.0 + 1.0 / N_pts + (tx_line - mean_tx) ** 2 / ss_tx)
                    se_pred = np.sqrt(var_pred)
                    
                    # Calculate upper and lower bounds
                    ci_upper = ln_line + t_crit * se_line
                    ci_lower = ln_line - t_crit * se_line
                    
                    pi_upper = ln_line + t_crit * se_pred
                    pi_lower = ln_line - t_crit * se_pred

                    # Plot PI Band first (Wider, more transparent)
                    ax.fill_between(tx_line,
                                    pi_lower,
                                    pi_upper,
                                    color=color, alpha=0.08,
                                    label='95% PI Band')

                    # Plot CI Band on top (Narrower, slightly darker)
                    ax.fill_between(tx_line,
                                    ci_lower,
                                    ci_upper,
                                    color=color, alpha=0.2,
                                    label='95% CI Band')

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
