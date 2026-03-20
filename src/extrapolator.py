from scipy.optimize import least_squares, lsq_linear
from sklearn.linear_model import HuberRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

class VarProIRLS:
    def __init__(self, df, x_col, y_col, err_df=None, inf_df=None, b_init=None, n_fit=None):

        x_all = df[x_col].values.astype(float)
        y_all = df[y_col].values.astype(float)

        self.y_col  = y_col
        self.x_col  = x_col
        self.err_df = err_df

        if inf_df is not None and y_col in inf_df.columns:
            self.truth_val = float(inf_df[y_col].iloc[-1])
        else:
            self.truth_val = None  

        # ── y scaling uses ALL rows so the scale is consistent ───────────────
        self.y_min   = float(np.min(y_all))
        self.y_max   = float(np.max(y_all))
        self.y_range = self.y_max - self.y_min
        if self.y_range == 0:
            self.y_range = 1.0

        # ── x scaling uses ALL rows for the same reason ──────────────────────
        self.x_min   = float(x_all.min())
        self.x_max   = float(x_all.max())
        self.range_x = self.x_max - self.x_min

        # ── select fitting rows ───────────────────────────────────────────────
        if n_fit is not None and n_fit < len(df):
            x_fit = x_all[:n_fit]
            y_fit = y_all[:n_fit]
        else:
            x_fit = x_all
            y_fit = y_all

        # raw_x / raw_y are what the solver actually sees
        self.raw_x = x_fit
        self.raw_y = (y_fit - self.y_min) / self.y_range

        # ── trend detection on fitting rows ───────────────────────────────────
        huber = HuberRegressor()
        x_norm = (self.raw_x - self.raw_x.min()) / (self.raw_x.max() - self.raw_x.min())
        huber.fit(x_norm.reshape(-1, 1), self.raw_y)
        self.is_increasing = huber.coef_[0] > 0

        # User can provide the initial guess for B, if not provided, default value is 1.0.
        self.override_b_init = b_init
        self.b_init = None
        self.results = {}
        # ─────────────────────────────────────────────────────────────────────

    def _setup_model(self, model_type):

        self.model_type = model_type
        if self.model_type in ['power_law', 'sqrt_exponential', 'exponential']:
            self.t_scaled = self.raw_x / self.x_max

        # If the user provides the initial guess for B, the algorithm starts off with this value.    
        if self.override_b_init is not None:
            self.b_init = self.override_b_init
        # If the user does not provide any initial guess for B, the algorithm starts grid search scanning. 
        else:
            self.b_init = self._grid_search_initialization()

    def _get_A_bounds(self):
        """Only constrain A based on curve direction, C is unconstrained."""
        if self.is_increasing:
            lb = [-np.inf, -np.inf] # [C,A]
            ub = [np.inf, 0]        # [C,A]
        else: 
            lb = [-np.inf, 0]       # [C,A]
            ub = [np.inf, np.inf]   # [C,A]
        return lb, ub
    
    def _get_b_bounds(self):
        """
        Compute meaningful B bounds based on model type and data range.
        The key constraint: the basis function must not be numerically zero
        at the smallest t value in the dataset.
        """
        t_min = self.t_scaled.min()
        t_max = self.t_scaled.max()  # always 1.0 by construction
        
        # We want exp(-B * t_min) > epsilon, i.e. B < -log(eps)/t_min
        # At B_max, the basis function at t_min should still be ~1e-6 (not zero)
        eps_basis = 1e-3  # minimum meaningful basis function value
        
        if self.model_type == 'exponential':
            # exp(-B * t_min) > eps  =>  B < -log(eps) / t_min
            b_max = -np.log(eps_basis) / t_min  # ~13.8 / t_min
        elif self.model_type == 'sqrt_exponential':
            # exp(-B * sqrt(t_min)) > eps  =>  B < -log(eps) / sqrt(t_min)
            b_max = -np.log(eps_basis) / np.sqrt(t_min)
        elif self.model_type == 'power_law':
            # t_min^(-B) < 1/eps  =>  B < log(1/eps) / log(1/t_min)
            b_max = np.log(1.0 / eps_basis) / np.log(1.0 / t_min + 1e-12)
        
        # Also cap at a hard maximum to avoid insane extrapolation
        b_max = min(b_max, 20.0)
        b_max = max(b_max, 2.0)  # ensure b_max > b_min=1
        
        return 1.0, b_max

    def _grid_search_initialization(self):
        lb, ub = self._get_A_bounds()
        
        def evaluate_grid(grid_values, current_best_ssr, current_best_b):
            local_best_ssr = current_best_ssr
            local_best_b = current_best_b
            
            for b_val in grid_values:
                phi = self._compute_basis(b_val, self.t_scaled)
                try:
                    res = lsq_linear(phi, self.raw_y, bounds=(lb, ub), method='bvls')
                    y_pred = np.dot(phi, res.x)
                    ssr = np.sum((self.raw_y - y_pred)**2)
                except:
                    ssr = np.inf
                
                if ssr < local_best_ssr:
                    local_best_ssr = ssr
                    local_best_b = b_val
            return local_best_ssr, local_best_b

        b_lo, b_hi = self._get_b_bounds()
        # Logarithmic grid: dense at low B, sparse at high B
        coarse_grid = np.geomspace(b_lo, b_hi, 100) 
        best_ssr, best_B = evaluate_grid(coarse_grid, np.inf, 1.0)
        
        # Fine grid: also logarithmic around the best B
        idx = np.searchsorted(coarse_grid, best_B)
        fine_min = coarse_grid[max(0, idx - 1)]
        fine_max = coarse_grid[min(len(coarse_grid) - 1, idx + 1)]
        
        fine_grid = np.geomspace(fine_min, fine_max, 100)
        best_ssr, best_B = evaluate_grid(fine_grid, best_ssr, best_B)
        
        return best_B

    def _compute_basis(self, B, t):
        phi = np.zeros((len(t), 2))
        phi[:, 0] = 1.0
        if self.model_type == 'exponential':
            phi[:, 1] = np.exp(-B * t)
        elif self.model_type == 'sqrt_exponential':
            phi[:, 1] = np.exp(-B * np.sqrt(t))
        elif self.model_type == 'power_law': 
            phi[:, 1] = np.power(t, -B)
        return phi

    def _compute_model_weights(self, B, t):

        n = len(t)

        if self.model_type == 'exponential':
            log_w = B * t
        elif self.model_type == 'sqrt_exponential':
            log_w = B * np.sqrt(t)
        elif self.model_type == 'power_law':
            log_w = B * np.log(np.maximum(t, 1e-12))

        # Center (ratio-preserving normalisation)
        log_w -= np.median(log_w)
        

        # n-adaptive cap: ratio ≤ n²  (for n=4: ratio ≤ 16; n=100: ratio ≤ 10000)
        # log of max ratio divided by 2 gives the symmetric half-width of the clip
        max_log_half = 0.5 * np.log(max(float(n * n), 4.0))
        log_w = np.clip(log_w, -max_log_half, max_log_half)

        return np.exp(log_w)

    def _solve_varpro_step(self, weights, start_b=None):
        y_w = self.raw_y * np.sqrt(weights)
        if start_b is None:  
            start_b = self.b_init
        
        lb, ub = self._get_A_bounds()

        def residual_func(alpha_curr):
            phi = self._compute_basis(alpha_curr[0], self.t_scaled)
            phi_w = phi * np.sqrt(weights)[:, np.newaxis]
            res_lin = lsq_linear(phi_w, y_w, bounds=(lb, ub), method='bvls')
            y_model = np.dot(phi, res_lin.x)
            return (self.raw_y - y_model) * np.sqrt(weights)
        

        b_lo, b_hi = self._get_b_bounds()
        res_opt = least_squares(residual_func, x0=[start_b], bounds=(b_lo, b_hi), 
                                method='trf', loss='linear')
        
        best_B = res_opt.x[0]
        phi = self._compute_basis(best_B, self.t_scaled)
        phi_w = phi * np.sqrt(weights)[:, np.newaxis]
        final_lin = lsq_linear(phi_w, y_w, bounds=(lb, ub), method='bvls')
        
        return best_B, final_lin.x[0], final_lin.x[1]

    # ── CHANGE 1: added `progress_callback=None` to the signature.
    # All existing calls (verbose=False, compute_uq=True, etc.) are unaffected
    # because it sits at the end with a default of None.
    def fit_irls(self, max_iter=110, tol=1e-9, damping=0.5, models=None,
                verbose=False, compute_uq=True, on_iteration=None):

        if models is None:
            models = ['exponential', 'sqrt_exponential', 'power_law']

        eps            = 1e-12
        tol_w          = 10.0 * tol
        stall_patience = 3

        for model in models:
            self._setup_model(model)

            if verbose:
                print(f"\n--- Fitting Model: {self.model_type} ---")
                header = (f"{'Iter':<5} | {'B (Decay)':<15} | {'C (Asymptote)':<15} | "
                        f"{'A (Scale)':<15} | {'Weight Ratio':<25} | "
                        f"{'rel_obj':<12} | {'rel_w':<12}")
                print(header)
                print("-" * 120)

            n_pts           = len(self.raw_x)
            current_weights = np.ones(n_pts, dtype=float)
            current_B_guess = self.b_init
            final_B, final_C, final_A = 0.0, 0.0, 0.0
            prev_obj        = np.inf
            stall           = 0
            C_history: list[float] = []

            # Weight floor derived from n² ratio cap (no tunable parameter)
            weight_floor = 1.0 / (n_pts ** 3)

            # ── FIX 1: initialise prev_B/C to None, not np.inf ───────────────
            # np.inf caused abs(B - inf)/(inf + eps) = inf/inf = nan on iter 0,
            # triggering RuntimeWarning at the call site in __init__ (misleading
            # traceback). None forces an explicit k > 0 guard.
            prev_B: float | None = None
            prev_C: float | None = None
            param_stall          = 0

            # ── Geometric tail extrapolation state ────────────────────────────
            # Tracks the last three C-deltas to detect a geometric series and
            # extrapolate to convergence without running out the full decay.
            # delta_history stores (C_{k} - C_{k-1}) for the last 3 iters.
            delta_history: list[float] = []

            for k in range(max_iter):
                B, C, A   = self._solve_varpro_step(current_weights,
                                                    start_b=current_B_guess)
                current_B_guess       = B
                final_B, final_C, final_A = B, C, A
                C_history.append(float(C))

                # Weighted objective
                phi    = self._compute_basis(B, self.t_scaled)
                y_pred = C + A * phi[:, 1]
                resid  = self.raw_y - y_pred
                obj    = float(np.sum(current_weights * resid ** 2))
                rel_obj = abs(obj - prev_obj) / (abs(prev_obj) + eps)

                # New model weights + MAD reliability
                new_weights  = self._compute_model_weights(B, self.t_scaled)
                sigma_resid  = 1.4826 * np.median(np.abs(resid - np.median(resid)))
                if sigma_resid > 0:
                    z           = np.abs(resid) / sigma_resid
                    reliability = np.where(z <= 2.0, 1.0, (2.0 / z) ** 2)
                else:
                    reliability = np.ones_like(resid)

                effective_new = new_weights * reliability

                # Log-space damped blend
                cur_n = current_weights / (current_weights.sum() + eps)
                new_n = effective_new   / (effective_new.sum()   + eps)
                log_p = ((1 - damping) * np.log(cur_n + eps)
                        + damping      * np.log(new_n + eps))
                proposed_normed = np.exp(log_p)
                proposed_normed /= proposed_normed.sum()

                # Apply weight floor (prevents geometric drift to zero)
                proposed_normed = np.maximum(proposed_normed, weight_floor)
                proposed_normed /= proposed_normed.sum()
                proposed = proposed_normed * current_weights.sum()

                rel_w   = (np.sum(np.abs(proposed - current_weights))
                        / (np.sum(np.abs(current_weights)) + eps))
                w_ratio = float(current_weights.max()
                                / (current_weights.min() + eps))

                # ── Stopping criterion 1: objective + weight stagnation ───────
                converged_ow = (k > 0 and rel_obj < tol and rel_w < tol_w)

                # ── Stopping criterion 2: parameter stability ─────────────────
                # FIX 1 applied: guard on k > 0 AND prev_B is not None
                if k > 0 and prev_B is not None and prev_C is not None:
                    rel_B = abs(B - prev_B) / (abs(prev_B) + eps)
                    rel_C = abs(C - prev_C) / (abs(prev_C) + eps)
                    param_stable = (rel_B < tol and rel_C < tol)
                else:
                    rel_B = rel_C = float('nan')
                    param_stable  = False

                if param_stable:
                    param_stall += 1
                else:
                    param_stall = 0

                converged_params = (param_stall >= stall_patience)

                # ── Stopping criterion 3: geometric tail extrapolation ────────
                #
                # When C changes form a geometric series, we can compute the
                # exact limit analytically instead of iterating to convergence.
                #
                # Detection: if the last 3 deltas satisfy d_{k}/d_{k-1} ≈ r
                # with |r| < 1 consistently, the series converges to:
                #   C* = C_k + d_k / (1 - r)
                #
                # We accept extrapolation when:
                #   (a) B is already frozen  (parameter stability reached for B)
                #   (b) The geometric ratio r is consistent over 3 steps: |r₁ - r₂| < 0.05
                #   (c) |r| < 0.7  (the series is actually contracting quickly)
                #   (d) The extrapolated correction is small: |d_k/(1-r)| < 1e-4
                #
                # Condition (d) ensures we only extrapolate when near convergence,
                # not during the transient phase where the geometric approximation
                # may not hold.
                converged_extrap = False
                if (k >= 3 and prev_C is not None
                        and param_stall >= 1):   # B must be frozen first

                    delta = C - prev_C
                    delta_history.append(float(delta))
                    if len(delta_history) > 3:
                        delta_history.pop(0)   # keep only the last 3

                    if len(delta_history) == 3:
                        d1, d2, d3 = delta_history

                        # Compute decay ratios (guard division by zero)
                        r1 = d2 / (d1 + eps * np.sign(d1 + eps))
                        r2 = d3 / (d2 + eps * np.sign(d2 + eps))

                        ratio_consistent = abs(r1 - r2) < 0.05
                        contracting      = abs(r2) < 0.70
                        correction       = abs(d3 / (1.0 - r2 + eps))

                        if ratio_consistent and contracting and correction < 1e-4:
                            # Extrapolate to series limit and override C
                            C_extrap      = C + d3 / (1.0 - r2)
                            final_C       = C_extrap
                            C_history[-1] = float(C_extrap)   # replace last entry
                            converged_extrap = True

                            if verbose:
                                print("-" * 120)
                                print(f"Geometric extrapolation at iter {k}: "
                                    f"C {C:.10f} → {C_extrap:.10f} "
                                    f"(r={r2:.4f}, correction={correction:.2e})")
                else:
                    if prev_C is not None:
                        delta_history.append(float(C - prev_C))
                        if len(delta_history) > 3:
                            delta_history.pop(0)

                # ── Convergence decision ──────────────────────────────────────
                any_converged = converged_ow or converged_params or converged_extrap

                if any_converged:
                    stall += 1
                    if stall >= stall_patience or converged_extrap:
                        if verbose:
                            if converged_extrap:
                                reason = "geometric extrapolation"
                            elif converged_ow:
                                reason = "obj+weight"
                            else:
                                reason = "parameter stability"
                            print("-" * 120)
                            print(f"Converged at iter {k} via {reason} "
                                f"(rel_obj={rel_obj:.2e}, "
                                f"rel_B={rel_B:.2e}, rel_C={rel_C:.2e})")

                        if on_iteration is not None:
                            info = dict(
                                model=model, iteration=k, converged=True,
                                B=B, C=final_C, A=A,
                                rel_obj=rel_obj, rel_w=rel_w, w_ratio=w_ratio,
                                weights=current_weights.copy(),
                                y_pred=y_pred.copy(),
                                raw_x=self.raw_x.copy(),
                                raw_y=self.raw_y.copy(),
                                t_scaled=self.t_scaled.copy(),
                            )
                            on_iteration(info)
                        break
                else:
                    stall = 0

                prev_obj = obj
                prev_B   = B
                prev_C   = C
                current_weights = proposed

                if verbose:
                    print(f"{k:<5} | {B:<15.10f} | {C:<15.10f} | "
                        f"{A:<15.10f} | {w_ratio:<25.3f} | "
                        f"{rel_obj:<12.3e} | {rel_w:<12.3e}")

                if on_iteration is not None:
                    info = dict(
                        model=model, iteration=k, converged=False,
                        B=B, C=C, A=A, rel_obj=rel_obj, rel_w=rel_w,
                        w_ratio=w_ratio, weights=current_weights.copy(),
                        y_pred=y_pred.copy(),
                        raw_x=self.raw_x.copy(),
                        raw_y=self.raw_y.copy(),
                        t_scaled=self.t_scaled.copy(),
                    )
                    if on_iteration(info) is False:
                        break

            # Final metrics using final_C (may be extrapolated)
            phi    = self._compute_basis(final_B, self.t_scaled)
            y_pred = final_C + final_A * phi[:, 1]
            ssr    = float(np.sum((self.raw_y - y_pred) ** 2))
            dof    = max(1, len(self.raw_y) - 3)

            self.results[model] = {
                'B':             final_B,
                'C':             final_C,
                'A':             final_A,
                'ssr':           ssr,
                't_scaled':      self.t_scaled.copy(),
                'sigma_noise':   float(np.sqrt(ssr / dof)),
                'y_pred':        y_pred.copy(),
                'final_weights': current_weights.copy(),
                '_C_history':    C_history,
                '_n_iters':      len(C_history),
            }

        if compute_uq and self.results:
            self.compute_uncertainty()

        return self.results

    def compute_uncertainty(self, n_bootstrap=80, confidence_level=80.0):
            """
            Computes uncertainty for the asymptote C using a Parametric Monte Carlo 
            approach with Heteroscedastic noise scaling derived from IRLS weights.
            """
            if not self.results:
                raise RuntimeError("No results found. Run fit_irls() first.")

            # Percentiles for the uncertainty band
            lower_p = (100.0 - confidence_level) / 2.0
            upper_p = 100.0 - lower_p

            for model, res in self.results.items():
                self.model_type = model
                
                # --- 1. SETUP PARAMETERS ---
                B_best = res['B']
                y_fit = res['y_pred']  # These are the predicted values at raw_x
                weights = res['final_weights']
                t_scaled = res['t_scaled']
                lb, ub = self._get_A_bounds()
                
                # --- 2. COMPUTE WEIGHTED GLOBAL NOISE (sigma_base) ---
                # Using the weighted residual standard error formula
                resid = self.raw_y - y_fit
                ssr_weighted = np.sum(weights * (resid**2))
                dof = max(1, len(self.raw_y) - 3)
                sigma_base = np.sqrt(ssr_weighted / dof)
                
                # --- 3. COMPUTE POINT-BY-POINT NOISE SCALING ---
                # sigma_i = sigma_base / sqrt(w_i)
                # We add a tiny epsilon to weights to avoid division by zero
                eps = 1e-12
                sigma_i = sigma_base / np.sqrt(weights + eps)

                boot_C_values = []

                # --- 4. MONTE CARLO LOOP ---
                for _ in range(n_bootstrap):
                    # Generate synthetic data with heteroscedastic noise
                    noise = np.random.normal(0, sigma_i)
                    y_synth = y_fit + noise
                    
                    # Define the VarPro residual function for this synthetic set
                    # We still weight the residuals during the fit to maintain IRLS logic
                    def boot_residual(alpha_curr):
                        b_val = alpha_curr[0]
                        phi = self._compute_basis(b_val, t_scaled)
                        # Weighting the synthetic solve identically to the original fit
                        phi_w = phi * np.sqrt(weights)[:, np.newaxis]
                        y_synth_w = y_synth * np.sqrt(weights)
                        
                        res_lin = lsq_linear(phi_w, y_synth_w, bounds=(lb, ub), method='bvls')
                        y_model = np.dot(phi, res_lin.x)
                        return (y_synth - y_model) * np.sqrt(weights)

                    try:
                        # Re-optimize B for every single synthetic realization
                        b_lo, b_hi = self._get_b_bounds()

                        res_opt = least_squares(boot_residual, x0=[B_best], bounds=(b_lo, b_hi), 
                                                method='trf', loss='linear')
                        
                        # Final linear solve to extract C for this bootstrap sample
                        best_B_boot = res_opt.x[0]
                        phi_boot = self._compute_basis(best_B_boot, t_scaled)
                        phi_w_boot = phi_boot * np.sqrt(weights)[:, np.newaxis]
                        y_synth_w = y_synth * np.sqrt(weights)
                        
                        final_lin = lsq_linear(phi_w_boot, y_synth_w, bounds=(lb, ub), method='bvls')
                        boot_C_values.append(final_lin.x[0])
                    except:
                        continue # Skip failed optimizations

                # --- 5. EXTRACT STATISTICS ---
                if boot_C_values:
                    boot_C_values = np.array(boot_C_values)
                    
                    # Percentiles in scaled space
                    C_low_scaled = np.percentile(boot_C_values, lower_p)
                    C_high_scaled = np.percentile(boot_C_values, upper_p)
                    
                    # Unscale to original units
                    C_best_unscaled = self.y_min + self.y_range * res['C']
                    C_low_unscaled = self.y_min + self.y_range * C_low_scaled
                    C_high_unscaled = self.y_min + self.y_range * C_high_scaled
                    
                    # Store results
                    res['sigma_C_lower_unscaled'] = abs(C_best_unscaled - C_low_unscaled)
                    res['sigma_C_upper_unscaled'] = abs(C_high_unscaled - C_best_unscaled)
                    
                    # Use the larger of the two for a "pessimistic" symmetric sigma_mc
                    max_err_scaled = max(abs(res['C'] - C_low_scaled), abs(res['C'] - C_high_scaled))
                    res['sigma_mc'] = max_err_scaled
                else:
                    res['sigma_mc'] = 0.0

            return self.results

    def plot(self, truth_val=None):
        # Use stored truth_val from inf_df if caller did not pass one explicitly
        if truth_val is None:
            truth_val = self.truth_val
        if not self.results:
            return

        # ---- unscale helper: y_scaled -> y_original ----
        def unscale(y_scaled):
            return self.y_min + self.y_range * np.asarray(y_scaled, dtype=float)

        colors = {'exponential': 'blue', 'sqrt_exponential': 'orange', 'power_law': 'green'}
        labels = {'exponential': 'Exp($e^{-Bx}$)', 'sqrt_exponential': 'SqrtExp($e^{-B\\sqrt{x}}$)', 'power_law': 'Power($x^{-B}$)'}

        # Unscale raw data for plotting
        y_data = unscale(self.raw_y)

        print("\n" + "="*80)
        print(f"{'Model':<20} | {'C (Asymptote)':<15} | {'MC Uncertainty':<20} | {'Diff from Reference'}")
        print("="*80)

        for model, res in self.results.items():
            C_scaled = res['C']
            sigma_scaled = res.get('sigma_mc', 0.0)

            C = float(unscale(C_scaled))
            sigma = float(self.y_range * sigma_scaled)

            diff_str = "-"
            if truth_val is not None:
                diff_str = f"{abs(C - truth_val):.2e}"

            print(f"{model:<20} | {C:<15.9f} | {sigma:<20.2e} | {diff_str}")

        # --- PLOT 1: FULL DATA ---
        plt.figure(figsize=(12, 7))
        plt.plot(self.raw_x, y_data, 'ko', label='Data', zorder=5, markersize=6)

        for model, res in self.results.items():
            C_scaled = res['C']
            sigma_scaled = res.get('sigma_mc', 0.0)

            C = float(unscale(C_scaled))
            sigma = float(self.y_range * sigma_scaled)

            x_plot = np.linspace(self.x_min, self.x_max * 1.5, 200)
            t_plot = x_plot / self.x_max

            self.model_type = model
            phi_plot = self._compute_basis(res['B'], t_plot)
            y_plot_scaled = C_scaled + res['A'] * phi_plot[:, 1]
            y_plot = unscale(y_plot_scaled)

            plt.plot(x_plot, y_plot, '-', color=colors[model], linewidth=2, alpha=0.8, label=f"{labels[model]}")
            plt.fill_between([self.x_min, self.x_max * 1.5],
                            C - sigma, C + sigma,
                            color=colors[model], alpha=0.1)
            plt.axhline(C, color=colors[model], linestyle='--', alpha=0.3)

        if truth_val is not None:
            plt.axhline(truth_val, color='r', linestyle=':', linewidth=2, label=f'Truth ({truth_val:.8f})')
            if self.err_df is not None and self.y_col in self.err_df.columns:
                try:
                    truth_err = self.err_df[self.y_col].values[-1]
                    plt.fill_between([self.x_min, self.x_max * 1.5],
                                    truth_val - truth_err, truth_val + truth_err,
                                    color='r', alpha=0.15, zorder=0,
                                    label=f'Reference Error (±{truth_err:.1e})')
                except IndexError:
                    pass

        plt.title(f"{self.y_col} - Full View")
        plt.xlabel("Basis Size")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # --- PLOT 2: ZOOMED TAIL & EXTRAPOLATION ---
        plt.figure(figsize=(12, 7))
        plt.plot(self.raw_x, y_data, 'ko', label='Data', zorder=5, markersize=6)

        zoom_start = self.x_min + 0.6 * self.range_x
        zoom_end   = self.x_max * 1.5

        # Seed y bounds with visible tail data points
        y_min_zoom, y_max_zoom = np.inf, -np.inf
        mask = self.raw_x >= zoom_start
        if np.any(mask):
            y_min_zoom = min(y_min_zoom, np.min(y_data[mask]))
            y_max_zoom = max(y_max_zoom, np.max(y_data[mask]))

        for model, res in self.results.items():
            C_scaled = res['C']
            sigma_scaled = res.get('sigma_mc', 0.0)

            C     = float(unscale(C_scaled))
            sigma = float(self.y_range * sigma_scaled)

            x_plot = np.linspace(self.x_min, self.x_max * 1.5, 200)
            t_plot = x_plot / self.x_max

            self.model_type = model
            phi_plot      = self._compute_basis(res['B'], t_plot)
            y_plot_scaled = C_scaled + res['A'] * phi_plot[:, 1]
            y_plot        = unscale(y_plot_scaled)

            plt.plot(x_plot, y_plot, '-', color=colors[model], linewidth=2,
                     alpha=0.8, label=f"{labels[model]}")
            plt.fill_between([self.x_min, self.x_max * 1.5],
                             C - sigma, C + sigma,
                             color=colors[model], alpha=0.1)
            plt.axhline(C, color=colors[model], linestyle='--', alpha=0.3)

            # Always include every asymptote C ± σ in the y bounds —
            # this is the key fix so fitted infinities are never clipped.
            y_min_zoom = min(y_min_zoom, C - sigma)
            y_max_zoom = max(y_max_zoom, C + sigma)

            # Also include the visible part of each fit curve
            mask_plot = (x_plot >= zoom_start) & (x_plot <= zoom_end)
            if np.any(mask_plot):
                y_min_zoom = min(y_min_zoom, np.min(y_plot[mask_plot]))
                y_max_zoom = max(y_max_zoom, np.max(y_plot[mask_plot]))

        if truth_val is not None:
            plt.axhline(truth_val, color='r', linestyle=':', linewidth=2,
                        label=f'Truth ({truth_val:.8f})')
            # Always include truth_val (and its error band) in y bounds
            y_min_zoom = min(y_min_zoom, truth_val)
            y_max_zoom = max(y_max_zoom, truth_val)
            if self.err_df is not None and self.y_col in self.err_df.columns:
                try:
                    truth_err = self.err_df[self.y_col].values[-1]
                    plt.fill_between([self.x_min, self.x_max * 1.5],
                                     truth_val - truth_err, truth_val + truth_err,
                                     color='r', alpha=0.15, zorder=0,
                                     label=f'Reference Error (±{truth_err:.1e})')
                    y_min_zoom = min(y_min_zoom, truth_val - truth_err)
                    y_max_zoom = max(y_max_zoom, truth_val + truth_err)
                except IndexError:
                    pass

        plt.xlim(zoom_start, zoom_end)

        if not np.isinf(y_min_zoom) and not np.isinf(y_max_zoom):
            y_span = y_max_zoom - y_min_zoom
            if y_span == 0:
                y_span = abs(y_min_zoom) * 0.01 + 1e-10
            plt.ylim(y_min_zoom - 0.1 * y_span, y_max_zoom + 0.1 * y_span)

        plt.title(f"{self.y_col} - Zoomed Tail & Extrapolation")
        plt.xlabel("Basis Size")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    def plot_final_weights(self, model_type='exponential', sort_by_x=True, normalize=False):

        if not self.results:
            raise RuntimeError("No results found. Run fit_irls() first.")

        if model_type not in self.results:
            raise ValueError(f"Model '{model_type}' not found in results. Available: {list(self.results.keys())}")

        w = np.asarray(self.results[model_type].get('final_weights', None), dtype=float)
        if w is None or w.size == 0:
            raise RuntimeError(f"No 'final_weights' stored for model '{model_type}'.")

        x = np.asarray(self.raw_x, dtype=float)

        if normalize:
            s = w.sum()
            if s > 0:
                w = w / s

        if sort_by_x:
            idx = np.argsort(x)
            x_plot = x[idx]
            w_plot = w[idx]
        else:
            x_plot = x
            w_plot = w

        plt.figure(figsize=(12, 5))
        plt.bar(np.arange(len(x_plot)), w_plot)
        plt.title(f"Final IRLS Weights per Data Point ({model_type})")
        plt.xlabel("Data point index" + (" (sorted by x)" if sort_by_x else " (original order)"))
        plt.ylabel("Weight" + (" (normalized)" if normalize else ""))
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()