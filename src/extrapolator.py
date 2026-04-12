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

        eps_basis = 1e-3  
        
        if self.model_type == 'exponential':

            b_max = -np.log(eps_basis) / t_min  # ~13.8 / t_min
        elif self.model_type == 'sqrt_exponential':

            b_max = -np.log(eps_basis) / np.sqrt(t_min)
        elif self.model_type == 'power_law':

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
        phi = np.ones((len(t), 2))
        if self.model_type == 'exponential':
            phi[:, 1] = np.exp(-B * t)
        elif self.model_type == 'sqrt_exponential':
            phi[:, 1] = np.exp(-B * np.sqrt(t))
        elif self.model_type == 'power_law': 
            phi[:, 1] = np.power(t, -B)
        return phi

    def _compute_leverage_weights(self, B, current_weights):
        phi   = self._compute_basis(B, self.t_scaled)
        phi_w = phi * np.sqrt(current_weights)[:, np.newaxis]

        try:
            PhiTWPhi     = phi_w.T @ phi_w
            PhiTWPhi_inv = np.linalg.inv(PhiTWPhi)
            h            = PhiTWPhi_inv[0, :] @ phi_w.T   # shape (n,)
            
            h_sq         = h ** 2
        except np.linalg.LinAlgError:
            h_sq = np.ones(len(self.raw_y))
        
        
        h_sq = np.maximum(h_sq, 1e-30)
        return h_sq / h_sq.sum() * len(self.raw_y)

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


    def fit_irls(self, max_iter=200, tol=1e-9, damping=0.5, models=None,
                verbose=False, compute_uq=False):

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

            # Weight floor derived from n² ratio cap (tunable parameter)
            weight_floor = 1.0 / (n_pts ** 5)

            prev_B = None
            prev_C = None

            for k in range(max_iter):
                B, C, A = self._solve_varpro_step(current_weights,
                                                start_b=current_B_guess)
                current_B_guess        = B
                final_B, final_C, final_A = B, C, A

                phi    = self._compute_basis(B, self.t_scaled)
                y_pred = C + A * phi[:, 1]
                resid  = self.raw_y - y_pred
                obj    = float(np.sum(current_weights * resid ** 2))
                rel_obj = abs(obj - prev_obj) / (abs(prev_obj) + eps)

                # ── leverage weights — replace _compute_model_weights ────────
                new_weights = self._compute_leverage_weights(B, current_weights)

                # MAD reliability on top (handles outlier points)
                sigma_resid = 1.4826 * np.median(np.abs(resid - np.median(resid)))
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
                proposed = proposed_normed * len(self.raw_y)

                rel_w   = (np.sum(np.abs(proposed - current_weights))
                        / (np.sum(np.abs(current_weights)) + eps))
                w_ratio = float(proposed.max() / (proposed.min() + eps))

                # ── Stopping criterion 1: objective + weight stagnation ───────
                converged_ow = (k > 0 and rel_obj < tol and rel_w < tol_w)

                # ── Stopping criterion 2: parameter stability ─────────────────
                if k > 0 and prev_B is not None and prev_C is not None:
                    rel_B = abs(B - prev_B) / (abs(prev_B) + eps)
                    rel_C = abs(C - prev_C) / (abs(prev_C) + eps)
                    param_stable = (rel_B < tol and rel_C < tol)
                else:
                    rel_B = rel_C = float('nan')
                    param_stable  = False

                # ── Convergence decision ──────────────────────────────────────
                if converged_ow or param_stable:
                    stall += 1
                    if stall >= stall_patience:
                        if verbose:
                            reason = "obj+weight" if converged_ow else "parameter stability"
                            print("-" * 120)
                            print(f"Converged at iter {k} via {reason} "
                                f"(rel_obj={rel_obj:.2e}, rel_B={rel_B:.2e}, rel_C={rel_C:.2e})")
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

            # Final metrics using final_C
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
                '_n_iters':      k + 1,
            }

        if compute_uq and self.results:
            self.compute_uncertainty()

        return self.results

    def compute_uncertainty(self):
        eps = np.finfo(float).eps

        for model, res in self.results.items():
            self.model_type = model

            B = res['B']
            A = res['A']
            C = res['C']
            w = res['final_weights']
            y_pred = res['y_pred']
            r = self.raw_y - y_pred
            y = self.raw_y
            x = self.raw_x

            # Step 1: robust noise scale
            w_sqrt = np.sqrt(w)
            sigma_noise = np.median(np.abs(r) / w_sqrt) / np.median(w_sqrt)

            # Step 2: extrapolation gap
            phi = self._compute_basis(B, self.t_scaled)
            phi_vals = np.abs(phi[:, 1])
            delta_inf = abs(A) * phi_vals[-1]

            # Step 3a: extrapolation leverage ratio
            lam = phi_vals[0] / (phi_vals[-1] + eps)

            # Step 3b: effective sample size
            n_eff = (np.sum(w_sqrt) ** 2) / (np.sum(w) + eps)

            sigma_fit = sigma_noise * lam / np.sqrt(n_eff)

            # Step 3c: gap uncertainty
            mad_y = np.median(np.abs(y - np.median(y))) + eps
            sigma_gap = delta_inf * (sigma_noise / mad_y)

            sigma_fit_total = np.sqrt(sigma_fit**2 + sigma_gap**2)

            # Step 4: reference floor
            sigma_ref = np.max(np.abs(np.diff(y)))

            sigma_C = max(sigma_fit_total, sigma_ref)

            res['sigma_mc'] = sigma_C

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
        print(f"{'Model':<20} | {'C (Asymptote)':<20} | {'Uncertainty':<20} | {'Diff from Reference'}")
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