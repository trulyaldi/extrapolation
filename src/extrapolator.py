from scipy.optimize import least_squares, lsq_linear
from sklearn.linear_model import HuberRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

class VarProIRLS:
    def __init__(self, df, x_col, y_col, err_df=None, b_init=None):
        
        self.raw_x = df[x_col].values.astype(float)

        # --------- MINMAX SCALE y (ONLY CHANGE) ----------
        y = df[y_col].values.astype(float)
        self.y_col = y_col
        self.x_col = x_col
        self.err_df = err_df

        self.y_min = float(np.min(y))
        self.y_max = float(np.max(y))
        self.y_range = self.y_max - self.y_min
        if self.y_range == 0:
            self.y_range = 1.0  # avoid division by zero for constant y

        self.raw_y = (y - self.y_min) / self.y_range
        # -------------------------------------------------

        # We need to identify the min and max values for x to perform scaling on them. 
        self.x_min = self.raw_x.min()
        self.x_max = self.raw_x.max()
        self.range_x = self.x_max - self.x_min
        # ---------------------------------------

        # --- TREND DETECTION --- TO DETERMINE IF THE TREND IS INCREASING / DECREASING
        huber = HuberRegressor()
        huber.fit(self.raw_x.reshape(-1, 1), self.raw_y)
        self.is_increasing = huber.coef_[0] > 0
        # --------------------------------

        # User can provide the initial guess for B, if not provided, default value is 1.0. 
        self.override_b_init = b_init
        self.b_init = None
        self.results = {}
        # --------------------------------

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

        # Logarithmic grid: dense at low B, sparse at high B
        coarse_grid = np.geomspace(1, 100, 100)
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
        # Calculate raw log components
        if self.model_type == 'exponential':
            log_w = B * t
        elif self.model_type == 'sqrt_exponential': 
            log_w = B * np.sqrt(t)
        elif self.model_type == 'power_law': 
            log_w = B * np.log(t)
        
        # Inverse variance approximation (square the weights)
        log_w = 2.0 * log_w
        
        log_center = np.median(log_w)
        log_w_shifted = log_w - log_center

        log_w_safe = np.clip(log_w_shifted, -50, 50)
        weights = np.exp(log_w_safe)
        
        return weights

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

        res_opt = least_squares(residual_func, x0=[start_b], bounds=(1, 100), 
                                method='trf', loss='linear')
        
        best_B = res_opt.x[0]
        phi = self._compute_basis(best_B, self.t_scaled)
        phi_w = phi * np.sqrt(weights)[:, np.newaxis]
        final_lin = lsq_linear(phi_w, y_w, bounds=(lb, ub), method='bvls')
        
        return best_B, final_lin.x[0], final_lin.x[1]

    def fit_irls(self, max_iter=110, tol=1e-9, damping=0.5, models=None, verbose=False, compute_uq=False):
        if models is None:
            models = ['exponential', 'sqrt_exponential', 'power_law']

        # stopping hyperparams (kept internal so you don't change the signature)
        eps = 1e-12
        stall = 0
        stall_patience = 3
        tol_w = 10.0 * tol  

        for model in models:
            self._setup_model(model)
            if verbose:
                print(f"\n--- Fitting Model: {self.model_type} ---")
                print(f"{'Iter':<5} | {'B (Decay)':<15} | {'C (Asymptote)':<15} | {'A (Scale)':<15} | {'Weight Ratio':<25} | {'rel_obj':<12} | {'rel_w':<12}")
                print("-" * 120)

            current_weights = np.ones(len(self.raw_x), dtype=float)

            current_B_guess = self.b_init
            final_B, final_C, final_A = 0.0, 0.0, 0.0

            prev_obj = np.inf

            for k in range(max_iter):
                # ---- VarPro solve under current weights ----
                
                B, C, A = self._solve_varpro_step(current_weights, start_b=current_B_guess)
                current_B_guess = B
                final_B, final_C, final_A = B, C, A

                # ---- objective under current weights (what you're minimizing this iter) ----
                phi = self._compute_basis(B, self.t_scaled)
                y_pred = C + A * phi[:, 1]
                resid = self.raw_y - y_pred
                obj = float(np.sum(current_weights * resid**2))  # weighted SSR

                rel_obj = abs(obj - prev_obj) / (abs(prev_obj) + eps)

                new_weights = self._compute_model_weights(B, self.t_scaled)

                # Robust scale of residuals (MAD — unchanged)
                med_resid = np.median(resid)
                mad_resid = np.median(np.abs(resid - med_resid))
                sigma_resid = 1.4826 * mad_resid

                if sigma_resid > 0:
                    z = np.abs(resid) / sigma_resid
                    reliability = np.where(z <= 2.0, 1.0, (2.0 / z) ** 2)
                else:
                    reliability = np.ones_like(resid)

                effective_new = new_weights * reliability

                # Normalize both to sum to 1 before blending
                current_normed = current_weights / np.sum(current_weights)
                new_normed = effective_new / np.sum(effective_new)

                log_cur = np.log(current_normed + eps)
                log_new = np.log(new_normed + eps)

                log_prop = (1 - damping) * log_cur + damping * log_new
                proposed_normed = np.exp(log_prop)
                proposed_normed /= proposed_normed.sum()

                # Restore to original scale of current weights
                proposed_weights = proposed_normed * np.sum(current_weights)


                rel_w = np.sum(np.abs(proposed_weights - current_weights)) / (np.sum(np.abs(current_weights)) + eps)

                # ---- stopping: objective + weights both stagnate ----
                if k > 0:
                    if (rel_obj < tol) and (rel_w < tol_w):
                        stall += 1
                        if stall >= stall_patience:
                            if verbose:
                                print("-" * 120)
                                print(f"Converged at iteration {k} (rel_obj={rel_obj:.3e}, rel_w={rel_w:.3e})")
                            break
                    else:
                        stall = 0

                prev_obj = obj
                current_weights = proposed_weights

                if verbose:
                    w_ratio = float(np.max(current_weights) / (np.min(current_weights) + eps))
                    print(f"{k:<5} | {B:<15.10f} | {C:<15.10f} | {A:<15.10f} | {w_ratio:<25.3f} | {rel_obj:<12.3e} | {rel_w:<12.3e}")

            phi = self._compute_basis(final_B, self.t_scaled)
            y_pred = final_C + final_A * phi[:, 1]
            ssr = np.sum((self.raw_y - y_pred) ** 2)

            dof = max(1, len(self.raw_y) - 3)
            sigma_noise = np.sqrt(ssr / dof)

            self.results[model] = {
                'B': final_B, 'C': final_C, 'A': final_A,
                'ssr': ssr, 't_scaled': self.t_scaled.copy(),
                'sigma_noise': sigma_noise,
                'y_pred': y_pred,
                'final_weights': current_weights
            }

        if compute_uq and len(self.results) > 0:
            self.compute_uncertainty()

        return self.results


    def compute_uncertainty(self):
        pass

    def plot(self, truth_val=None):
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
        zoom_end = self.x_max * 1.5

        y_min_zoom, y_max_zoom = np.inf, -np.inf
        mask = self.raw_x >= zoom_start
        if np.any(mask):
            y_min_zoom = min(y_min_zoom, np.min(y_data[mask]))
            y_max_zoom = max(y_max_zoom, np.max(y_data[mask]))

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

            y_min_zoom = min(y_min_zoom, C - 2 * sigma)
            y_max_zoom = max(y_max_zoom, C + 2 * sigma)

            mask_plot = (x_plot >= zoom_start) & (x_plot <= zoom_end)
            if np.any(mask_plot):
                y_min_zoom = min(y_min_zoom, np.min(y_plot[mask_plot]))
                y_max_zoom = max(y_max_zoom, np.max(y_plot[mask_plot]))

        if truth_val is not None:
            plt.axhline(truth_val, color='r', linestyle=':', linewidth=2, label=f'Truth ({truth_val:.8f})')
            if self.err_df is not None and self.y_col in self.err_df.columns:
                try:
                    truth_err = self.err_df[self.y_col].values[-1]
                    plt.fill_between([self.x_min, self.x_max * 1.5],
                                    truth_val - truth_err, truth_val + truth_err,
                                    color='r', alpha=0.15, zorder=0,
                                    label='Reference Error')
                except IndexError:
                    pass

        plt.xlim(zoom_start, zoom_end)

        if not np.isinf(y_min_zoom) and not np.isinf(y_max_zoom):
            y_span = y_max_zoom - y_min_zoom
            if y_span == 0:
                y_span = 0.1
            plt.ylim(y_min_zoom - 0.1 * y_span, y_max_zoom + 0.1 * y_span)

        plt.title(f"{self.y_col} - Zoomed Tail & Extrapolation")
        plt.xlabel("Basis Size")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    def plot_final_weights(self, model_type='exponential', sort_by_x=True, normalize=False):
        """
        Bar chart of final IRLS weights for each data point.

        Parameters
        ----------
        model_type : str
            One of: 'exponential', 'sqrt_exponential', 'power_law'
        sort_by_x : bool
            If True, bars follow increasing x. If False, uses original row order.
        normalize : bool
            If True, scales weights to sum to 1 for easier comparison.
        """
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
        plt.bar(np.arange(len(w_plot)), w_plot)
        plt.title(f"Final IRLS Weights per Data Point ({model_type})")
        plt.xlabel("Data point index" + (" (sorted by x)" if sort_by_x else " (original order)"))
        plt.ylabel("Weight" + (" (normalized)" if normalize else ""))
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
