import matplotlib.patches as patches
from lmfit import Model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- 1. Define the Mathematical Model Functions ---

def exponential_decay(x, C, A, B):
    """3-parameter exponential decay: y = C + A * exp(-B * x)."""
    x = np.asarray(x, float)
    arg = -B * x
    arg = np.clip(arg, -700, 700)  # avoid overflow in exp
    return C + A * np.exp(arg)

def exponential_decay_sq(x, C, A, B):
    """3-parameter exponential decay with sqrt(x): y = C + A * exp(-B * sqrt(x))."""
    x = np.asarray(x, float)
    sqrt_x = np.sqrt(np.clip(x, 0.0, None))
    arg = -B * sqrt_x
    arg = np.clip(arg, -700, 700)  # avoid overflow
    return C + A * np.exp(arg)

def power_law(x, C, A, B):
    """3-parameter power law: y = C + A * x^(-B)."""
    x = np.asarray(x, float)
    eps = 1e-12
    x_safe = np.clip(x, eps, None)  # avoid x=0 and negative x
    return C + A * np.power(x_safe, -B)


# --- 2. Shared B initialisation: grid search over B for all models ---

def _init_B(x_scaled, y, C=None, model_type='exp'):
    """
    Grid-search initializer for B for all three models:

        model_type = 'power'  : y ≈ C + A * x_scaled^(-B)
        model_type = 'exp'    : y ≈ C + A * exp(-B * x_scaled)
        model_type = 'exp_sq' : y ≈ C + A * exp(-B * sqrt(x_scaled))

    x_scaled should be the scaled x used in the fit.
    We scan B in (0.1, 0.2, ..., 10.0), fix B, and fit only C and A.
    """
    x_scaled = np.asarray(x_scaled, float)
    y = np.asarray(y, float)

    if C is None:
        C = y[-1]

    if model_type == 'power':
        model = Model(power_law)
    elif model_type == 'exp':
        model = Model(exponential_decay)
    elif model_type == 'exp_sq':
        model = Model(exponential_decay_sq)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'power', 'exp', or 'exp_sq'.")

    A0 = y[0] - C
    C0 = C

    B_candidates = np.arange(0.1, 10.0, 0.1)

    best_B = None
    best_chi2 = np.inf

    for B0 in B_candidates:
        params = model.make_params(C=C0, A=A0, B=B0)
        params['B'].vary = False

        try:
            tmp_result = model.fit(y, params, x=x_scaled)
        except Exception:
            continue

        chi2 = np.sum(tmp_result.residual**2)
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_B = B0

    if best_B is None:
        best_B = 1.0

    return float(best_B)


# --- 3. Base Uncertainty Calculator (Shared Logic) ---

class UncertaintyCalculator:

    def _calculate_uncertainty_at_x(self, params, x_value, model_type='exp'):

        C = params['C'].value
        A = params['A'].value
        B = params['B'].value
        
        dC = params['C'].stderr if params['C'].stderr is not None else 0
        dA = params['A'].stderr if params['A'].stderr is not None else 0
        dB = params['B'].stderr if params['B'].stderr is not None else 0
        
        try:
            if model_type == 'power':
                return self._calculate_power_uncertainty(C, A, B, dC, dA, dB, x_value)
            else:
                return self._calculate_exponential_uncertainty(C, A, B, dC, dA, dB, x_value, model_type)
        except (OverflowError, ValueError, ZeroDivisionError) as e:
            print(f"Warning: Uncertainty calculation failed at x={x_value}: {e}")
            return C, C
    
    def _calculate_exponential_uncertainty(self, C, A, B, dC, dA, dB, x_value, model_type='exp'):
        x_term = np.sqrt(x_value) if model_type == 'exp_sq' else x_value
        
        if A >= 0:
            term_min = (A - dA) * np.exp(-(B + dB) * x_term)
            term_max = (A + dA) * np.exp(-(B - dB) * x_term)
        else:
            term_min = (A - dA) * np.exp(-(B - dB) * x_term)
            term_max = (A + dA) * np.exp(-(B + dB) * x_term)
        
        f_min = (C - dC) + term_min
        f_max = (C + dC) + term_max
        
        if f_min > f_max:
            f_min, f_max = f_max, f_min
        
        return f_min, f_max
    
    def _calculate_power_uncertainty(self, C, A, B, dC, dA, dB, x_value):
        x_value = max(x_value, 1e-12)  # avoid x=0
        if A >= 0:
            term_min = (A - dA) / np.power(x_value, B + dB)
            term_max = (A + dA) / np.power(x_value, B - dB)
        else:
            term_min = (A - dA) / np.power(x_value, B - dB)
            term_max = (A + dA) / np.power(x_value, B + dB)
        
        f_min = (C - dC) + term_min
        f_max = (C + dC) + term_max
        
        if f_min > f_max:
            f_min, f_max = f_max, f_min
        
        return f_min, f_max
    
    def _calculate_extrapolation_uncertainty(self, result):
        params = result.params
        C = params['C'].value
        dC = params['C'].stderr if params['C'].stderr is not None else 0
        
        if dC == 0:
            try:
                from lmfit import conf_interval
                ci = conf_interval(result.minimizer, result, sigmas=[-1, 1])
                lower_sigma, lower_val = ci['C'][0]
                upper_sigma, upper_val = ci['C'][1]
                dC_lower = C - lower_val
                dC_upper = upper_val - C
                dC = max(dC_lower, dC_upper)
            except Exception as e:
                print(f"Warning: Confidence interval calculation failed: {e}")
                dC = 0
        
        return dC


# --- 4. IRLS-style weighted fitting (same logic for all models) ---

class FittingMixin:

    def _fit_with_weights(self, y_data, x_scaled, params, model, model_type, weight_power=1):

        n_iterations = 100
        convergence_threshold = 1e-10
        current_weights = np.ones(len(x_scaled))

        for i in range(n_iterations):
            result = model.fit(y_data, params, x=x_scaled, weights=current_weights)

            B_val = result.params['B'].value

            if model_type == 'exp':
                W_pos = np.exp(B_val * x_scaled)
            elif model_type == 'exp_sq':
                W_pos = np.exp(B_val * np.sqrt(x_scaled))
            elif model_type == 'power':
                W_pos = np.power(x_scaled, B_val)
                W_pos = np.nan_to_num(W_pos, nan=0.0, posinf=0.0, neginf=0.0)
                if np.all(W_pos == 0):
                    W_pos = np.ones_like(x_scaled)
            else:
                raise ValueError(f"Unknown model_type '{model_type}' in _fit_with_weights")

            W_pos /= np.mean(W_pos)

            new_weights = W_pos**weight_power
            new_weights /= np.mean(new_weights)

            current_weights = 0.5 * current_weights + 0.5 * new_weights
            print(f"Iteration {i}, weights: {current_weights}")

            old_params = np.array(list(params.valuesdict().values()))
            params = result.params
            new_params = np.array(list(params.valuesdict().values()))
            param_change = np.sum((old_params - new_params)**2)

            if param_change < convergence_threshold and i > 0:
                break

        return result
    
    def _optimize_weights_and_fit(self, y_data, x_scaled, params, model, model_name, model_type):
        print(f"  No known convergent value provided, using weight power n=1 for {model_name}")
        return self._fit_with_weights(y_data, x_scaled, params, model, model_type, weight_power=1)


# --- 5. Unified Fitter Class ---

class unified_extrapolator(UncertaintyCalculator, FittingMixin):

    def __init__(self, dataframe):
        if 'basis size' not in dataframe.columns:
            raise ValueError("Input DataFrame must contain a 'basis size' column.")
        self.df = dataframe
        self.x_data = self.df['basis size']
        self.results = {}
        self.uncertainties = {}
        self.column_name = None
        self.max_x = None
        self.known_convergent_value = None
        self.known_convergent_uncertainty = None

    def _fit_model(self, column_name, max_x, model_func, model_name, model_type):

        print(f"\n--- Fitting {model_name} Model ---")

        y_data = np.asarray(self.df[column_name].values, dtype=float)
        x_data = np.asarray(self.x_data.values, dtype=float)

        if max_x is None:
            max_x = x_data.max()

        if len(y_data) > 1:
            max_allowed_uncertainty = np.abs(y_data[0] - y_data[-1])
        else:
            max_allowed_uncertainty = np.inf

        # common scaling
        x_min = x_data.min()
        x_max = x_data.max()
        if x_max == x_min:
            raise ValueError("All x values identical; cannot scale.")
        x_scaled = (x_data - x_min) / (x_max - x_min)
        if model_type == 'power':
            x_scaled = x_scaled + 1e-6  # avoid 0

        print(f'x_scaled ({model_name}): {x_scaled}')

        model = Model(model_func)
        params = model.make_params()

        y_first = y_data[0]

        n_tail = min(5, max(3, len(y_data) // 3))
        C_guess = float(np.mean(y_data[-n_tail:]))
        print(f"{model_name}: C_guess = {C_guess}")
        params['C'].set(value=C_guess)

        A_guess = float(y_first - C_guess)
        if A_guess >= 0:
            params['A'].set(value=A_guess, min=1e-9, max=abs(A_guess)*3 + 1e-10)
        else:
            params['A'].set(value=A_guess, min=-abs(A_guess)*3 - 1e-10, max=-1e-9)

        if model_type == 'power':
            B_init = _init_B(x_scaled, y_data, C=C_guess, model_type='power')
        elif model_type == 'exp_sq':
            B_init = _init_B(x_scaled, y_data, C=C_guess, model_type='exp_sq')
        else:
            B_init = _init_B(x_scaled, y_data, C=C_guess, model_type='exp')

        params['B'].set(value=B_init, min=1e-6, max=50.0)

        print(f"{model_name} initial guesses: C={C_guess:.6g}, A={A_guess:.6g}, B={B_init:.6g}")

        result = self._optimize_weights_and_fit(y_data, x_scaled, params, model, model_name, model_type)

        raw_uncertainty = self._calculate_extrapolation_uncertainty(result)
        final_uncertainty = min(raw_uncertainty, max_allowed_uncertainty)
        if final_uncertainty < raw_uncertainty:
            print(f"  Uncertainty CAPPED. Original: {raw_uncertainty:.4e}, Capped: {final_uncertainty:.4e}")
        
        C = result.params['C'].value
        A = result.params['A'].value
        B = result.params['B'].value
        dC = result.params['C'].stderr if result.params['C'].stderr is not None else 0.0
        dA = result.params['A'].stderr if result.params['A'].stderr is not None else 0.0
        dB = result.params['B'].stderr if result.params['B'].stderr is not None else 0.0

        print(f"\n{model_name} fitted parameters:")
        print(f"  C = {C:.10f} ± {dC:.10f}  (asymptote)")
        print(f"  A = {A:.10f} ± {dA:.10f}  (amplitude)")
        print(f"  B = {B:.10f} ± {dB:.10f}  (decay / exponent)")

        residuals = result.residual
        chi2 = np.sum(residuals**2)
        rmse = np.sqrt(np.mean(residuals**2))
        print(f"{model_name} fit quality: χ² = {chi2:.6e}, RMSE = {rmse:.6e}")

        return result, final_uncertainty

    def fit_column(self, column_name, max_x):
        self.column_name = column_name
        self.max_x = max_x
        self.results = {}
        self.uncertainties = {}

        result1, unc1 = self._fit_model(column_name, max_x, exponential_decay,    "Exponential",      "exp")
        result2, unc2 = self._fit_model(column_name, max_x, exponential_decay_sq, "Exponential SQ",   "exp_sq")
        result3, unc3 = self._fit_model(column_name, max_x, power_law,            "Power Law",        "power")

        self.results = {
            'exponential': result1,
            'exponential_sq': result2,
            'power': result3
        }

        self.uncertainties = {
            'exponential': unc1,
            'exponential_sq': unc2,
            'power': unc3
        }

        print(f"\n{'='*60}")
        print(f"FINAL RESULTS FOR '{column_name}'")
        print(f"{'='*60}")

        model_names = ['Exponential', 'Exponential √x', 'Power Law']
        model_keys  = ['exponential', 'exponential_sq', 'power']

        for name, key in zip(model_names, model_keys):
            result = self.results[key]
            uncertainty = self.uncertainties[key]
            extrapolated_limit = result.params['C'].value

            print(f"\n{name} Model:")
            print(f"  Extrapolated Limit (C): {extrapolated_limit:.18f}")
            print(f"  Total Uncertainty:      ± {uncertainty:.18f}")

            if self.known_convergent_value is not None:
                difference = extrapolated_limit - self.known_convergent_value
                print(f"  Difference from known:  {difference:.18f}")

    def get_uncertainty_band(self, model_key, x_values_orig):

        if model_key not in self.results:
            return None, None
        
        result = self.results[model_key]
        x_min_orig = self.x_data.min()
        x_max_orig = self.x_data.max()
        
        f_min_curve = []
        f_max_curve = []
        
        if model_key == 'power':
            x_scaled = (x_values_orig - x_min_orig) / (x_max_orig - x_min_orig)
            x_scaled = x_scaled + 1e-6
            model_type = 'power'
        else:
            x_scaled = (x_values_orig - x_min_orig) / (x_max_orig - x_min_orig)
            model_type = 'exp_sq' if model_key == 'exponential_sq' else 'exp'
        
        for x_val in x_scaled:
            f_min, f_max = self._calculate_uncertainty_at_x(result.params, x_val, model_type)
            f_min_curve.append(f_min)
            f_max_curve.append(f_max)
        
        return np.array(f_min_curve), np.array(f_max_curve)

    def _draw_combined_plot(self, ax, zoom=False):
        if not self.results:
            return

        y_data = self.df[self.column_name]
        x_min_orig, x_max_orig = self.x_data.min(), self.x_data.max()

        colors = {
            'exponential':    '#1f77b4',
            'exponential_sq': '#ff7f0e',
            'power':          '#2ca02c'
        }

        ax.plot(self.x_data, y_data, 'ko', label='Original Data', markersize=6, zorder=5)

        model_names = {
            'exponential':    'Exponential',
            'exponential_sq': 'Exponential √x',
            'power':          'Power Law'
        }

        model_keys = ['exponential', 'exponential_sq', 'power']
        
        if zoom:
            x_data_values = self.x_data.values
            zoom_start_idx = max(0, int(len(x_data_values) * 0.75))
            x_zoom_min = x_data_values[zoom_start_idx]
            
            ax.set_xlim(x_zoom_min, self.max_x)

            y_values_in_zoom = []
            zoom_mask = self.x_data >= x_zoom_min
            y_values_in_zoom.extend(y_data[zoom_mask].values)

            plot_x_zoom = np.linspace(x_zoom_min, self.max_x, 100)
            for model_key in model_keys:
                result = self.results[model_key]
                if model_key == 'power':
                    plot_x_scaled = (plot_x_zoom - x_min_orig) / (x_max_orig - x_min_orig)
                    plot_x_scaled = plot_x_scaled + 1e-6
                else:
                    plot_x_scaled = (plot_x_zoom - x_min_orig) / (x_max_orig - x_min_orig)
                y_values_in_zoom.extend(result.eval(x=plot_x_scaled))

            if self.known_convergent_value is not None:
                y_values_in_zoom.append(self.known_convergent_value)

            if y_values_in_zoom:
                y_min_zoom = np.min(y_values_in_zoom)
                y_max_zoom = np.max(y_values_in_zoom)
                y_range = y_max_zoom - y_min_zoom
                
                y_padding = y_range * 0.2
                if y_padding < 1e-9:
                    y_padding = 0.1 * abs(y_min_zoom) if y_min_zoom != 0 else 0.1
                    
                ax.set_ylim(y_min_zoom - y_padding, y_max_zoom + y_padding)

        for model_key in model_keys:
            if model_key not in self.results:
                continue

            result = self.results[model_key]
            uncertainty = self.uncertainties[model_key]
            color = colors[model_key]

            plot_x_orig = np.linspace(x_min_orig, self.max_x, 400)

            if model_key == 'power':
                plot_x_scaled = (plot_x_orig - x_min_orig) / (x_max_orig - x_min_orig)
                plot_x_scaled = plot_x_scaled + 1e-6
                extrap_x_orig = np.arange(x_max_orig + 1000, self.max_x + 1, 1000)
                if len(extrap_x_orig) > 0:
                    extrap_x_scaled = (extrap_x_orig - x_min_orig) / (x_max_orig - x_min_orig)
                    extrap_x_scaled = extrap_x_scaled + 1e-6
            else:
                plot_x_scaled = (plot_x_orig - x_min_orig) / (x_max_orig - x_min_orig)
                extrap_x_orig = np.arange(x_max_orig + 1000, self.max_x + 1, 1000)
                if len(extrap_x_orig) > 0:
                    extrap_x_scaled = (extrap_x_orig - x_min_orig) / (x_max_orig - x_min_orig)

            plot_y = result.eval(x=plot_x_scaled)
            ax.plot(plot_x_orig, plot_y, '-', color=color,
                    label=f'{model_names[model_key]} Fit',
                    linewidth=2, zorder=4)

            if len(extrap_x_orig) > 0:
                extrap_y = result.eval(x=extrap_x_scaled)
                ax.plot(extrap_x_orig, extrap_y, 'o', color=color,
                        markersize=5, zorder=3)

            extrapolated_limit = result.params['C'].value
            ax.axhline(extrapolated_limit, color=color, linestyle='--',
                       linewidth=1.5,
                       label=f'{model_names[model_key]} Limit', zorder=2)

            if uncertainty > 0:
                upper_bound = extrapolated_limit + uncertainty
                lower_bound = extrapolated_limit - uncertainty
                
                ax.axhline(upper_bound, color=color, linestyle='--',
                           linewidth=1, alpha=0.5, zorder=1)
                ax.axhline(lower_bound, color=color, linestyle='--',
                           linewidth=1, alpha=0.5, zorder=1)
                
                ax.axhspan(lower_bound, upper_bound, color=color,
                           alpha=0.12, zorder=0,
                           label=f'{model_names[model_key]} Uncertainty (ΔC)')

        if self.known_convergent_value is not None:
            ax.axhline(self.known_convergent_value, color='black', linestyle=':',
                       linewidth=2.5,
                       label=f'Known CV ({self.known_convergent_value:.6f})', zorder=6)
            if self.known_convergent_uncertainty is not None:
                upper_cv = self.known_convergent_value + self.known_convergent_uncertainty
                lower_cv = self.known_convergent_value - self.known_convergent_uncertainty
                ax.axhline(upper_cv, color='black', linestyle='--',
                           linewidth=1, alpha=0.5, zorder=1)
                ax.axhline(lower_cv, color='black', linestyle='--',
                           linewidth=1, alpha=0.5, zorder=1)
                ax.axhspan(lower_cv, upper_cv, color='black', alpha=0.08,
                           zorder=0, label='Known CV Uncertainty')

        ax.set_xlabel("Basis Size", fontsize=11)
        ax.set_ylabel(self.column_name, fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='best', fontsize=9)
        
        if zoom:
            ax.set_title(f"Zoom: All Models", fontsize=12, fontweight='bold')
        else:
            ax.set_title(f"All Models Overview", fontsize=12, fontweight='bold')

    def plot_all_results(self):
        if not self.results:
            print("No results to plot. Run fit_column() first.")
            return

        fig = plt.figure(figsize=(18, 7))

        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

        ax_left = fig.add_subplot(gs[0, 0])
        self._draw_combined_plot(ax_left, zoom=False)

        ax_right = fig.add_subplot(gs[0, 1])
        self._draw_combined_plot(ax_right, zoom=True)

        fig.suptitle(f"Unified Extrapolation Results for '{self.column_name}'",
                     fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def fit(self):
        available_columns = self.df.columns.drop('basis size').tolist()
        print("Available columns to analyze:")
        for col in available_columns:
            print(f"- {col}")
        print("-" * 30)

        self.known_convergent_value = None
        self.known_convergent_uncertainty = None

        column_name = input("Please enter the name of the column to fit: ")
        if column_name.lower() in ['q', 'quit']:
            print("Exiting.")
            return

        if column_name not in available_columns:
            print(f"Error: Invalid column name '{column_name}'. Please choose from the list above.")
            return

        try:
            max_x_val = int(input(f"Enter the extrapolation limit for '{column_name}': "))
        except ValueError:
            print("Invalid input. Using the max value from data as the limit.")
            max_x_val = self.x_data.max()

        cv_input = input("Enter a known convergent value for comparison (or press Enter to skip): ").strip()
        if cv_input:
            try:
                self.known_convergent_value = float(cv_input)
                cv_unc_input = input(f"Enter the uncertainty for {self.known_convergent_value} (or press Enter to skip): ").strip()
                if cv_unc_input:
                    try:
                        self.known_convergent_uncertainty = float(cv_unc_input)
                    except ValueError:
                        print("Invalid number for uncertainty. It will be ignored.")
            except ValueError:
                print("Invalid number for convergent value. It will be ignored.")
                self.known_convergent_value = None

        self.fit_column(column_name, max_x_val)
        self.plot_all_results()