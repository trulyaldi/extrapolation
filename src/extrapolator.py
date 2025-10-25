import matplotlib.patches as patches
from lmfit import Model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- 1. Define the Mathematical Model Functions ---

# Exponential Models (kept separate as required by lmfit)
def exponential_decay(x, C, A, B):
    """A 3-parameter exponential decay function."""
    return C + A * np.exp(-B * x)

def exponential_decay_sq(x, C, A, B):
    """A 3-parameter exponential decay function with square root."""
    return C + A * np.exp(-B * (x ** (0.5)))

# Power Law Model
def power_law(x, C, A, B):
    """A power law function."""
    return C + A * np.power(x, -B)

# --- 2. Shared Utility Functions ---

def _init_B(x, y, C=None):
    """
    Improved estimation using log-log regression.
    Used for both exponential and power law initial guesses.
    """
    x, y = np.asarray(x), np.asarray(y)
    if C is None:
        C = y[-1]
    
    # Remove the asymptote for better linearization
    y_shifted = y - C
    
    # Filter out invalid values
    valid_mask = (x > 0) & (y_shifted > 0)
    if np.sum(valid_mask) < 3:
        return 0.1  # fallback
    
    x_valid, y_valid = x[valid_mask], y_shifted[valid_mask]
    
    # Log-log linear regression: log(y-C) = log(A) - B*log(x)
    try:
        log_x = np.log(x_valid)
        log_y = np.log(y_valid)
        # Fit: log_y = a - B * log_x
        B_est = -np.polyfit(log_x, log_y, 1)[0]
        return float(np.clip(B_est, 1e-6, 5.0))
    except:
        return 0.1

def _init_B_exponential(x, y, C=None, use_sqrt=False):
    """
    Improved B estimation for exponential models using log-linear regression.
    
    For exponential: f(x) = C + A * exp(-B * x)
    Linear form: ln(y - C) = ln(A) - B * x
    Slope = -B
    
    For exponential_sq: f(x) = C + A * exp(-B * sqrt(x))
    Linear form: ln(y - C) = ln(A) - B * sqrt(x)
    Slope w.r.t. sqrt(x) = -B
    """
    x, y = np.asarray(x), np.asarray(y)
    if C is None:
        C = y[-1]
    
    y_shifted = y - C
    
    # Filter out invalid values for log transform
    valid_mask = (y_shifted > 1e-15)
    if np.sum(valid_mask) < 3:
        return 0.1  # fallback
    
    x_valid = x[valid_mask]
    y_valid = y_shifted[valid_mask]
    
    try:
        log_y = np.log(y_valid)
        
        if use_sqrt:
            # For exp_sq: use sqrt(x) as the transformed variable
            x_transformed = np.sqrt(x_valid)
        else:
            # For regular exp: use x directly
            x_transformed = x_valid
        
        # Linear regression: log_y = intercept - B * x_transformed
        # polyfit returns [slope, intercept]
        coeffs = np.polyfit(x_transformed, log_y, 1)
        slope = coeffs[0]
        B_est = -slope  # Because slope = -B
        
        # Ensure B is positive and in reasonable range
        B_est = float(np.clip(B_est, 1e-6, 50.0))
        
        return B_est
    except:
        return 0.1

# --- 3. Base Uncertainty Calculator (Shared Logic) ---

class UncertaintyCalculator:
    """Base class for uncertainty calculations shared between model types."""
    
    def _calculate_uncertainty_at_x(self, params, x_value, model_type='exp'):
        """
        Calculate uncertainty bounds for exponential or power law models at a specific x value.
        
        Exponential models: f(x) = C + A * exp(-B * x_term)
        Power law model: f(x) = C + A / x^B
        
        Handles all parameter sign combinations.
        
        Returns (f_min, f_max) at the given x value.
        """
        C = params['C'].value
        A = params['A'].value
        B = params['B'].value
        
        dC = params['C'].stderr if params['C'].stderr is not None else 0
        dA = params['A'].stderr if params['A'].stderr is not None else 0
        dB = params['B'].stderr if params['B'].stderr is not None else 0
        
        try:
            if model_type == 'power':
                return self._calculate_power_uncertainty(C, A, B, dC, dA, dB, x_value)
            else:  # exponential or exp_sq
                return self._calculate_exponential_uncertainty(C, A, B, dC, dA, dB, x_value, model_type)
        except (OverflowError, ValueError, ZeroDivisionError) as e:
            print(f"Warning: Uncertainty calculation failed at x={x_value}: {e}")
            return C, C
    
    def _calculate_exponential_uncertainty(self, C, A, B, dC, dA, dB, x_value, model_type='exp'):
        """Calculate exponential model uncertainty."""
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
        """Calculate power law model uncertainty."""
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
    
    def _calculate_extrapolation_uncertainty(self, params):
        """
        Calculate uncertainty at the extrapolation limit (x -> infinity).
        At infinity, the model term -> 0, so uncertainty is simply ΔC.
        """
        C = params['C'].value
        dC = params['C'].stderr if params['C'].stderr is not None else 0
        return dC

# --- 4. Shared Fitting Logic (Mixin) ---

class FittingMixin:
    """Mixin class containing shared fitting logic for all model types."""
    
    def _trim_early_inconsistent_data(self, y_data, x_data, column_name):
        """
        Trim early inconsistent data points based on global trend.
        Used for all model types.
        """
        y_data = np.asarray(y_data, dtype=float)
        x_data = np.asarray(x_data)
        
        if len(y_data) < 4:
            return x_data, y_data
        
        # Detect global trend direction
        slope = np.polyfit(np.arange(len(y_data)), y_data, 1)[0]
        trend = "decreasing" if slope < 0 else "increasing"
        
        # Identify the key point (max for decreasing, min for increasing)
        if trend == "decreasing":
            key_idx = np.argmax(y_data)
        else:
            key_idx = np.argmin(y_data)
        
        # Trim only if the key point is not at the start
        if key_idx > 0:
            x_trimmed = x_data[key_idx:]
            y_trimmed = y_data[key_idx:]
            print(f"Removed {key_idx} early inconsistent point(s) for '{column_name}'")
        else:
            x_trimmed, y_trimmed = x_data, y_data
        
        return x_trimmed, y_trimmed
    
    def _fit_with_weights(self, y_data, x_scaled, params, model, weight_power):
        """
        Helper method to fit with specific weight power.
        Universal for all model types.
        """
        n_iterations = 100
        convergence_threshold = 1e-8
        current_weights = np.ones(len(x_scaled))

        for i in range(n_iterations):
            result = model.fit(y_data, params, x=x_scaled, weights=current_weights)
            residuals = result.residual

            # Calculate weights based on model type
            if 'exp' in model.func.__name__:
                if 'sq' in model.func.__name__:
                    W_pos = np.exp(params['B'].value * np.sqrt(x_scaled))
                else:
                    W_pos = np.exp(params['B'].value * x_scaled)
            else:  # power law
                W_pos = x_scaled ** params['B'].value

            W_pos /= np.mean(W_pos)

            # Apply weight power
            new_weights = W_pos ** weight_power
            new_weights /= np.mean(new_weights)

            current_weights = 0.5 * current_weights + 0.5 * new_weights

            old_params = np.array(list(params.valuesdict().values()))
            params = result.params
            new_params = np.array(list(params.valuesdict().values()))
            param_change = np.sum((old_params - new_params)**2)

            if param_change < convergence_threshold and i > 0:
                break

        return result
    
    def _optimize_weights_and_fit(self, y_data, x_scaled, params, model, model_name):
        """
        Optimize weights and fit the model.
        Universal for all model types.
        """
        best_result = None
        best_distance = np.inf
        best_n = None

        weight_powers = [1]

        if self.known_convergent_value is not None:
            print(f"Optimizing {model_name} weights using known value: {self.known_convergent_value:.8f}")

            for n in weight_powers:
                try:
                    temp_result = self._fit_with_weights(
                        y_data, x_scaled, params.copy(), model, n
                    )

                    # Calculate distance to known value
                    extrapolated_limit = temp_result.params['C'].value
                    distance = abs(extrapolated_limit - self.known_convergent_value)

                    if distance < best_distance:
                        best_distance = distance
                        best_result = temp_result
                        best_n = n

                except Exception as e:
                    continue

            if best_result is not None:
                print(f"  Best weight power: {best_n} with distance {best_distance:.2e}")
                return best_result
            else:
                print(f"  Weight optimization failed, using default weights (n=1)")
                result = self._fit_with_weights(
                    y_data, x_scaled, params.copy(), model, 1
                )
                return result
        else:
            # No known value, use default weight power
            print(f"  No known convergent value provided, using weight power n=1 for {model_name}")
            result = self._fit_with_weights(
                y_data, x_scaled, params.copy(), model, 1
            )
            return result

# --- 5. Create the Unified Fitter Class ---

class unified_extrapolator(UncertaintyCalculator, FittingMixin):
    """
    A unified class to fit exponential and power law models using
    Self-Consistent Iteratively Reweighted Least Squares (IRLS) approach
    and calculate comprehensive uncertainties for extrapolated limits.
    """
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
        """Fit a specific model to the data."""
        print(f"\n--- Fitting {model_name} Model ---")

        y_data = self.df[column_name].values
        y_data = np.asarray(y_data, dtype=float)
        x_data = np.asarray(self.x_data.values)

        # Trim early inconsistent data points
        x_trimmed, y_trimmed = self._trim_early_inconsistent_data(y_data, x_data, column_name)

        x_min_orig, x_max_orig = x_trimmed.min(), x_trimmed.max()

        if model_type == 'power':
            x_scaled = x_trimmed / x_max_orig
        else:
            x_scaled = (x_trimmed - x_min_orig) / (x_max_orig - x_min_orig)

        model = Model(model_func)
        params = model.make_params()

        # Initial parameter guesses using trimmed data
        y_last, y_first = y_trimmed[-1], y_trimmed[0]
        
        # IMPROVED: Use averaged tail for C estimate (better asymptote guess)
        n_tail = min(5, max(3, len(y_trimmed) // 3))
        C_guess = np.mean(y_trimmed[-n_tail:])
        params['C'].set(value=C_guess)

        if model_type == 'power':
            # Power law specific initialization
            B0 = _init_B(x_scaled, y_trimmed, C=C_guess)
            params['B'].set(value=B0, min=1e-6, max=5)

            amplitude_guess = y_first - C_guess
            amplitude_bound = abs(amplitude_guess) + 1e-10

            if amplitude_guess >= 0:
                params['A'].set(value=amplitude_guess, min=1e-9, max=amplitude_bound * 2)
            else:
                params['A'].set(value=amplitude_guess, min=-amplitude_bound * 2, max=-1e-9)
        else:
            # ===== IMPROVED EXPONENTIAL INITIALIZATION =====
            # Step 1: Better A estimate
            A_guess = y_first - C_guess
            
            if A_guess >= 0:
                params['A'].set(value=A_guess, min=1e-9, max=abs(A_guess) * 3 + 1e-10)
            else:
                params['A'].set(value=A_guess, min=abs(A_guess) * 3 * -1 - 1e-10, max=-1e-9)

            # Step 2: CRUCIAL - Use log-linear regression for B estimation
            # This is similar to how exp_sq works but tailored for exponential
            use_sqrt = (model_type == 'exp_sq')
            B_init = _init_B_exponential(x_scaled, y_trimmed, C=C_guess, use_sqrt=use_sqrt)
            
            print(f"  Initial guesses: C={C_guess:.6f}, A={A_guess:.6f}, B={B_init:.6f}")
            print(f"  Model type: {'exp_sq' if use_sqrt else 'exp'}")
            
            params['B'].set(value=B_init, min=1e-6, max=50.0)

        # Optimize weights and fit using trimmed data
        result = self._optimize_weights_and_fit(y_trimmed, x_scaled, params, model, model_name)

        # Calculate uncertainty at extrapolation limit
        uncertainty = self._calculate_extrapolation_uncertainty(result.params)
        
        # Print detailed parameter information for debugging
        C = result.params['C'].value
        A = result.params['A'].value
        B = result.params['B'].value
        dC = result.params['C'].stderr if result.params['C'].stderr is not None else 0
        dA = result.params['A'].stderr if result.params['A'].stderr is not None else 0
        dB = result.params['B'].stderr if result.params['B'].stderr is not None else 0
        
        print(f"Fitted parameters:")
        print(f"  C = {C:.10f} ± {dC:.10f} (asymptote)")
        print(f"  A = {A:.10f} ± {dA:.10f} (amplitude)")
        print(f"  B = {B:.10f} ± {dB:.10f} (decay rate)")
        print(f"Parameter signs: C={'positive' if C>=0 else 'negative'}, "
              f"A={'positive' if A>=0 else 'negative'}, "
              f"B={'positive' if B>=0 else 'negative'}")
        
        # Show fit quality
        residuals = result.residual
        chi2 = np.sum(residuals**2)
        rmse = np.sqrt(np.mean(residuals**2))
        print(f"Fit quality: χ² = {chi2:.6e}, RMSE = {rmse:.6e}")

        return result, uncertainty

    def fit_column(self, column_name, max_x):
        """
        Fit all three models to the specified column.
        """
        self.column_name = column_name
        self.max_x = max_x
        self.results = {}
        self.uncertainties = {}

        # Fit all three models
        result1, unc1 = self._fit_model(column_name, max_x, exponential_decay, "Exponential", "exp")
        result2, unc2 = self._fit_model(column_name, max_x, exponential_decay_sq, "Exponential SQ", "exp_sq")
        result3, unc3 = self._fit_model(column_name, max_x, power_law, "Power Law", "power")

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

        # Print results
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS FOR '{column_name}'")
        print(f"{'='*60}")

        model_names = ['Exponential', 'Exponential √x', 'Power Law']
        model_keys = ['exponential', 'exponential_sq', 'power']

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
        """
        Get uncertainty band for plotting any model.
        
        Args:
            model_key: 'exponential', 'exponential_sq', or 'power'
            x_values_orig: Original (unscaled) x values
            
        Returns:
            f_min_curve, f_max_curve: Arrays of lower and upper bounds
        """
        if model_key not in self.results:
            return None, None
        
        result = self.results[model_key]
        x_min_orig = self.x_data.min()
        x_max_orig = self.x_data.max()
        
        f_min_curve = []
        f_max_curve = []
        
        if model_key == 'power':
            x_scaled = x_values_orig / x_max_orig
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
        """Draw all three models on a single axis."""
        if not self.results:
            return

        y_data = self.df[self.column_name]
        x_min_orig, x_max_orig = self.x_data.min(), self.x_data.max()

        # Define distinct colors for each model
        colors = {
            'exponential': '#1f77b4',      # Blue
            'exponential_sq': '#ff7f0e',   # Orange
            'power': '#2ca02c'              # Green
        }

        # Plot original data
        ax.plot(self.x_data, y_data, 'ko', label='Original Data', markersize=6, zorder=5)

        model_names = {
            'exponential': 'Exponential',
            'exponential_sq': 'Exponential √x',
            'power': 'Power Law'
        }

        model_keys = ['exponential', 'exponential_sq', 'power']
        
        # --- MODIFIED ZOOM LOGIC ---
        if zoom:
            # Determine the x-range for the zoom (e.g., last 25% of the data)
            x_data_values = self.x_data.values
            zoom_start_idx = max(0, int(len(x_data_values) * 0.75))
            x_zoom_min = x_data_values[zoom_start_idx]
            
            ax.set_xlim(x_zoom_min, self.max_x)

            # Find the min/max y-values within this new x-range to set y-limits
            y_values_in_zoom = []
            
            # 1. Add original data points in the zoom range
            zoom_mask = self.x_data >= x_zoom_min
            y_values_in_zoom.extend(y_data[zoom_mask].values)

            # 2. Add fitted curve points in the zoom range
            plot_x_zoom = np.linspace(x_zoom_min, self.max_x, 100)
            for model_key in model_keys:
                result = self.results[model_key]
                if model_key == 'power':
                    plot_x_scaled = plot_x_zoom / x_max_orig
                else:
                    plot_x_scaled = (plot_x_zoom - x_min_orig) / (x_max_orig - x_min_orig)
                y_values_in_zoom.extend(result.eval(x=plot_x_scaled))

            # 3. Add the known CV value to ensure it's visible
            if self.known_convergent_value is not None:
                y_values_in_zoom.append(self.known_convergent_value)

            # Calculate y-limits with padding
            if y_values_in_zoom:
                y_min_zoom = np.min(y_values_in_zoom)
                y_max_zoom = np.max(y_values_in_zoom)
                y_range = y_max_zoom - y_min_zoom
                
                # Add padding to the y-axis
                y_padding = y_range * 0.2  # 20% padding
                if y_padding < 1e-9: # Add a small absolute padding if range is tiny
                    y_padding = 0.1 * abs(y_min_zoom) if y_min_zoom != 0 else 0.1
                    
                ax.set_ylim(y_min_zoom - y_padding, y_max_zoom + y_padding)
        # --- END OF MODIFIED ZOOM LOGIC ---

        # Plot each model
        for model_key in model_keys:
            if model_key not in self.results:
                continue

            result = self.results[model_key]
            uncertainty = self.uncertainties[model_key]
            color = colors[model_key]

            # Generate x values for plotting
            plot_x_orig = np.linspace(x_min_orig, self.max_x, 400)

            # Scale x values based on model type
            if model_key == 'power':
                plot_x_scaled = plot_x_orig / x_max_orig
                extrap_x_orig = np.arange(x_max_orig + 1000, self.max_x + 1, 1000)
                if len(extrap_x_orig) > 0:
                    extrap_x_scaled = extrap_x_orig / x_max_orig
            else:
                plot_x_scaled = (plot_x_orig - x_min_orig) / (x_max_orig - x_min_orig)
                extrap_x_orig = np.arange(x_max_orig + 1000, self.max_x + 1, 1000)
                if len(extrap_x_orig) > 0:
                    extrap_x_scaled = (extrap_x_orig - x_min_orig) / (x_max_orig - x_min_orig)

            # Plot fitted curve
            plot_y = result.eval(x=plot_x_scaled)
            ax.plot(plot_x_orig, plot_y, '-', color=color, label=f'{model_names[model_key]} Fit',
                   linewidth=2, zorder=4)

            # Plot extrapolated points
            if len(extrap_x_orig) > 0:
                extrap_y = result.eval(x=extrap_x_scaled)
                ax.plot(extrap_x_orig, extrap_y, 'o', color=color, markersize=5, zorder=3)

            # Plot extrapolated limit
            extrapolated_limit = result.params['C'].value
            ax.axhline(extrapolated_limit, color=color, linestyle='--', linewidth=1.5,
                      label=f'{model_names[model_key]} Limit', zorder=2)

            # Plot uncertainty band with dashed lines
            if uncertainty > 0:
                # Create uncertainty band with dashed border lines
                upper_bound = extrapolated_limit + uncertainty
                lower_bound = extrapolated_limit - uncertainty
                
                ax.axhline(upper_bound, color=color, linestyle='--', linewidth=1, alpha=0.5, zorder=1)
                ax.axhline(lower_bound, color=color, linestyle='--', linewidth=1, alpha=0.5, zorder=1)
                
                # Fill uncertainty band with light shade
                ax.axhspan(lower_bound, upper_bound, color=color, alpha=0.12, zorder=0,
                          label=f'{model_names[model_key]} Uncertainty (ΔC)')

        # Plot known convergent value if provided
        if self.known_convergent_value is not None:
            ax.axhline(self.known_convergent_value, color='black', linestyle=':', linewidth=2.5,
                      label=f'Known CV ({self.known_convergent_value:.6f})', zorder=6)
            if self.known_convergent_uncertainty is not None:
                upper_cv = self.known_convergent_value + self.known_convergent_uncertainty
                lower_cv = self.known_convergent_value - self.known_convergent_uncertainty
                ax.axhline(upper_cv, color='black', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
                ax.axhline(lower_cv, color='black', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
                ax.axhspan(lower_cv, upper_cv, color='black', alpha=0.08, zorder=0,
                          label='Known CV Uncertainty')

        ax.set_xlabel("Basis Size", fontsize=11)
        ax.set_ylabel(self.column_name, fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='best', fontsize=9)
        
        if zoom:
            ax.set_title(f"Zoom: All Models", fontsize=12, fontweight='bold')
        else:
            ax.set_title(f"All Models Overview", fontsize=12, fontweight='bold')

    def plot_all_results(self):
        """Create a 1x2 grid plot with all models on each subplot."""
        if not self.results:
            print("No results to plot. Run fit_column() first.")
            return

        fig = plt.figure(figsize=(18, 7))

        # Create 1 row x 2 columns grid
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

        # Left plot: Overall view with all models
        ax_left = fig.add_subplot(gs[0, 0])
        self._draw_combined_plot(ax_left, zoom=False)

        # Right plot: Zoomed view with all models
        ax_right = fig.add_subplot(gs[0, 1])
        self._draw_combined_plot(ax_right, zoom=True)

        fig.suptitle(f"Unified Extrapolation Results for '{self.column_name}'", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def fit(self):
        """
        Runs a single interactive fit and then exits.
        """
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
    """
    Generates and displays a grid of plots for each feature in a DataFrame
    against the 'basis size' column.

    Args:
        df (pd.DataFrame): The input DataFrame. It must contain a column
                           named 'basis size' (case-insensitive).
        n_cols (int): The number of columns to use in the plot grid.
                      Defaults to 4.
    """
    # Create a copy to avoid changing the original DataFrame
    df_plot = df.copy()

    # Standardize column names to lowercase for consistency
    df_plot.columns = [col.lower() for col in df_plot.columns]

    # Check for the required 'basis size' column
    if 'basis size' not in df_plot.columns:
        raise ValueError("Input DataFrame must contain a 'basis size' column.")

    # Prepare the data and identify features to plot
    df_plot['basis size'] = df_plot['basis size'].astype(int)
    features = sorted([col for col in df_plot.columns if col != 'basis size'])
    n_features = len(features)

    # Handle the case of no features to plot
    if n_features == 0:
        print("No feature columns found to plot.")
        return

    # Calculate the required number of rows for the grid
    n_rows = (n_features + n_cols - 1) // n_cols

    # Create the figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), sharex=True)
    fig.suptitle('Features vs. Basis Size', fontsize=16, y=1.03)

    # Flatten the axes array for easy, single-loop iteration
    axes = axes.flatten()

    # Plot each feature against the basis size
    for i, feature in enumerate(features):
        ax = axes[i]
        ax.scatter(df_plot['basis size'], df_plot[feature], marker='o')

        # Format titles and labels for readability
        ax.set_title(feature.replace('_', ' ').title())
        ax.set_xlabel('Basis Size')
        ax.set_ylabel('Value')
        ax.grid(True, linestyle='--', alpha=0.6)

    # Clean up by removing any empty, unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout to prevent plot elements from overlapping
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()