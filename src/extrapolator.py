import matplotlib.patches as patches
from lmfit import Model
import numpy as np
import matplotlib.pyplot as plt

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

        weight_powers = [1, 2]

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

    def _draw_model_plot(self, ax, model_key, model_name):
        """Draw a specific model's fit on the given axis."""
        if model_key not in self.results:
            return

        result = self.results[model_key]
        uncertainty = self.uncertainties[model_key]
        y_data = self.df[self.column_name]
        x_min_orig, x_max_orig = self.x_data.min(), self.x_data.max()

        # Plot original data
        ax.plot(self.x_data, y_data, 'o', label='Original Data', markersize=6)

        # Plot fitted curve
        if model_key == 'power':
            plot_x_orig = np.linspace(x_min_orig, self.max_x, 400)
            plot_x_scaled = plot_x_orig / x_max_orig
            extrap_x_orig = np.arange(x_max_orig + 1000, self.max_x + 1, 1000)
            if len(extrap_x_orig) > 0:
                extrap_x_scaled = extrap_x_orig / x_max_orig
        else:
            plot_x_orig = np.linspace(x_min_orig, self.max_x, 400)
            plot_x_scaled = (plot_x_orig - x_min_orig) / (x_max_orig - x_min_orig)
            extrap_x_orig = np.arange(x_max_orig + 1000, self.max_x + 1, 1000)
            if len(extrap_x_orig) > 0:
                extrap_x_scaled = (extrap_x_orig - x_min_orig) / (x_max_orig - x_min_orig)

        # Plot fitted curve
        plot_y = result.eval(x=plot_x_scaled)
        ax.plot(plot_x_orig, plot_y, '-', label=f'{model_name} Fit', linewidth=2)

        # Plot extrapolated points
        if len(extrap_x_orig) > 0:
            extrap_y = result.eval(x=extrap_x_scaled)
            ax.plot(extrap_x_orig, extrap_y, 'o', color='red', markersize=6, label='Extrapolated Points')

        # Plot extrapolated limit and uncertainty
        extrapolated_limit = result.params['C'].value
        ax.axhline(extrapolated_limit, color='red', linestyle='--', label=f'Extrapolated Limit')
        if uncertainty > 0:
            ax.axhspan(extrapolated_limit - uncertainty, extrapolated_limit + uncertainty,
                      color='red', alpha=0.15, label='Limit Uncertainty (ΔC)')

        # Plot known convergent value if provided
        if self.known_convergent_value is not None:
            ax.axhline(self.known_convergent_value, color='black', linestyle=':', linewidth=2.5,
                      label=f'Known CV ({self.known_convergent_value:.6f})')
            if self.known_convergent_uncertainty is not None:
                ax.axhspan(self.known_convergent_value - self.known_convergent_uncertainty,
                          self.known_convergent_value + self.known_convergent_uncertainty,
                          color='black', alpha=0.15, label='Known CV Uncertainty')

        ax.set_xlabel("Basis Size")
        ax.set_ylabel(self.column_name)
        ax.set_title(f"{model_name} Model")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

    def _draw_zoom_plot(self, ax, model_key, model_name):
        """Draw a zoomed view of a specific model's fit."""
        if model_key not in self.results:
            return

        result = self.results[model_key]
        uncertainty = self.uncertainties[model_key]
        y_data = self.df[self.column_name]
        x_min_orig, x_max_orig = self.x_data.min(), self.x_data.max()

        # Plot original data
        ax.plot(self.x_data, y_data, 'o', label='Original Data', markersize=6)

        # Determine zoom range
        x_data_values = self.x_data.values
        zoom_start_idx = max(0, int(len(x_data_values) * 0.75))
        x_zoom_min = x_data_values[zoom_start_idx]
        if len(x_data_values) <= 3:
            x_zoom_min = x_data_values[0]

        ax.set_xlim(x_zoom_min, self.max_x)

        # Plot fitted curve in zoom range
        if model_key == 'power':
            plot_x_orig = np.linspace(x_zoom_min, self.max_x, 200)
            plot_x_scaled = plot_x_orig / x_max_orig
            extrap_x_orig = np.arange(x_max_orig + 1000, self.max_x + 1, 1000)
            if len(extrap_x_orig) > 0:
                extrap_x_scaled = extrap_x_orig / x_max_orig
        else:
            plot_x_orig = np.linspace(x_zoom_min, self.max_x, 200)
            plot_x_scaled = (plot_x_orig - x_min_orig) / (x_max_orig - x_min_orig)
            extrap_x_orig = np.arange(x_max_orig + 1000, self.max_x + 1, 1000)
            if len(extrap_x_orig) > 0:
                extrap_x_scaled = (extrap_x_orig - x_min_orig) / (x_max_orig - x_min_orig)

        # Plot fitted curve
        plot_y = result.eval(x=plot_x_scaled)
        ax.plot(plot_x_orig, plot_y, '-', label=f'{model_name} Fit', linewidth=2)

        # Plot extrapolated points
        if len(extrap_x_orig) > 0:
            extrap_y = result.eval(x=extrap_x_scaled)
            ax.plot(extrap_x_orig, extrap_y, 'o', color='red', markersize=6, label='Extrapolated Points')

        # Plot extrapolated limit and uncertainty
        extrapolated_limit = result.params['C'].value
        ax.axhline(extrapolated_limit, color='red', linestyle='--', label=f'Extrapolated Limit')
        if uncertainty > 0:
            ax.axhspan(extrapolated_limit - uncertainty, extrapolated_limit + uncertainty,
                      color='red', alpha=0.15, label='Limit Uncertainty (ΔC)')

        # Plot known convergent value if provided
        if self.known_convergent_value is not None:
            ax.axhline(self.known_convergent_value, color='black', linestyle=':', linewidth=2.5,
                      label=f'Known CV ({self.known_convergent_value:.18f})')
            if self.known_convergent_uncertainty is not None:
                ax.axhspan(self.known_convergent_value - self.known_convergent_uncertainty,
                          self.known_convergent_value + self.known_convergent_uncertainty,
                          color='black', alpha=0.15, label='Known CV Uncertainty')

        # Smart y-range determination
        mask = self.x_data >= x_zoom_min
        visible_y_data = y_data[mask].values

        if len(visible_y_data) > 0:
            y_min_zoom = min(np.min(visible_y_data), extrapolated_limit)
            y_max_zoom = max(np.max(visible_y_data), extrapolated_limit)
        else:
            y_min_zoom = min(y_data.min(), extrapolated_limit)
            y_max_zoom = max(y_data.max(), extrapolated_limit)

        y_range = y_max_zoom - y_min_zoom
        if y_range < 1e-12:
            y_range = abs(extrapolated_limit * 0.1) if abs(extrapolated_limit) > 1e-12 else 0.1

        y_padding = y_range * 0.3
        ax.set_ylim(y_min_zoom - y_padding, y_max_zoom + y_padding)

        ax.set_xlabel("Basis Size")
        ax.set_ylabel(self.column_name)
        ax.set_title(f"Zoom: {model_name} Model")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

    def plot_all_results(self):
        """Create a 3x2 grid plot with all models and their zoomed views."""
        if not self.results:
            print("No results to plot. Run fit_column() first.")
            return

        fig = plt.figure(figsize=(20, 20))

        # Create 3 rows x 2 columns grid
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])

        model_names = ['Exponential', 'Exponential √x', 'Power Law']
        model_keys = ['exponential', 'exponential_sq', 'power']

        # Plot full views (left column)
        for i, (key, name) in enumerate(zip(model_keys, model_names)):
            ax = fig.add_subplot(gs[i, 0])
            self._draw_model_plot(ax, key, name)

        # Plot zoomed views (right column)
        for i, (key, name) in enumerate(zip(model_keys, model_names)):
            ax = fig.add_subplot(gs[i, 1])
            self._draw_zoom_plot(ax, key, name)

        fig.suptitle(f"Unified Extrapolation Results for '{self.column_name}'", fontsize=18)
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