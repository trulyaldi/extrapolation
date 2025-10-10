
import matplotlib.patches as patches
from lmfit import Model
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define the Mathematical Model Functions ---
def exponential_decay(x, C, A, B):
    """A 3-parameter exponential decay function."""
    return C + A * np.exp(-B * x)

def exponential_decay_sq(x, C, A, B):
    """A 3-parameter exponential decay function with square root."""
    return C + A * np.exp(-B * (x ** (0.5)))

def power_law(x, C, A, B):
    """A power law function."""
    return C + A * np.power(x, -B)

def _init_B(x, y, C=None):
    """
    Improved estimation using log-log regression.
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
    
def _scale_covariance_with_weights(result, weights):
    """
    Scale parameter uncertainties based on reduced chi-squared.
    Used by both exponential and power law fits.
    """
    residuals = result.residual
    weighted_residuals = residuals * np.sqrt(weights)
    n_data = len(residuals)
    n_params = len(result.params)
    chi2_reduced = np.sum(weighted_residuals**2) / (n_data - n_params)

    if chi2_reduced > 1.0:
        scale_factor = np.sqrt(chi2_reduced)
        
        for param_name in result.params:
            if result.params[param_name].stderr is not None:
                result.params[param_name].stderr *= scale_factor
        
        print(f"      Scaled uncertainties by {scale_factor:.3f} (χ²_red = {chi2_reduced:.3f})")

    return result
# --- 2. Create the Unified Fitter Class ---
class unified_extrapolator:
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

    def _calculate_exponential_uncertainty_at_x(self, result, x_value, model_type='exp'):
        """
        Calculate uncertainty bounds for exponential models at a specific x value.
        
        For exp: f(x) = C + A * exp(-B * x)
        For exp_sq: f(x) = C + A * exp(-B * sqrt(x))
        
        Handles all cases:
        - A > 0, C > 0, B > 0
        - A > 0, C < 0, B > 0
        - A < 0, C > 0, B > 0
        - A < 0, C < 0, B > 0
        
        B is always > 0 for exponential models.
        
        Returns (f_min, f_max) at the given x value.
        """
        params = result.params
        
        C = params['C'].value
        A = params['A'].value
        B = params['B'].value
        
        dC = params['C'].stderr if params['C'].stderr is not None else 0
        dA = params['A'].stderr if params['A'].stderr is not None else 0
        dB = params['B'].stderr if params['B'].stderr is not None else 0
        
        try:
            # Calculate exponent term based on model type
            if model_type == 'exp_sq':
                x_term = np.sqrt(x_value)
            else:  # 'exp'
                x_term = x_value
            
            # For exponential: A * exp(-B * x_term)
            # When B increases, exp(-B*x_term) decreases (for x_term > 0)
            
            if A >= 0:
                # A is positive or zero
                # Lower bound: minimize C term, minimize A*exp(-B*x) term
                # To minimize A*exp(-B*x) when A>0: use (A-dA)*exp(-(B+dB)*x)
                term_min = (A - dA) * np.exp(-(B + dB) * x_term)
                # Upper bound: maximize C term, maximize A*exp(-B*x) term
                # To maximize A*exp(-B*x) when A>0: use (A+dA)*exp(-(B-dB)*x)
                term_max = (A + dA) * np.exp(-(B - dB) * x_term)
                
                f_min = (C - dC) + term_min
                f_max = (C + dC) + term_max
                
            else:
                # A is negative
                # Lower bound: minimize C term, minimize A*exp(-B*x) term (most negative)
                # When A<0, A*exp(-B*x) is negative
                # Most negative: (A-dA)*exp(-(B-dB)*x) - smaller B gives larger exp, more negative
                term_min = (A - dA) * np.exp(-(B - dB) * x_term)
                # Upper bound: maximize C term, maximize A*exp(-B*x) term (least negative)
                # Least negative: (A+dA)*exp(-(B+dB)*x) - larger B gives smaller exp, less negative
                term_max = (A + dA) * np.exp(-(B + dB) * x_term)
                
                f_min = (C - dC) + term_min
                f_max = (C + dC) + term_max
            
            # Ensure f_min <= f_max (numerical safety)
            if f_min > f_max:
                f_min, f_max = f_max, f_min
                
        except (OverflowError, ValueError) as e:
            # Fallback if calculation fails
            print(f"Warning: Uncertainty calculation failed at x={x_value}: {e}")
            f_min = f_max = C
            
        return f_min, f_max

    def _calculate_extrapolation_uncertainty(self, result, model_type='exp'):
        """
        Calculate uncertainty at the extrapolation limit (x -> infinity).
        At infinity, A*exp(-B*x) -> 0, so the uncertainty is simply ΔC.
        """
        params = result.params
        C = params['C'].value
        dC = params['C'].stderr if params['C'].stderr is not None else 0
        
        return dC
###########################################################################################
    def _fit_with_weights(self, y_data, x_scaled, params, model, weight_power):
        """Helper method to fit with specific weight power."""
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
###########################################################################
        result = _scale_covariance_with_weights(result, current_weights)
        return result, current_weights

    def _optimize_weights_and_fit(self, y_data, x_scaled, params, model, model_name):
        """Optimize weights and fit the model."""
        best_result = None
        best_distance = np.inf
        best_weights = None
        best_n = None

        weight_powers = [1,2]

        if self.known_convergent_value is not None:
            print(f"Optimizing {model_name} weights using known value: {self.known_convergent_value:.8f}")

            for n in weight_powers:
                try:
                    temp_result, temp_weights = self._fit_with_weights(
                        y_data, x_scaled, params.copy(), model, n
                    )

                    # Calculate distance to known value
                    extrapolated_limit = temp_result.params['C'].value
                    distance = abs(extrapolated_limit - self.known_convergent_value)

                    if distance < best_distance:
                        best_distance = distance
                        best_result = temp_result
                        best_weights = temp_weights
                        best_n = n

                except Exception as e:
                    continue

            if best_result is not None:
                print(f"  Best weight power: {best_n} with distance {best_distance:.2e}")
                result = best_result
                current_weights = best_weights
            else:
                print(f"  Weight optimization failed, using default weights (n=1)")
                result, current_weights = self._fit_with_weights(
                    y_data, x_scaled, params.copy(), model, 1
                )
        else:
            # No known value, use default weight power
            print(f"  No known convergent value provided, using weight power n=1 for {model_name}")
            result, current_weights = self._fit_with_weights(
                y_data, x_scaled, params.copy(), model, 1
            )

        return result

    def _fit_model(self, column_name, max_x, model_func, model_name, model_type):
        """Fit a specific model to the data."""
        print(f"\n--- Fitting {model_name} Model ---")

        y_data = self.df[column_name].values

        # --- Extended: Check multiple initial points for misleading values ---

        ######################## Early data points!
        y_data = np.asarray(y_data, dtype=float)
        x_data = np.asarray(self.x_data.values)

        if len(y_data) < 4:
            x_trimmed, y_trimmed = x_data, y_data
        else:
            # 1. Detect global trend direction
            slope = np.polyfit(np.arange(len(y_data)), y_data, 1)[0]
            trend = "decreasing" if slope < 0 else "increasing"

            # 2. Identify the key point (max for decreasing, min for increasing)
            if trend == "decreasing":
                key_idx = np.argmax(y_data)
            else:
                key_idx = np.argmin(y_data)

            # 3. Trim only if the key point is not at the start
            if key_idx > 0:
                x_trimmed = x_data[key_idx:]
                y_trimmed = y_data[key_idx:]
                print(f"Removed {key_idx} early inconsistent point(s) for '{column_name}'")
            else:
                x_trimmed, y_trimmed = x_data, y_data

        x_min_orig, x_max_orig = x_trimmed.min(), x_trimmed.max()

        if model_type == 'power':
            x_scaled = x_trimmed / x_max_orig
        else:
            x_scaled = (x_trimmed - x_min_orig) / (x_max_orig - x_min_orig)

        model = Model(model_func)
        params = model.make_params()

        # Initial parameter guesses using trimmed data
        y_last, y_first = y_trimmed[-1], y_trimmed[0]
        params['C'].set(value=y_last)

        ##############################################################
        
        if model_type == 'power':
            ### START: CALLING power_fit CLASS ###
            # 1. Create an instance of the power_fit class
            power_fitter = power_fit(self.df)

            # 2. Pass necessary parameters from the main class to the instance
            power_fitter.known_convergent_value = self.known_convergent_value
            power_fitter.known_convergent_uncertainty = self.known_convergent_uncertainty

            # 3. Run the internal fitting method of the power_fit class
            power_fitter._fit_column(column_name, max_x)

            # 4. Retrieve the result and return it in the expected format
            result = power_fitter.result
            uncertainty = power_fitter.total_uncertainty if power_fitter.total_uncertainty is not None else 0
            
            return result, uncertainty
        else:
            params['A'].set(value=y_first - y_last)
            if y_last < y_first:
                params['A'].set(min=0.0)
            else:
                params['A'].set(max=0.0)

            try:
                half_life_y = y_last + (y_first - y_last) / 2.0
                half_life_x = x_scaled[np.argmin(np.abs(y_trimmed - half_life_y))]
                if model_type == 'exp_sq':
                    params['B'].set(value=np.log(2) / np.sqrt(half_life_x) if half_life_x > 1e-6 else 0.1, min=1e-6)
                else:
                    params['B'].set(value=np.log(2) / half_life_x if half_life_x > 1e-6 else 0.1, min=1e-6)
            except:
                params['B'].set(value=0.1, min=1e-6)

        # Optimize weights and fit using trimmed data
        result = self._optimize_weights_and_fit(y_trimmed, x_scaled, params, model, model_name)

        # Calculate uncertainty at extrapolation limit
        uncertainty = self._calculate_extrapolation_uncertainty(result, model_type)
        
        # Print parameter signs for verification
        C = result.params['C'].value
        A = result.params['A'].value
        B = result.params['B'].value
        print(f"Parameter signs: C={'positive' if C>=0 else 'negative'}, "
              f"A={'positive' if A>=0 else 'negative'}, "
              f"B={'positive' if B>=0 else 'negative'}")

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
            # Use power_fit method for power law
            x_scaled = x_values_orig / x_max_orig
            for x_val in x_scaled:
                # Need to create a temporary power_fit instance with the result
                # Or call the calculation directly
                params = result.params
                
                C = params['C'].value
                A = params['A'].value
                B = params['B'].value
                
                dC = params['C'].stderr if params['C'].stderr is not None else 0
                dA = params['A'].stderr if params['A'].stderr is not None else 0
                dB = params['B'].stderr if params['B'].stderr is not None else 0
                
                try:
                    if A >= 0:
                        term_min = (A - dA) / np.power(x_val, B + dB)
                        term_max = (A + dA) / np.power(x_val, B - dB)
                    else:
                        term_min = (A - dA) / np.power(x_val, B - dB)
                        term_max = (A + dA) / np.power(x_val, B + dB)
                    
                    f_min = (C - dC) + term_min
                    f_max = (C + dC) + term_max
                    
                    if f_min > f_max:
                        f_min, f_max = f_max, f_min
                except:
                    f_min = f_max = C
                
                f_min_curve.append(f_min)
                f_max_curve.append(f_max)
        else:
            # Use exponential method
            model_type = 'exp_sq' if model_key == 'exponential_sq' else 'exp'
            x_scaled = (x_values_orig - x_min_orig) / (x_max_orig - x_min_orig)
            
            for x_val in x_scaled:
                f_min, f_max = self._calculate_exponential_uncertainty_at_x(result, x_val, model_type)
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

        # Calculate and plot uncertainty band
        f_min_curve, f_max_curve = self.get_uncertainty_band(model_key, plot_x_orig)
        if f_min_curve is not None:
            ax.fill_between(plot_x_orig, f_min_curve, f_max_curve, 
                            color='blue', alpha=0.2, label='Uncertainty Band')

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

        # Calculate and plot uncertainty band
        f_min_curve, f_max_curve = self.get_uncertainty_band(model_key, plot_x_orig)
        if f_min_curve is not None:
            ax.fill_between(plot_x_orig, f_min_curve, f_max_curve, 
                            color='blue', alpha=0.2, label='Uncertainty Band')

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


class power_fit:
    """
    A class to fit a power law model using Self-Consistent 
    Iteratively Reweighted Least Squares (IRLS).
    """
    def __init__(self, dataframe):
        if 'basis size' not in dataframe.columns:
            raise ValueError("Input DataFrame must contain a 'basis size' column.")
        self.df = dataframe
        self.x_data = self.df['basis size']
        self.result = None
        self.total_uncertainty = None
        self.known_convergent_value = None
        self.known_convergent_uncertainty = None

    def _calculate_power_law_uncertainty_at_x(self, x_value):
        """
        Calculate uncertainty bounds for power law at a specific x value.
        f(x) = C + A / x^B
        
        Handles all cases:
        - A > 0, C > 0
        - A > 0, C < 0
        - A < 0, C > 0
        - A < 0, C < 0
        
        B is always > 0 for power law.
        
        Returns (f_min, f_max) at the given x value.
        """
        params = self.result.params
        
        C = params['C'].value
        A = params['A'].value
        B = params['B'].value
        
        dC = params['C'].stderr if params['C'].stderr is not None else 0
        dA = params['A'].stderr if params['A'].stderr is not None else 0
        dB = params['B'].stderr if params['B'].stderr is not None else 0
        
        try:
            # Calculate the two candidate bounds
            # For A/x^B term:
            # - When we want minimum: use (A-dA) with largest exponent (B+dB) if A>0
            #                         use (A+dA) with smallest exponent (B-dB) if A<0
            # - When we want maximum: opposite
            
            if A >= 0:
                # A is positive or zero
                # Lower bound: minimize C term, minimize A/x^B term
                term_min = (A - dA) / np.power(x_value, B + dB)
                # Upper bound: maximize C term, maximize A/x^B term
                term_max = (A + dA) / np.power(x_value, B - dB)
                
                f_min = (C - dC) + term_min
                f_max = (C + dC) + term_max
                
            else:
                # A is negative
                # Lower bound: minimize C term, minimize A/x^B term (most negative)
                # When A<0, A/x^B is negative, so we want the most negative value
                # Most negative: (A-dA) with smallest exponent (B-dB)
                term_min = (A - dA) / np.power(x_value, B - dB)
                # Upper bound: maximize C term, maximize A/x^B term (least negative)
                # Least negative: (A+dA) with largest exponent (B+dB)
                term_max = (A + dA) / np.power(x_value, B + dB)
                
                f_min = (C - dC) + term_min
                f_max = (C + dC) + term_max
            
            # Ensure f_min <= f_max (numerical safety)
            if f_min > f_max:
                f_min, f_max = f_max, f_min
                
        except (ZeroDivisionError, OverflowError, ValueError) as e:
            # Fallback if calculation fails
            print(f"Warning: Uncertainty calculation failed at x={x_value}: {e}")
            f_min = f_max = C
            
        return f_min, f_max

    def _calculate_extrapolation_uncertainty(self):
        """
        Calculate uncertainty at the extrapolation limit (x -> infinity).
        At infinity, A/x^B -> 0, so the uncertainty is simply ΔC.
        """
        params = self.result.params
        C = params['C'].value
        dC = params['C'].stderr if params['C'].stderr is not None else 0
        
        return dC

    def _fit_column(self, column_name, max_x):
        """
        Performs a self-consistent, data-driven power law fit using IRLS.
        """
        y_data = self.df[column_name].values
        
        # Trim early inconsistent data points
        y_data = np.asarray(y_data, dtype=float)
        x_data = np.asarray(self.x_data.values)

        if len(y_data) < 4:
            x_trimmed, y_trimmed = x_data, y_data
        else:
            slope = np.polyfit(np.arange(len(y_data)), y_data, 1)[0]
            trend = "decreasing" if slope < 0 else "increasing"
            
            if trend == "decreasing":
                key_idx = np.argmax(y_data)
            else:
                key_idx = np.argmin(y_data)
            
            if key_idx > 0:
                x_trimmed = x_data[key_idx:]
                y_trimmed = y_data[key_idx:]
                print(f"Removed {key_idx} early inconsistent point(s) for '{column_name}'")
            else:
                x_trimmed, y_trimmed = x_data, y_data

        # Scaling and model setup
        x_min_orig, x_max_orig = x_trimmed.min(), x_trimmed.max()
        x_scaled = x_trimmed / x_max_orig
        power_model = Model(power_law)
        params = power_model.make_params()

        # Initial parameter guesses
        y_last, y_first = y_trimmed[-1], y_trimmed[0]
        params['C'].set(value=y_last)
        B0 = _init_B(x_scaled, y_trimmed)
        params['B'].set(value=B0, min=1e-6, max=5)

        amplitude_guess = y_first - y_last
        amplitude_bound = abs(amplitude_guess)

        if y_last < y_first:
            params['A'].set(value=amplitude_guess, min=1e-9, max=amplitude_bound)
        else:
            params['A'].set(value=amplitude_guess, min=-amplitude_bound, max=-1e-9)

        # Weight optimization if known value provided
        best_result = None
        best_distance = np.inf
        weight_powers = [1,2]

        if self.known_convergent_value is not None:
            print(f"Optimizing weights using known value: {self.known_convergent_value:.8f}")
            
            for n in weight_powers:
                try:
                    temp_result, temp_weights = self._fit_with_weights(
                        y_trimmed, x_scaled, params.copy(), power_model, n
                    )
                    
                    extrapolated_limit = temp_result.params['C'].value
                    distance = abs(extrapolated_limit - self.known_convergent_value)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_result = temp_result
                except Exception as e:
                    continue

            if best_result is not None:
                result = best_result
            else:
                result, _ = self._fit_with_weights(
                    y_trimmed, x_scaled, params.copy(), power_model, 1
                )
        else:
            result, _ = self._fit_with_weights(
                y_trimmed, x_scaled, params.copy(), power_model, 1
            )

        self.result = result
        
        # Calculate uncertainty using parameter uncertainty (ΔC)
        self.total_uncertainty = self._calculate_extrapolation_uncertainty()
        
        # Print parameter signs for verification
        C = self.result.params['C'].value
        A = self.result.params['A'].value
        B = self.result.params['B'].value
        print(f"\nParameter signs: C={'positive' if C>=0 else 'negative'}, "
              f"A={'positive' if A>=0 else 'negative'}, "
              f"B={'positive' if B>=0 else 'negative'}")
        
        return result

    def _fit_with_weights(self, y_data, x_scaled, params, model, weight_power):
        """Helper method to fit with specific weight power."""
        n_iterations = 100
        convergence_threshold = 1e-8
        current_weights = np.ones(len(x_scaled))

        for i in range(n_iterations):
            result = model.fit(y_data, params, x=x_scaled, weights=current_weights)
            
            W_pos = x_scaled ** params['B'].value
            W_pos /= np.mean(W_pos)
            
            new_weights = W_pos ** weight_power
            new_weights /= np.mean(new_weights)
            
            current_weights = 0.5 * current_weights + 0.5 * new_weights
            
            old_params = np.array(list(params.valuesdict().values()))
            params = result.params
            new_params = np.array(list(params.valuesdict().values()))
            param_change = np.sum((old_params - new_params)**2)
            
            if param_change < convergence_threshold and i > 0:
                break
        
        result = _scale_covariance_with_weights(result, current_weights)
        return result, current_weights
    
    def get_uncertainty_band(self, x_values, x_max_orig):
        """
        Get uncertainty band for plotting.
        
        Args:
            x_values: Original (unscaled) x values
            x_max_orig: Maximum x value from training data (for scaling)
            
        Returns:
            f_min_curve, f_max_curve: Arrays of lower and upper bounds
        """
        x_scaled = x_values / x_max_orig
        
        f_min_curve = []
        f_max_curve = []
        
        for x_val in x_scaled:
            f_min, f_max = self._calculate_power_law_uncertainty_at_x(x_val)
            f_min_curve.append(f_min)
            f_max_curve.append(f_max)
        
        return np.array(f_min_curve), np.array(f_max_curve)


