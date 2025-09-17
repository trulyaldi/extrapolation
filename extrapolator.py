
import matplotlib.patches as patches
from lmfit import Model
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define the Mathematical Model Functions ---
def exponential_decay(x, C, A, B):
    """A 3-parameter exponential decay function."""
    x_safe = np.maximum(x, 1e-10)
    return C + A * np.exp(-B * x_safe)

def exponential_decay_sq(x, C, A, B):
    """A 3-parameter exponential decay function with square root."""
    return C + A * np.exp(-B * (x ** (0.5)))

def power_law(x, C, A, B):
    """A power law function."""
    return C + A * np.power(x, -B)


def init_B_half_life(x, y, C=None):
    """
    Estimate initial guess for B in y = C + A * x^(-B)
    using a half-life analogy.
    """
    x, y = np.asarray(x), np.asarray(y)

    if C is None:
        C = y[-1]  # stable asymptote guess

    x0, y0 = x[0], y[0]
    half_val = C + 0.5 * (y0 - C)

    idx = np.argmin(np.abs(y - half_val))
    x_half = x[idx]

    if x_half <= x0 or (y0 - C) <= 0:
        return 0.1  # fallback

    B_est = np.log(2.0) / np.log(x_half / x0)
    return float(np.clip(B_est, 1e-6, 5.0))

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

        return result, current_weights

    def _optimize_weights_and_fit(self, y_data, x_scaled, params, model, model_name):
        """Optimize weights and fit the model."""
        best_result = None
        best_distance = np.inf
        best_weights = None
        best_n = None

        weight_powers = [1,2,3,4,5]

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

    def _calculate_monte_carlo_uncertainty(self, result, model_type):
        """Calculate Monte Carlo uncertainty for the fitted model."""
        C_val = result.params['C'].value
        A_val = result.params['A'].value
        B_val = result.params['B'].value
        c_stderr = result.params['C'].stderr or 0.0
        a_stderr = result.params['A'].stderr or 0.0
        b_stderr = result.params['B'].stderr or 0.0

        # Monte Carlo uncertainty calculation
        n_samples = 5000

        # For finite extrapolation point
        x_extrap_actual = self.max_x
        if model_type == 'power':
            x_scaled_extrap = x_extrap_actual / self.x_data.max()
        else:
            x_scaled_extrap = (x_extrap_actual - self.x_data.min()) / (self.x_data.max() - self.x_data.min())

        # Generate Monte Carlo samples and evaluate model
        if (c_stderr > 0) and (a_stderr > 0) and (b_stderr > 0):
            try:
                # Sample parameters using multivariate normal
                param_vals = [C_val, A_val, B_val]
                param_errs = [c_stderr, a_stderr, b_stderr]

                # Simple diagonal covariance (uncorrelated approximation)
                samples = np.random.multivariate_normal(param_vals,
                                                      np.diag([err**2 for err in param_errs]),
                                                      n_samples)

                # Evaluate model at extrapolation point for each sample
                extrapolated_samples = []
                for sample in samples:
                    C_s, A_s, B_s = sample
                    if model_type == 'power':
                        # Power law evaluation: C + A * x^(-B)
                        value = C_s + A_s * (x_scaled_extrap**(-B_s))
                    elif model_type == 'exp_sq':
                        # Square root exponential: C + A * exp(-B * sqrt(x))
                        value = C_s + A_s * np.exp(-B_s * np.sqrt(x_scaled_extrap))
                    else:
                        # Regular exponential: C + A * exp(-B * x)
                        value = C_s + A_s * np.exp(-B_s * x_scaled_extrap)
                    extrapolated_samples.append(value)

                # Calculate Monte Carlo uncertainty
                mc_uncertainty = np.std(extrapolated_samples)

            except:
                # Fallback to simple C parameter uncertainty
                mc_uncertainty = c_stderr
        else:
            # Fallback to simple C parameter uncertainty
            mc_uncertainty = c_stderr

        return mc_uncertainty

    def _fit_model(self, column_name, max_x, model_func, model_name, model_type):
        """Fit a specific model to the data."""
        print(f"\n--- Fitting {model_name} Model ---")

        y_data = self.df[column_name].values

        # --- Extended: Check multiple initial points for misleading values ---
        if len(y_data) >= 3:
            # Check first few points (up to 3) for misleading values
            points_to_check = min(3, len(y_data) - 1)  # Don't check all points
            bad_points = []

            for i in range(points_to_check):
                # If point is significantly lower than last point, mark as bad
                if y_data[i] < y_data[-1] - 0.5 * (y_data[-1] - np.min(y_data[1:])):
                    bad_points.append(i)
                else:
                    # Stop at first "good" point
                    break

            # Remove bad points but keep at least 2 points
            if bad_points and len(bad_points) < len(y_data) - 1:
                keep_indices = [i for i in range(len(y_data)) if i not in bad_points]
                x_trimmed = self.x_data.values[keep_indices]
                y_trimmed = y_data[keep_indices]
                print(f"Removed {len(bad_points)} early outlier(s) for '{column_name}'")
            else:
                x_trimmed = self.x_data.values
                y_trimmed = y_data
        else:
            x_trimmed = self.x_data.values
            y_trimmed = y_data

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

        if model_type == 'power':
            # Use half-life initializer for B
            try:
                B_init = init_B_half_life(x_trimmed, y_trimmed, C=y_last)
            except Exception:
                B_init = 2  # fallback if init fails

            params['B'].set(value=B_init, min=1e-6, max=10.0)

            # Amplitude guess
            amplitude_guess = y_first - y_last
            amplitude_bound = abs(amplitude_guess)

            if y_last < y_first:
                params['A'].set(value=amplitude_guess, min=1e-9, max=amplitude_bound)
            else:
                params['A'].set(value=amplitude_guess, min=-amplitude_bound, max=-1e-9)
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

        # Calculate uncertainty
        mc_uncertainty = self._calculate_monte_carlo_uncertainty(result, model_type)

        return result, mc_uncertainty

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
                      color='red', alpha=0.15, label='Total Uncertainty Band')

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
                      color='red', alpha=0.15, label='Total Uncertainty Band')

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

        column_name = input("Please enter the name of the column to fit (or 'q' to quit): ")
        if column_name.lower() in ['q', 'quit']:
            print("Exiting.")
            return

        if column_name not in available_columns:
            print(f"Error: Invalid column name '{column_name}'. Please choose from the list above.")
            return

        try:
            max_x_val = int(input(f"Enter the extrapolation limit for '{column_name}' (e.g., 20000): "))
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
