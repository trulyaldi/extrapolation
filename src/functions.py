import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def graph(df: pd.DataFrame, n_cols: int = 4):
    df_plot = df.copy()
    df_plot.columns = [col.lower() for col in df_plot.columns]

    if 'basis size' not in df_plot.columns:
        raise ValueError("Input DataFrame must contain a 'basis size' column.")

    df_plot['basis size'] = df_plot['basis size'].astype(int)
    features = sorted([col for col in df_plot.columns if col != 'basis size'])
    n_features = len(features)

    if n_features == 0:
        return

    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7 * n_cols, 5.5 * n_rows)
    )
    
    fig.suptitle(r'Be($^3D^e$)', fontsize=22, y=1.05, fontweight='bold')

    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Define the data points and the limit
    basis_sizes = df_plot['basis size'].values
    # This is the x-value of the point just before the last one
    x_limit = basis_sizes[-2] 

    for i, feature in enumerate(features):
        ax = axes[i]
        y = df_plot[feature].values

        # Plot points up until the second to last
        ax.scatter(basis_sizes[:-1], y[:-1], marker='o', s=50, edgecolors='royalblue', alpha=0.8)

        # Horizontal dashed line at the very last value
        ax.axhline(y[-1], color='red', linestyle='--', linewidth=2, alpha=0.9)

        ax.set_title(feature, fontsize=15, pad=10)
        ax.set_xlabel('Basis Size', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        
        # --- KEY FIX: SET X-AXIS LIMIT ---
        # This cuts the plot off right at the last scatter point
        ax.set_xlim(left=basis_sizes[0], right=x_limit)
        
        # Clean up ticks so they don't overlap
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
        ax.ticklabel_format(style='plain', axis='x')
        ax.grid(True, linestyle='--', alpha=0.3)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=3.0)
    plt.show()