# plotting.py
import matplotlib.pyplot as plt
import numpy as np

def plot_mints(mints, pl, show_plots=True):
    if not show_plots:
        return

    # Scatter plot
    mints.plot(
        kind='scatter',
        x='x1',
        y='x2',
        s=32,
        alpha=0.8,
        color=mints['y'].map({0: 'red', 1: 'blue'})
    )

    # Set fixed limits for the x and y axes
    x1_min, x1_max = mints['x1'].min() - 0.1, mints['x1'].max() + 0.1
    x2_min, x2_max = mints['x2'].min() - 0.1, mints['x2'].max() + 0.1
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)

    # Clear previous scatter plot lines
    current_axes = plt.gca()
    scatter_lines = [line for line in current_axes.lines if line.get_label() == 'scatter']
    for line in scatter_lines:
        line.remove()

    # Plot the decision boundary if weights are available
    if hasattr(pl, 'weights') and pl.weights is not None:
        if hasattr(pl, 'offset') and pl.offset is not None:
            x_vals = np.linspace(x1_min, x1_max, 100)
            y_vals = (-pl.weights[0] / pl.weights[1]) * x_vals - pl.offset / pl.weights[1]
            plt.plot(x_vals, y_vals, color='green', linestyle='--', linewidth=2)
        elif hasattr(pl, 'bias') and pl.bias is not None:
            x_vals = np.linspace(x1_min, x1_max, 100)
            y_vals = (-pl.weights[0] / pl.weights[1]) * x_vals - pl.bias / pl.weights[1]
            plt.plot(x_vals, y_vals, color='green', linestyle='--', linewidth=2)

    # Show the plot
    plt.show()
    plt.pause(0.1)
