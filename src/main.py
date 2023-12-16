# main.py
from perceptron import Perceptron
from perceptron_gradient_descent import PerceptronGradientDescent
import pandas as pd
from plotting import plot_mints

def main(runs=1, perceptron_type=0):
    # Specify the file path
    file_path = r'seperable_data_2d.csv'

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    total_guesses = 0

    for _ in range(runs):
        if perceptron_type == 0:
            pl = Perceptron() # Average guesses over 1000 runs: 376.916
        elif perceptron_type == 1:
            pl = PerceptronGradientDescent() # Same 121 epochs every run
        pl.fit(df[['x1', 'x2']].to_numpy()[:80], df['y'].to_numpy()[:80])
        total_guesses += pl.counter if hasattr(pl, 'counter') else pl.epochs

    average_guesses = total_guesses / runs
    print(f"Average guesses over {runs} runs: {average_guesses}")

    # Use the last trained perceptron for plotting
    plot_mints(df, pl)

if __name__ == "__main__":
    main(runs=1, perceptron_type=1)
