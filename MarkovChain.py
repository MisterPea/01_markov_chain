import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

class MarkovChainModel:
    """
    Class for implementing Markov Chain Model
    """
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = self.fetch_data()
        self.states = None
        self.transition_matrix = None

    def fetch_data(self):
        """
        Fetch historical data from yfinance
        """
        data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
        data["Returns"] = data["Close"].pct_change()  # Daily returns
        data["Vol"] = data["Volume"].pct_change()
        data.dropna(inplace=True)  # Drop missing data
        return data

    def classify_states(self, threshold=0.0005):
        # 0.05% change needed to be a state change
        """
        Classify price movements into 3 states:
        - State 0: Price decreases
        - State 1: Price stays the same (within threshold)
        - State 2: Price increases
        """
        self.data["State"] = np.where(
            self.data["Returns"] > threshold,
            2,
            np.where(self.data["Returns"] < -threshold, 0, 1),
        )
        self.states = self.data["State"].values

    def calculate_transition_matrix(self):
        """
        Calculate transition matrix based upon classified states
        """
        num_states = 3
        # Create a num_states x num_states matrix
        transition_matrix = np.zeros((num_states, num_states))
        for prev_state, next_state in zip(self.states[:-1], self.states[1:]):
            transition_matrix[prev_state][next_state] += 1
        # Normalize the matrix to get probabilities
        transition_matrix = transition_matrix / transition_matrix.sum(axis=1)[:, None]
        self.transition_matrix = transition_matrix

    def simulate(self, num_steps=100):
        """
        Simulate future price movements using the Markov Chain
        """
        current_state = self.states[-1]
        states = [current_state]

        for _ in range(num_steps):
            next_state = np.random.choice(
                [0, 1, 2], p=self.transition_matrix[current_state]
            )
            states.append(next_state)
            current_state = next_state

        # Convert list to ints
        py_int_list = [int(s) for s in states]
        return py_int_list

    def print_matrix_heatmap(self):
        """Print heatmap of transition matrix"""
        plt.figure(figsize=(4, 4))
        transition_matrix = self.transition_matrix * 100
        sns.heatmap(
            transition_matrix,
            annot=True,
            cmap="YlGn",
            fmt=".2f",
            xticklabels=["Down", "Stable", "Up"],
            yticklabels=["Down", "Stable", "Up"],
            annot_kws={"fontsize": 10},
            linewidths=0.25,
            linecolor="black",
            cbar=False,
        )
        # Add '%' to the end of the numbers
        for text in plt.gca().texts:
            text.set_text(text.get_text() + "%")

        plt.title(f"""Transition Matrix Heatmap for {self.symbol}
{self.start_date} to {self.end_date}""",
            fontsize=10,
        )
        plt.xlabel("Next State")
        plt.ylabel("Current State")
        plt.show()


model = MarkovChainModel(symbol="SPY", start_date="2020-01-01", end_date="2023-01-01")
model.classify_states()
model.calculate_transition_matrix()
model.print_matrix_heatmap()
# simulated_states = model.simulate(num_steps=100)
# print(simulated_states)
