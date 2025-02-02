import matplotlib.pyplot as plt


def plot_portfolio(sorted_assets: list, sorted_weights: list) -> None:
    """
    Plot the weights of an optimized portfolio.

    Args:
        sorted_assets (list): List of asset names.
        sorted_weights (list): List of corresponding weights.
    """
    plt.bar(sorted_assets, sorted_weights)
    plt.title("Optimal Portfolio Weights")
    plt.ylabel("Weight")
    plt.xlabel("Assets")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
