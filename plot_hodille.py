import pandas as pd
import matplotlib.pyplot as plt


def plot_hodille_data(series="0001"):

    # --- Manually parse the two-row header ---
    with open("hodille_2022_data.csv") as f:
        row0 = f.readline().rstrip("\n").split(",")
        row1 = f.readline().rstrip("\n").split(",")

    # Forward-fill the dataset name across its two columns (X and Y)
    for i in range(1, len(row0)):
        if row0[i] == "":
            row0[i] = row0[i - 1]

    # Build flat column names: "750_0001_X", "750_0001_Y", etc.
    # zip stops at shortest, which drops any ghost column from the trailing comma
    col_names = [f"{dataset}_{xy}" for dataset, xy in zip(row0, row1) if xy != ""]

    # --- Read data (skip the 2 header rows) ---
    df = pd.read_csv(
        "hodille_2022_data.csv",
        skiprows=2,
        header=None,
        names=col_names,
        usecols=range(len(col_names)),
        dtype=float,
    )

    # Drop any phantom column from the trailing comma
    df = df.dropna(axis=1, how="all")
    print(df.columns.tolist())
    # --- Plot ---
    datasets = {
        f"750_{series}": "750 K",
        f"1000_{series}": "1000 K",
        f"1250_{series}": "1250 K",
    }

    for col_name, label in datasets.items():
        x = df[f"{col_name}_X"]  # .dropna().values
        y = df[f"{col_name}_Y"]  # .dropna().values

        # Sort by x for a clean line
        order = x.argsort()
        plt.scatter(x, y, marker="o", label=label, alpha=0.5)


if __name__ == "__main__":
    plot_hodille_data(series="0001b")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(title="Temperature")
    plt.tight_layout()
    plt.show()
