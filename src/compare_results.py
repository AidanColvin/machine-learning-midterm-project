import pandas as pd
import glob

def compare_results() -> None:
    """
    given nothing
    read all model result CSVs from data/preprocessed
    return nothing — prints and saves a sorted model comparison
    """
    files = glob.glob('data/preprocessed/*_results.csv')
    if not files:
        print("No result files found. Run the training scripts first.")
        return

    comparison_df = pd.concat(
        [pd.read_csv(f) for f in files], ignore_index=True
    ).sort_values(by='Accuracy', ascending=False)

    comparison_df.to_csv('data/preprocessed/model_comparison.csv', index=False)

    print("")
    print("  ── Model Comparison ────────────────────────────")
    print(f"     {'Model':<35} {'Accuracy':>10}")
    print(f"     {'─────':<35} {'────────':>10}")
    for _, row in comparison_df.iterrows():
        bar   = "█" * int(row['Accuracy'] * 30)
        print(f"     {row['Model']:<35} {row['Accuracy']:>10.4f}  {bar}")
    print("")
    best = comparison_df.iloc[0]
    print(f"  ✔  Best model: {best['Model']} ({best['Accuracy']:.4f} accuracy)")
    print(f"  ── Saved to  : data/preprocessed/model_comparison.csv")
    print("")

if __name__ == "__main__":
    compare_results()
