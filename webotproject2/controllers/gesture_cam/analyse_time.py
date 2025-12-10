"""
analyse_experiment.py

Used for analysing experiment data:
- results_time_led.csv     (task completion time)
- results_trials_led.csv   (time + collisions + parking success)

Run this script in the directory where the CSV files are located:
    D:/python/python.exe analyse_experiment.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


# ========= Path Settings =========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TIME_CSV = os.path.join(BASE_DIR, "results_time_led.csv")
TRIAL_CSV = os.path.join(BASE_DIR, "results_trials_led.csv")


# ============================================================
#  Task Time Analysis
# ============================================================
def analyse_time():
    """Analyse task completion time for each control mode and return summary."""
    if not os.path.exists(TIME_CSV):
        print(f"[Time] File not found: {TIME_CSV}, skip time analysis.")
        return None

    print("\n========== Task Time Analysis (results_time_led.csv) ==========")
    df = pd.read_csv(TIME_CSV)

    expected_cols = {"participant", "mode", "trial", "duration_sec"}
    if not expected_cols.issubset(df.columns):
        print("[Time] Unexpected column names:", df.columns.tolist())
        return None

    summary = (
        df.groupby("mode")["duration_sec"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )

    print("\n[Time] Summary by control mode:")
    print(
        summary.to_string(
            index=False,
            formatters={"mean": "{:.3f}".format, "std": "{:.3f}".format}
        )
    )

    return summary



# ============================================================
#  Collision Count & Parking Success Analysis
# ============================================================
def analyse_trials():
    """Return average collision count and parking success rate."""
    if not os.path.exists(TRIAL_CSV):
        print(f"[Trials] File not found: {TRIAL_CSV}, skip analysis.")
        return None

    print("\n========== Collision & Parking Success Analysis (results_trials_led.csv) ==========")
    df = pd.read_csv(TRIAL_CSV)

    expected_cols = {
        "participant", "mode", "trial",
        "duration_sec", "collision", "parking"
    }
    if not expected_cols.issubset(df.columns):
        print("[Trials] Unexpected column names:", df.columns.tolist())
        return None

    df["collision"] = df["collision"].astype(float)
    df["parking"] = df["parking"].astype(float)

    summary = (
        df.groupby("mode")[["collision", "parking"]]
        .mean()
        .reset_index()
    )

    print("\n[Trials] Summary by control mode:")
    display = summary.copy()
    display["collision"] = display["collision"].apply(lambda x: f"{x:.2f} collisions/trial")
    display["parking"] = display["parking"].apply(lambda x: f"{x:.2%}")
    print(display.to_string(index=False))

    return summary



# ============================================================
#   Combined Plot: Time / Collision / Parking Success Rate
# ============================================================
def plot_combined_figure(time_summary, trial_summary):
    """Plot three bar charts in one figure: task time, collision count, parking success rate."""

    # ======= Time Data =======
    modes_time = time_summary["mode"].tolist()
    means = time_summary["mean"].tolist()
    stds = time_summary["std"].tolist()

    # ======= Trial Data =======
    modes_trial = trial_summary["mode"].tolist()
    avg_collision = trial_summary["collision"].tolist()
    parking_rates = trial_summary["parking"].tolist()

    # ======= Create 1×3 Subplots =======
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # ---------------------- Subplot 1: Task Time ----------------------
    ax1 = axes[0]
    x1 = range(len(modes_time))
    bars1 = ax1.bar(x1, means, yerr=stds, capsize=5)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(modes_time)
    ax1.set_title("Average Task Time")
    ax1.set_ylabel("Time (s)")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for i, bar in enumerate(bars1):
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2,
            h + 0.05,
            f"{h:.2f}s",
            ha="center",
            va="bottom"
        )

    # ---------------------- Subplot 2: Average Collision Count ----------------------
    ax2 = axes[1]
    x2 = range(len(modes_trial))
    bars2 = ax2.bar(x2, avg_collision)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(modes_trial)
    ax2.set_title("Average Collision Count")
    ax2.set_ylabel("Collision Count")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for i, bar in enumerate(bars2):
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            h + 0.01,
            f"{h:.2f}",
            ha="center",
            va="bottom"
        )

    # ---------------------- Subplot 3: Parking Success Rate ----------------------
    ax3 = axes[2]
    x3 = range(len(modes_trial))
    bars3 = ax3.bar(x3, parking_rates)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(modes_trial)
    ax3.set_title("Parking Success Rate")
    ax3.set_ylabel("Success Rate")
    ax3.set_ylim(0, 1.0)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    for i, bar in enumerate(bars3):
        h = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width()/2,
            h + 0.01,
            f"{h:.2%}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.show()



# ============================================================
#   Main Function
# ============================================================
def main():
    time_summary = analyse_time()
    trial_summary = analyse_trials()

    # Plot only when both summaries are valid
    if time_summary is not None and trial_summary is not None:
        plot_combined_figure(time_summary, trial_summary)


if __name__ == "__main__":
    main()
