"""Stage 4: create descriptive EDA tables and figures from cleaned project data."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "cleaned_players.csv"
OUTPUT_DIR = ROOT / "outputs" / "eda"
TARGET = "SpendingSegment"
ORDER = ["Minnow", "Dolphin", "Whale"]


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError("Run `python src/clean_data.py` before running EDA.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT_FILE)
    if TARGET not in data:
        raise ValueError(f"Target column {TARGET} was not found.")

    counts = data[TARGET].value_counts().reindex(ORDER, fill_value=0)
    summary = pd.DataFrame({"players": counts, "share_percent": (counts / len(data) * 100).round(2)})
    summary.to_csv(OUTPUT_DIR / "segment_distribution.csv")

    colors = ["#4C78A8", "#F2A541", "#58A6A6"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(summary.index, summary["players"], color=colors)
    ax.set_title("Players in each spending segment")
    ax.set_xlabel("Recorded spending segment")
    ax.set_ylabel("Number of rows")
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, share in zip(bars, summary["share_percent"]):
        ax.annotate(f"{share:.1f}%", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "segment_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    numeric = [c for c in ["Age", "SessionCount", "AverageSessionLength", "ActiveDays30d",
                           "PlayerLevel30d", "AdsWatched30d", "SocialActions30d", "WinRate"] if c in data]
    if numeric:
        profile = data.groupby(TARGET)[numeric].agg(["median", lambda values: values.quantile(.25), lambda values: values.quantile(.75)])
        profile.columns = [f"{name}_{stat}" for name, stat in profile.columns]
        profile.to_csv(OUTPUT_DIR / "numeric_profile_by_segment.csv")

    categorical = [c for c in ["Device", "GameGenre", "AcquisitionChannel", "GuildMember", "PushNotificationsOptIn"] if c in data]
    for column in categorical:
        table = pd.crosstab(data[column], data[TARGET], normalize="columns").reindex(columns=ORDER, fill_value=0)
        table.to_csv(OUTPUT_DIR / f"{column.lower()}_share_by_segment.csv")

    print("Segment distribution (descriptive; these rows include generated records):")
    print(summary.to_string())
    print(f"\nTables and figure saved under {OUTPUT_DIR.relative_to(ROOT)}/")
    print("Interpret differences as patterns in this generated dataset, not as causal effects.")


if __name__ == "__main__":
    main()
