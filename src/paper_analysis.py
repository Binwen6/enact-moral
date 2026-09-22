from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from config import PROJECT_ROOT


matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "blue": "#3266a8",
    "light_blue": "#91b6de",
    "orange": "#d9772d",
    "green": "#40855f",
    "red": "#b44c4c",
    "gray": "#777777",
}


def load_table(metrics_dir: Path, name: str, columns: set[str]) -> pd.DataFrame:
    path = metrics_dir / name
    table = pd.read_csv(path)
    missing = columns - set(table.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    return table


def validate_metrics(metrics_dir: Path) -> dict[str, int]:
    binding = load_table(
        metrics_dir,
        "table3_relational_binding.csv",
        {"causal_source", "outcome_valence", "additive_prediction", "joint_replacement", "interaction"},
    )
    np.testing.assert_allclose(
        binding["causal_source"] + binding["outcome_valence"],
        binding["additive_prediction"],
        atol=0.011,
    )
    np.testing.assert_allclose(
        binding["joint_replacement"] - binding["additive_prediction"],
        binding["interaction"],
        atol=0.011,
    )
    patients = load_table(
        metrics_dir,
        "table2_patient_intervention.csv",
        {"remove_1d", "remove_2d", "random_2d", "add_1d", "add_2d"},
    )
    if not (patients["remove_2d"].mean() < patients["random_2d"].mean() < patients["add_2d"].mean()):
        raise ValueError("Patient intervention directions violate the reported ordering")
    completion = load_table(
        metrics_dir,
        "figure2_patient_completion.csv",
        {"panel", "condition", "median", "q25", "q75", "metric"},
    )
    if not ((completion["q25"] <= completion["median"]) & (completion["median"] <= completion["q75"])).all():
        raise ValueError("Figure 2 quartiles are unordered")
    scale = load_table(
        metrics_dir,
        "scale_transition.csv",
        {"parameters_billion", "concordance", "human_agreement", "outcome_conditioned_gain", "wrongness_gain"},
    )
    if not scale["parameters_billion"].is_monotonic_increasing:
        raise ValueError("Scale rows must be ordered by parameter count")
    return {
        "figure2_rows": len(completion),
        "table2_models": len(patients),
        "table3_models": len(binding),
        "scale_points": len(scale),
    }


def finish_figure(figure: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(output_dir / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def figure2(metrics_dir: Path, output_dir: Path) -> None:
    table = pd.read_csv(metrics_dir / "figure2_patient_completion.csv")
    accuracy = table[(table["panel"] == "accuracy") & (table["metric"] == "identification_accuracy")]
    geometry = table[table["panel"] == "geometry"]
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.25))
    x_values = np.arange(len(accuracy))
    axes[0].errorbar(
        x_values,
        accuracy["median"],
        yerr=[accuracy["median"] - accuracy["q25"], accuracy["q75"] - accuracy["median"]],
        fmt="o",
        color=COLORS["blue"],
        capsize=3,
    )
    axes[0].axhline(0.5, color=COLORS["gray"], linestyle="--", linewidth=1)
    axes[0].set_xticks(x_values, ["Owner", "Distant", "Self", "Symbol", "No fixed"], rotation=25, ha="right")
    axes[0].set_ylim(0.45, 1.03)
    axes[0].set_ylabel("Identification accuracy")
    axes[0].set_title("Implicit patient identification")

    conditions = ["scene_only", "patient_query", "judgment_query"]
    labels = ["Scene only", "+ patient query", "+ judgment query"]
    colors = [COLORS["light_blue"], COLORS["blue"], "#183b70"]
    centers: list[tuple[float, float]] = []
    for condition, label, color in zip(conditions, labels, colors, strict=True):
        rows = geometry[geometry["condition"] == condition].set_index("metric")
        x_row = rows.loc["relative_magnitude"]
        y_row = rows.loc["direction_cosine"]
        axes[1].errorbar(
            x_row["median"],
            y_row["median"],
            xerr=[[x_row["median"] - x_row["q25"]], [x_row["q75"] - x_row["median"]]],
            yerr=[[y_row["median"] - y_row["q25"]], [y_row["q75"] - y_row["median"]]],
            fmt="o",
            color=color,
            capsize=3,
            label=label,
        )
        centers.append((float(x_row["median"]), float(y_row["median"])))
    for start, end in zip(centers[:-1], centers[1:], strict=True):
        axes[1].annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": COLORS["blue"]})
    axes[1].set_xlim(0, 0.75)
    axes[1].set_ylim(0, 0.85)
    axes[1].set_xlabel("Relative magnitude")
    axes[1].set_ylabel("Direction cosine")
    axes[1].set_title("Alignment with explicit state")
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    finish_figure(figure, output_dir, "figure2_patient_completion")


def figure3(metrics_dir: Path, output_dir: Path) -> None:
    table = pd.read_csv(metrics_dir / "table3_relational_binding.csv")
    x_values = np.arange(len(table))
    width = 0.25
    figure, axis = plt.subplots(figsize=(7.8, 3.4))
    axis.bar(x_values - width, table["causal_source"], width, label="Causal source", color=COLORS["blue"])
    axis.bar(x_values, table["outcome_valence"], width, label="Outcome valence", color=COLORS["orange"])
    axis.bar(x_values + width, table["joint_replacement"], width, label="Joint replacement", color=COLORS["green"])
    axis.scatter(x_values + width, table["additive_prediction"], marker="x", color="black", label="Additive prediction", zorder=4)
    axis.axhline(0, color="black", linewidth=0.7)
    axis.set_xticks(x_values, table["model"], rotation=20, ha="right")
    axis.set_ylabel("Change in intention readout")
    axis.set_title("Non-linear relational binding")
    axis.legend(frameon=False, ncol=2, fontsize=8)
    figure.tight_layout()
    finish_figure(figure, output_dir, "figure3_relational_binding")


def figure4(metrics_dir: Path, output_dir: Path) -> None:
    table = pd.read_csv(metrics_dir / "mediator_knockout.csv")
    x_values = np.arange(len(table))
    width = 0.18
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
    axes[0].bar(x_values, table["attribution_gap"], color=COLORS["blue"])
    axes[0].set_xticks(x_values, table["model"], rotation=25, ha="right")
    axes[0].set_ylabel("Attribution gap")
    axes[0].set_title("Mediator attribution")
    columns = [
        ("controllable_mediator", "Controllable / mediator", COLORS["blue"]),
        ("controllable_random", "Controllable / random", COLORS["light_blue"]),
        ("uncontrollable_mediator", "Uncontrollable / mediator", COLORS["orange"]),
        ("uncontrollable_random", "Uncontrollable / random", "#e8b98f"),
    ]
    for index, (column, label, color) in enumerate(columns):
        axes[1].bar(x_values + (index - 1.5) * width, table[column], width, label=label, color=color)
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set_xticks(x_values, table["model"], rotation=25, ha="right")
    axes[1].set_ylabel("Change in attribution")
    axes[1].set_title("Intermediate-event knockout")
    axes[1].legend(frameon=False, fontsize=7)
    figure.tight_layout()
    finish_figure(figure, output_dir, "figure4_mediator_knockout")


def figure5(metrics_dir: Path, output_dir: Path) -> None:
    interface = pd.read_csv(metrics_dir / "compact_interface.csv")
    components = pd.read_csv(metrics_dir / "component_mediation.csv")
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    axes[0].scatter(interface["behavior_effect"], interface["rank_effect"], color=COLORS["blue"])
    limits = [0, max(interface["behavior_effect"].max(), interface["rank_effect"].max()) * 1.05]
    axes[0].plot(limits, limits, color=COLORS["gray"], linestyle="--", linewidth=1)
    axes[0].set_xlim(limits)
    axes[0].set_ylim(limits)
    axes[0].set_xlabel("Behavioral joint effect")
    axes[0].set_ylabel("One-dimensional interface effect")
    axes[0].set_title("Compact causal interface")
    grouped = components.groupby("model")[["validation_signal", "validation_necessity", "validation_control"]].median()
    x_values = np.arange(len(grouped))
    width = 0.25
    axes[1].bar(x_values - width, grouped["validation_signal"], width, label="Sufficiency", color=COLORS["blue"])
    axes[1].bar(x_values, grouped["validation_necessity"], width, label="Necessity", color=COLORS["green"])
    axes[1].bar(x_values + width, grouped["validation_control"], width, label="Control", color=COLORS["gray"])
    axes[1].set_xticks(x_values, grouped.index, rotation=30, ha="right")
    axes[1].set_ylabel("Normalized effect")
    axes[1].set_title("Component mediation")
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    finish_figure(figure, output_dir, "figure5_compact_interface")


def figure6(metrics_dir: Path, output_dir: Path) -> None:
    table = pd.read_csv(metrics_dir / "scale_transition.csv")
    figure, axes = plt.subplots(1, 2, figsize=(8.1, 3.25))
    axes[0].plot(table["parameters_billion"], table["concordance"], marker="o", label="Patient concordance", color=COLORS["blue"])
    axes[0].plot(table["parameters_billion"], table["human_agreement"], marker="s", label="Causal-source recognition", color=COLORS["green"])
    axes[0].set_xscale("log")
    axes[0].set_ylim(0.45, 1.04)
    axes[0].set_xlabel("Parameters (billions)")
    axes[0].set_ylabel("Behavioral score")
    axes[0].set_title("Role construction")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(table["parameters_billion"], table["wrongness_gain"], marker="o", label="Wrongness gain", color=COLORS["orange"])
    axes[1].plot(table["parameters_billion"], table["outcome_conditioned_gain"], marker="s", label="Outcome-conditioned gain", color=COLORS["red"])
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Parameters (billions)")
    axes[1].set_ylabel("Readout change")
    axes[1].set_title("Relation-conditioned evaluation")
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    finish_figure(figure, output_dir, "figure6_scale_transition")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate released metrics and rebuild main-result summary figures.")
    parser.add_argument("--metrics-dir", type=Path, default=PROJECT_ROOT / "results/paper_metrics")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/paper_figures")
    arguments = parser.parse_args()
    counts = validate_metrics(arguments.metrics_dir)
    figure2(arguments.metrics_dir, arguments.output_dir)
    figure3(arguments.metrics_dir, arguments.output_dir)
    figure4(arguments.metrics_dir, arguments.output_dir)
    figure5(arguments.metrics_dir, arguments.output_dir)
    figure6(arguments.metrics_dir, arguments.output_dir)
    print({"status": "ok", "figures": 5, **counts})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
