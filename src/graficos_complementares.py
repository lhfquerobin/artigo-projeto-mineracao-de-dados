"""Recria os gráficos complementares a partir dos resultados dos experimentos."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd


def pt(value, digits=2):
    return f"{value:,.{digits}f}".translate(str.maketrans(".,", ",."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    results = parser.parse_args().project_root / "results"
    audit = json.loads((results / "audit.json").read_text(encoding="utf-8"))
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    blue = "#39718e"

    fields = {"budget": "Orçamento", "imdb_id": "Identificador IMDb", "runtime": "Duração", "genres": "Gêneros"}
    rates = sorted(((fields[key], 100 * audit["missing_fraction"][key]) for key in fields), key=lambda item: item[1])
    fig, ax = plt.subplots(figsize=(8, 3.4), layout="constrained")
    bars = ax.barh([key for key, _ in rates], [val for _, val in rates], color=blue)
    ax.bar_label(bars, labels=[f"{pt(val)}%" for _, val in rates], padding=5)
    ax.set(xlim=(0, 47), xlabel="Registros sem valor válido (%)")
    ax.xaxis.grid(alpha=.25)
    ax.set_axisbelow(True)
    fig.savefig(results / "cobertura_atributos.png", dpi=220)
    plt.close(fig)

    groups = [("Treino", audit["train"]), ("Validação", audit["validation"]), ("Teste", audit["test"])]
    fig, ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
    for i, (label, group) in enumerate(groups):
        value = group["median_revenue"]
        ax.scatter(value, i, color=blue, s=65, zorder=3)
        ax.annotate(pt(value, 0), (value, i), xytext=(9, 0), textcoords="offset points", va="center")
    ax.set_yticks(range(3), [f"{label} ({g['first_year']}-{g['last_year']})" for label, g in groups])
    ax.invert_yaxis()
    ax.set(xscale="log", xlim=(3e4, 1.6e7), ylim=(2.55, -.55),
           xlabel="Mediana da receita (unidades monetárias nominais, escala log)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: pt(value, 0)))
    ax.xaxis.grid(alpha=.25)
    fig.savefig(results / "receita_por_periodo.png", dpi=220)
    plt.close(fig)

    data = pd.read_csv(results / "test_predictions.csv")
    data["quartil"] = pd.qcut(data["observed_revenue"], q=4)
    data["erro_absoluto"] = (data["predicted_revenue"] - data["observed_revenue"]).abs()
    summary = data.groupby("quartil", observed=True).agg(
        n=("id", "size"), receita_min=("observed_revenue", "min"),
        receita_max=("observed_revenue", "max"), mae=("erro_absoluto", "mean")
    ).reset_index(drop=True)
    summary.index = range(1, len(summary) + 1)
    summary.index.name = "quartil"
    summary.to_csv(results / "erros_por_quartil.csv")
    assert summary["n"].sum() == len(data)
    assert abs((summary.mae * summary.n).sum() / len(data) - data.erro_absoluto.mean()) < .01
    fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
    bars = ax.bar(summary.index, summary.mae / 1e6, color=["#87a8b9"] * 3 + [blue], width=.63)
    ax.bar_label(bars, labels=[pt(x / 1e6) for x in summary.mae], padding=5)
    ax.set_xticks(summary.index, [f"Q{i}\n(n={n})" for i, n in zip(summary.index, summary.n)])
    ax.set(xlabel="Quartil de receita observada (Q1 = menor receita)", ylabel="MAE (milhões de unidades monetárias)",
           ylim=(0, max(summary.mae) / 1e6 * 1.18))
    ax.yaxis.grid(alpha=.25)
    ax.set_axisbelow(True)
    fig.savefig(results / "erros_por_quartil.png", dpi=220)
    plt.close(fig)
    print(summary.to_string())


if __name__ == "__main__":
    main()

