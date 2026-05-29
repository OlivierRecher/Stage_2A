import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import matplotlib.cm as cm

"""
The aim of this script is to classify the base stations based on the number of users
present at each cell hour by hour — one classification per day.

Method (applied independently for each day):
  - Adaptive baseline subtraction: subtract alpha * min(profile) where alpha scales
    with the coefficient of variation (CV = std/mean). Flat cells (low CV) keep their
    raw profile; peaked cells (high CV) get their baseline removed.
  - L1 normalisation on the adjusted profile
  - K-Means clustering on the normalised 24h vector
  - Elbow curve + centroid plots saved per-day folder
"""

MAIN_DIR = Path(__file__).parent.parent.parent
INPUT_DIR = MAIN_DIR / "results/intermediate_result"
BASE_OUTPUT_DIR = MAIN_DIR / "results"

HOUR_COLS = [f"{h}h-{h+1}h" for h in range(24)]

# Number of clusters — adjust after reading the elbow curves
N_CLUSTERS = 3

# CV threshold for adaptive baseline subtraction
CV_THRESHOLD = 0.4


def classify_day(day_str: str, cell_profiles: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cell_profiles = cell_profiles[cell_profiles.sum(axis=1) > 0].copy()
    if len(cell_profiles) < N_CLUSTERS:
        print(f"  [{day_str}] Pas assez de cellules actives ({len(cell_profiles)}), ignoré.")
        return

    print(f"  [{day_str}] {len(cell_profiles)} cellules actives.")

    # ── Adaptive baseline subtraction + L1 normalisation ────────────────────────
    raw = cell_profiles.values
    cell_mean = raw.mean(axis=1, keepdims=True)
    cell_std  = raw.std(axis=1,  keepdims=True)
    cell_min  = raw.min(axis=1,  keepdims=True)

    cv = np.where(cell_mean > 0, cell_std / cell_mean, 0)
    alpha = np.clip(cv / CV_THRESHOLD, 0, 1)

    adjusted = raw - alpha * cell_min
    profiles_norm = normalize(adjusted, norm="l1")

    # ── Elbow curve ──────────────────────────────────────────────────────────────
    K_range = range(2, min(11, len(cell_profiles)))
    inertias = []
    for k in K_range:
        km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_tmp.fit(profiles_norm)
        inertias.append(km_tmp.inertia_)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(K_range), inertias, marker="o")
    ax.set_xlabel("Nombre de clusters K")
    ax.set_ylabel("Inertie")
    ax.set_title(f"Méthode du coude — {day_str}")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "elbow_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Final K-Means clustering ─────────────────────────────────────────────────
    n_clusters = min(N_CLUSTERS, len(cell_profiles))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(profiles_norm)
    cell_profiles["cluster"] = labels

    # ── Bar charts — one subplot per cluster ─────────────────────────────────────
    hours = list(range(24))
    colors = cm.tab10.colors

    fig, axes = plt.subplots(n_clusters, 1, figsize=(13, 3 * n_clusters), sharex=True)
    if n_clusters == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        centroid = km.cluster_centers_[i]
        count = int((labels == i).sum())
        ax.bar(hours, centroid, color=colors[i % len(colors)], alpha=0.85)
        ax.set_title(f"Cluster {i}  ({count} cellules)", fontsize=10)
        ax.set_ylabel("Part relative")
        ax.set_xticks(hours)
        ax.set_xticklabels([f"{h}h" for h in hours], rotation=45, fontsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.suptitle(f"Centroïdes — {day_str}", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "cluster_centroids_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Heatmap ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, max(2, n_clusters)))
    im = ax.imshow(km.cluster_centers_, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels([f"Cluster {i}" for i in range(n_clusters)])
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h}h" for h in hours], rotation=45, fontsize=7)
    ax.set_title(f"Heatmap des centroïdes — {day_str}")
    plt.colorbar(im, ax=ax, label="Part relative d'utilisation")
    fig.tight_layout()
    fig.savefig(output_dir / "cluster_centroids_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Export classification ────────────────────────────────────────────────────
    cell_profiles[["cluster"]].to_csv(output_dir / "classification_base_stations.csv", sep=";")

    print(f"  [{day_str}] Résultats → {output_dir}")


# ── Load data ────────────────────────────────────────────────────────────────────

df_hourly = pd.read_csv(INPUT_DIR / "stats_use_cell_by_hour_by_day.csv", sep=";")
days = sorted(df_hourly["day"].unique())
print(f"{len(days)} jours trouvés dans le fichier.")

for day in days:
    day_df = df_hourly[df_hourly["day"] == day].set_index("cellid")[HOUR_COLS]
    day_output = BASE_OUTPUT_DIR / str(day)
    classify_day(str(day), day_df, day_output)

print("\nClassification jour par jour terminée.")
