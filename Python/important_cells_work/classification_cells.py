import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import matplotlib.cm as cm

"""
The aim of this script is to classify the base stations based on the number of users
present at each cell hour by hour.

Method:
  - Aggregate the hourly usage over all days (mean per cell)
  - Adaptive baseline subtraction: subtract alpha * min(profile) where alpha scales
    with the coefficient of variation (CV = std/mean). Flat cells (low CV) keep their
    raw profile; peaked cells (high CV) get their baseline removed.
  - L1 normalisation on the adjusted profile
  - K-Means clustering on the normalised 24h vector
  - Elbow curve to help choose the number of clusters
"""

MAIN_DIR = Path(__file__).parent.parent.parent
INPUT_DIR = MAIN_DIR / "results/intermediate_result"
OUTPUT_DIR = MAIN_DIR / "results/classification"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HOUR_COLS = [f"{h}h-{h+1}h" for h in range(24)]

# Number of clusters — adjust after reading the elbow curve
N_CLUSTERS = 3


# ── 1. Load & aggregate hourly usage ──────────────────────────────────────────

df_hourly = pd.read_csv(INPUT_DIR / "stats_use_cell_by_hour_by_day.csv", sep=";")

# Mean profile per cell across all available days
cell_profiles = df_hourly.groupby("cellid")[HOUR_COLS].mean()

# Drop cells that were never active
cell_profiles = cell_profiles[cell_profiles.sum(axis=1) > 0]

print(f"{len(cell_profiles)} active cells loaded.")


# ── 2. Adaptive baseline subtraction + L1 normalisation ───────────────────────
#
# CV threshold: below this value a cell is considered "flat" (alpha ≈ 0).
# Above this value the full minimum is subtracted (alpha → 1).
# Typical range: 0.1 – 0.3. Increase to be more conservative (subtract less).
CV_THRESHOLD = 0.4

raw = cell_profiles.values                          # shape (n_cells, 24)
cell_mean = raw.mean(axis=1, keepdims=True)
cell_std  = raw.std(axis=1,  keepdims=True)
cell_min  = raw.min(axis=1,  keepdims=True)

# alpha ∈ [0, 1]: how much of the minimum to remove
cv = np.where(cell_mean > 0, cell_std / cell_mean, 0)
alpha = np.clip(cv / CV_THRESHOLD, 0, 1)            # shape (n_cells, 1)

adjusted = raw - alpha * cell_min                   # partial baseline removal
profiles_norm = normalize(adjusted, norm="l1")


# ── 3. Elbow method — help choose N_CLUSTERS ──────────────────────────────────

K_range = range(2, 11)
inertias = []
for k in K_range:
    km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_tmp.fit(profiles_norm)
    inertias.append(km_tmp.inertia_)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(list(K_range), inertias, marker="o")
ax.set_xlabel("Nombre de clusters K")
ax.set_ylabel("Inertie")
ax.set_title("Méthode du coude — choix du nombre de clusters")
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "elbow_curve.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Courbe du coude → {OUTPUT_DIR / 'elbow_curve.png'}")


# ── 4. Final K-Means clustering ───────────────────────────────────────────────

km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
labels = km.fit_predict(profiles_norm)
cell_profiles = cell_profiles.copy()
cell_profiles["cluster"] = labels


# ── 5. Plot cluster centroids ──────────────────────────────────────────────────

hours = list(range(24))
colors = cm.tab10.colors

# Bar charts — one subplot per cluster
fig, axes = plt.subplots(N_CLUSTERS, 1, figsize=(13, 3 * N_CLUSTERS), sharex=True)
if N_CLUSTERS == 1:
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
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "cluster_centroids_bar.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Heatmap — all centroids on one figure
fig, ax = plt.subplots(figsize=(13, max(2, N_CLUSTERS)))
im = ax.imshow(km.cluster_centers_, aspect="auto", cmap="YlOrRd")
ax.set_yticks(range(N_CLUSTERS))
ax.set_yticklabels([f"Cluster {i}" for i in range(N_CLUSTERS)])
ax.set_xticks(hours)
ax.set_xticklabels([f"{h}h" for h in hours], rotation=45, fontsize=7)
ax.set_title("Heatmap des centroïdes (profil horaire normalisé)")
plt.colorbar(im, ax=ax, label="Part relative d'utilisation")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "cluster_centroids_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"Graphiques des centroïdes → {OUTPUT_DIR}")


# ── 6. Export classification ───────────────────────────────────────────────────

out_csv = OUTPUT_DIR / "classification_base_stations.csv"
cell_profiles[["cluster"]].to_csv(out_csv, sep=";")

print(f"\nClassification exportée → {out_csv}")
print("\nDistribution des clusters :")
print(cell_profiles["cluster"].value_counts().sort_index().to_string())
