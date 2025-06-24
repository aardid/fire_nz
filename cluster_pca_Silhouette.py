# silhouette_pca_only.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# ----------------------------- CONFIG ---------------------------------
DATA_FILE       = 'FOD_FENZ_updated2024-07-11(in).csv'   # <- update path if needed
CLUSTER_COLS    = ['Elev', 'Slope', 'Temp', 'RH']
N_COMPONENTS    = 3
N_CLUSTERS      = 4
RANDOM_STATE    = 42
OUTPUT_DIR      = 'pca'
# ----------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1) LOAD & BASIC CLEAN
df = pd.read_csv(DATA_FILE)
df = df.dropna(subset=['Lat', 'Lon'])            # need coords
df = df[df['Lon'] > 0]                           # NZ only (positive longitudes)
for col in CLUSTER_COLS:
    df[col] = df[col].replace(-9999, np.nan)     # sentinel → NaN
df_clean = df.dropna(subset=CLUSTER_COLS).copy()

# 2) STANDARDISE + PCA
X_scaled = StandardScaler().fit_transform(df_clean[CLUSTER_COLS])
pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

# -------------------------------------------------------------
# Average-silhouette-vs-k analysis  (drop in after X_pca exists)
# -------------------------------------------------------------
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

k_min, k_max = 2, 10            # <-- adjust if needed
sil_scores   = []

for k in range(k_min, k_max + 1):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_pca)
    sil = silhouette_score(X_pca, labels)
    sil_scores.append(sil)
    print(f'k={k}: silhouette = {sil:.4f}')

# Identify best k
best_idx = int(np.argmax(sil_scores))
best_k   = k_min + best_idx
best_val = sil_scores[best_idx]

# ----------------------------- PLOT ---------------------------
plt.figure(figsize=(10, 6))
k_range = list(range(k_min, k_max + 1))
plt.plot(k_range, sil_scores, 'o-', color='royalblue', linewidth=2, markersize=6)
plt.axvline(best_k, color='red', linestyle='--', label=f'Optimal: {best_k} clusters')
plt.annotate(f'Best: {best_val:.4f}',
             xy=(best_k, best_val),
             xytext=(best_k + 0.2, best_val + 0.005),
             color='red',
             arrowprops=dict(arrowstyle='->', color='red'))

plt.title('Silhouette Analysis for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Average Silhouette Score')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.savefig(os.path.join(OUTPUT_DIR, 'silhouette_vs_k.png'), dpi=300)
plt.close()


# 3) KMEANS IN PCA SPACE
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
labels = kmeans.fit_predict(X_pca)

# 4) SILHOUETTE ANALYSIS
sil_avg   = silhouette_score(X_pca, labels)
samples   = silhouette_samples(X_pca, labels)
print(f"\nAverage silhouette score (PCA, {N_CLUSTERS} clusters): {sil_avg:.4f}")

# 5) SILHOUETTE PLOT ---------------------------------------------------
sns.set_style("white")
fig, ax = plt.subplots(figsize=(8, 6))
y_lower = 10
for i in range(N_CLUSTERS):
    vals = samples[labels == i]
    vals.sort()
    size = vals.shape[0]
    y_upper = y_lower + size
    color = plt.cm.nipy_spectral(float(i) / N_CLUSTERS)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                     facecolor=color, alpha=0.7)
    ax.text(-0.05, y_lower + 0.5*size, str(i))
    y_lower = y_upper + 10      # spacing

ax.axvline(x=sil_avg, color='red', linestyle='--', label=f'Avg = {sil_avg:.3f}')
ax.set_xlabel('Silhouette coefficient')
ax.set_ylabel('Sample index')
ax.set_title(f'Silhouette plot – PCA space ({N_CLUSTERS} clusters)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'silhouette_pca_clusters.png'), dpi=300)
plt.close()

