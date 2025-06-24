import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Create output folder
output_dir = "pca"
os.makedirs(output_dir, exist_ok=True)

# Prepare logging
log_file_path = os.path.join(output_dir, "pca_cluster_analysis_log.txt")
log_file = open(log_file_path, 'w')

class Logger:
    def __init__(self, file):
        self.terminal = sys.stdout
        self.log = file
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_file)

# Config
DATA_FILE = 'FOD_FENZ_updated2024-07-11(in).csv'
CLUSTERING_COLUMNS = ['Elev', 'Slope', 'Temp', 'RH']
N_COMPONENTS = 3
N_CLUSTERS = 4
RANDOM_STATE = 42

# Load and clean data
df = pd.read_csv(DATA_FILE)
df = df.dropna(subset=['Lat', 'Lon'])
df = df[df['Lon'] > 0]
for col in CLUSTERING_COLUMNS:
    df[col] = df[col].replace(-9999, np.nan)
df_clean = df.dropna(subset=CLUSTERING_COLUMNS).copy()

# Standardize and PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[CLUSTERING_COLUMNS])
pca = PCA(n_components=N_COMPONENTS)
X_pca = pca.fit_transform(X_scaled)

# Add PCA components to DataFrame
for i in range(N_COMPONENTS):
    df_clean[f'PC{i+1}'] = X_pca[:, i]

# KMeans clustering
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
df_clean['cluster'] = kmeans.fit_predict(X_pca)

# Explained variance
print("Explained variance by PCA components:")
for i, var in enumerate(pca.explained_variance_ratio_, start=1):
    print(f"PC{i}: {var:.2%}")

# Component loadings
loading_matrix = pd.DataFrame(pca.components_.T, index=CLUSTERING_COLUMNS,
                              columns=[f'PC{i+1}' for i in range(N_COMPONENTS)])
print("\nPCA Component Loadings:")
print(loading_matrix)

# Cluster means in original variables
cluster_means = df_clean.groupby('cluster')[CLUSTERING_COLUMNS].mean()
print("\nCluster means in original variable space:")
print(cluster_means)

# Save PCA scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=df_clean, palette='deep', alpha=0.6)
plt.title('KMeans Clusters in PCA Space')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, "pca_clusters.png"), dpi=300, bbox_inches='tight')
plt.close()

# Save PCA loadings heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(loading_matrix, annot=True, cmap='coolwarm', center=0, cbar_kws={'label': 'Loading'})
plt.title('PCA Component Loadings')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pca_loadings_heatmap.png"), dpi=300)
plt.close()

# Save boxplots of original vars by cluster
plt.figure(figsize=(14, 8))
for i, col in enumerate(CLUSTERING_COLUMNS):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='cluster', y=col, data=df_clean, palette='deep')
    plt.title(f'{col} by Cluster')
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cluster_boxplots_original_vars.png"), dpi=300)
plt.close()

# Save fire map by cluster
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Lon', y='Lat', hue='cluster', data=df_clean, palette='deep', s=10, alpha=0.7)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Fire Clusters (Mapped)')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fire_clusters_map.png"), dpi=300)
plt.close()

# Save final dataset
df_clean.to_csv(os.path.join(output_dir, "fire_data_with_clusters.csv"), index=False)

# Define regions with bounding boxes: [lon_min, lon_max, lat_min, lat_max]
regions = {
    "Northland":     [172.5, 175.5, -36.0, -34.0],
    "Rotorua":       [175.5, 177.5, -39.0, -37.0],
    "Wellington":    [174.0, 176.0, -42.7, -40.7],
    "Canterbury":    [171.0, 173.0, -44.5, -42.5],
}

for name, (lon_min, lon_max, lat_min, lat_max) in regions.items():
    plt.figure(figsize=(8, 6))
    subset = df_clean[(df_clean['Lon'] >= lon_min) & (df_clean['Lon'] <= lon_max) &
                      (df_clean['Lat'] >= lat_min) & (df_clean['Lat'] <= lat_max)]
    sns.scatterplot(x='Lon', y='Lat', hue='cluster', data=subset, palette='deep', s=10, alpha=0.7)
    plt.title(f'Fire Clusters – {name}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.xlim(lon_min, lon_max)
    plt.ylim(lat_min, lat_max)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fire_clusters_{name.lower()}.png'), dpi=300)
    plt.close()


# Finalize log
sys.stdout.log.close()
sys.stdout = sys.stdout.terminal

print("All PCA clustering outputs have been saved in the 'pca/' folder.")


