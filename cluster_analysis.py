import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from datetime import datetime
import sys

warnings.filterwarnings('ignore')

class VerboseLogger:
    """Class to handle logging to both console and file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
        # Write header with timestamp
        self.log.write(f"Cluster Analysis Log\n")
        self.log.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.write("="*60 + "\n\n")
        self.log.flush()
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

def load_and_preprocess_data(file_path):
    """Load and preprocess the fire database."""
    # Read the CSV file
    data = pd.read_csv(file_path)
    print(f"Original dataset shape: {data.shape}")
    
    # Remove rows with missing Lat/Lon values
    data = data.dropna(subset=['Lat', 'Lon'])
    
    # Filter out negative longitudes (data errors for New Zealand)
    data = data[data['Lon'] > 0]
    print(f"After location filtering: {data.shape}")
    
    # Define clustering columns
    clustering_cols = ['Elev', 'Slope', 'Temp', 'RH']
    
    # Replace -9999 values with NaN
    for col in clustering_cols:
        if col in data.columns:
            data[col] = data[col].replace(-9999, np.nan)
    
    # Find columns with sufficient data (at least 10% valid)
    valid_cols = []
    for col in clustering_cols:
        if col in data.columns:
            valid_ratio = data[col].notna().sum() / len(data)
            if valid_ratio >= 0.1:  # At least 10% valid data
                valid_cols.append(col)
    
    print(f"Columns with sufficient data (>=10% valid): {valid_cols}")
    
    if len(valid_cols) == 0:
        print("No columns have sufficient data for clustering!")
        return None, None
    
    # Remove rows with missing values only in the valid columns
    data_clean = data.dropna(subset=valid_cols)
    print(f"After removing missing values in valid columns: {data_clean.shape}")
    
    # Standardize the data
    scaler = StandardScaler()
    data_standardised = scaler.fit_transform(data_clean[valid_cols])
    
    return data_clean, data_standardised, valid_cols

def compute_elbow_curve(data_standardised, max_clusters=10):
    """Compute and plot the elbow curve for optimal number of clusters."""
    inertias = []
    cluster_range = range(1, max_clusters + 1)
    
    print("\nComputing elbow curve...")
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(data_standardised)
        inertias.append(kmeans.inertia_)
        print(f"Clusters {n_clusters}: Inertia = {kmeans.inertia_:.2f}")
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Curve for Optimal Number of Clusters')
    plt.grid(True, alpha=0.3)
    plt.xticks(cluster_range)
    
    # Add annotations for the rate of change
    for i in range(1, len(inertias)):
        rate = (inertias[i-1] - inertias[i]) / inertias[i-1] * 100
        plt.annotate(f'{rate:.1f}%', 
                    xy=(i+1, inertias[i]),
                    xytext=(i+1, inertias[i] + inertias[0]*0.05),
                    ha='center')
    
    plt.tight_layout()
    plt.savefig('elbow_curve_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return inertias

def compute_silhouette_analysis(data_standardised, max_clusters=10):
    """Compute and plot silhouette analysis for different numbers of clusters."""
    silhouette_scores = []
    cluster_range = range(2, max_clusters + 1)  # Silhouette requires at least 2 clusters
    
    print("\nComputing silhouette analysis...")
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data_standardised)
        silhouette_avg = silhouette_score(data_standardised, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"Clusters {n_clusters}: Silhouette Score = {silhouette_avg:.4f}")
    
    # Find optimal number of clusters
    optimal_idx = np.argmax(silhouette_scores)
    optimal_clusters = cluster_range[optimal_idx]
    best_score = silhouette_scores[optimal_idx]
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
    plt.axvline(x=optimal_clusters, color='red', linestyle='--', alpha=0.7,
                label=f'Optimal: {optimal_clusters} clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Average Silhouette Score')
    plt.title('Silhouette Analysis for Optimal Number of Clusters')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(cluster_range)
    
    # Annotate the optimal point
    plt.annotate(f'Best: {best_score:.4f}',
                xy=(optimal_clusters, best_score),
                xytext=(optimal_clusters + 0.5, best_score + 0.01),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig('silhouette_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed silhouette plot for optimal number of clusters
    create_detailed_silhouette_plot(data_standardised, optimal_clusters)
    
    return silhouette_scores, optimal_clusters

def create_detailed_silhouette_plot(data_standardised, n_clusters):
    """Create a detailed silhouette plot for the optimal number of clusters."""
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(data_standardised)
    
    # Calculate silhouette scores
    silhouette_avg = silhouette_score(data_standardised, cluster_labels)
    sample_silhouette_values = silhouette_samples(data_standardised, cluster_labels)
    
    # Create silhouette plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Silhouette plot
    y_lower = 10
    colors = plt.cm.nipy_spectral(np.linspace(0, 1, n_clusters))
    
    for i in range(n_clusters):
        # Aggregate silhouette scores for samples belonging to cluster i
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = colors[i]
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
        
        # Calculate cluster statistics
        cluster_avg = np.mean(ith_cluster_silhouette_values)
        cluster_min = np.min(ith_cluster_silhouette_values)
        cluster_max = np.max(ith_cluster_silhouette_values)
        
        print(f"Cluster {i}: {size_cluster_i:5d} samples, "
              f"avg={cluster_avg:.4f}, min={cluster_min:.4f}, max={cluster_max:.4f}")
    
    ax1.set_xlabel('Silhouette Coefficient Values')
    ax1.set_ylabel('Cluster Label')
    ax1.set_title(f'Silhouette Plot for {n_clusters} Clusters')
    
    # Add vertical line for average silhouette score
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--",
                label=f'Average Score: {silhouette_avg:.4f}')
    ax1.legend()
    
    # Scatter plot colored by cluster
    colors = plt.cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(data_standardised[:, 0], data_standardised[:, 1],
                marker='o', s=30, c=colors, alpha=0.7)
    
    # Mark cluster centers
    centers = kmeans.cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1], marker='x',
                s=300, linewidths=3, color='red', label='Centroids')
    
    ax2.set_xlabel('First Standardized Feature')
    ax2.set_ylabel('Second Standardized Feature')
    ax2.set_title(f'Cluster Visualization (First 2 Features)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'detailed_silhouette_plot_{n_clusters}_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to execute the cluster analysis."""
    
    # Set up verbose logging
    logger = VerboseLogger('cluster_analysis_log.txt')
    sys.stdout = logger
    
    try:
        # Load and preprocess data
        print("Loading and preprocessing data...")
        data, data_standardised, valid_cols = load_and_preprocess_data('FOD_FENZ_updated2024-07-11(in).csv')
        
        if data is None:
            print("Cannot proceed with analysis - insufficient data")
            return
        
        # Compute and plot elbow curve
        print("\nComputing elbow curve...")
        inertias = compute_elbow_curve(data_standardised)
        
        # Compute and plot silhouette analysis
        print("\nComputing silhouette analysis...")
        silhouette_scores, optimal_clusters = compute_silhouette_analysis(data_standardised)
        
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Optimal number of clusters based on silhouette analysis: {optimal_clusters}")
        print("All plots have been saved as PNG files.")
        print("Detailed analysis log has been saved to 'cluster_analysis_log.txt'")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Restore stdout and close log file
        sys.stdout = logger.terminal
        logger.close()
        print("Analysis complete. Check 'cluster_analysis_log.txt' for detailed output.")

if __name__ == "__main__":
    main() 