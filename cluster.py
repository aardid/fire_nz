# Fire Database Clustering Analysis

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from geopy.distance import geodesic
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
        self.log.write(f"Fire Database Clustering Analysis Log\n")
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

def main():
    """Main function to execute the fire database clustering analysis."""
    
    # Set up verbose logging
    logger = VerboseLogger('clustering_analysis_log.txt')
    sys.stdout = logger
    
    try:
        # Load fire database
        print("Loading fire database...")
        result = load_fire_data('FOD_FENZ_updated2024-07-11(in).csv')
        
        if result is None:
            print("Cannot proceed with clustering - insufficient data")
            return
        
        data, columns_to_cluster = result
        
        if len(data) == 0:
            print("No valid data remaining after cleaning")
            return
        
        # Plot fire locations
        plot_fire_locations(data)
        
        print(f"\nUsing columns for clustering: {columns_to_cluster}")
        
        # Plot the selected variables
        plot_fire_variables(data, columns_to_cluster)
        
        # Standardize the data for clustering
        data_standardised = standardise_data(data, columns_to_cluster, plot_standardised=True)
        
        # Perform silhouette analysis to find optimal clusters
        #optimal_clusters, silhouette_scores = silhouette_analysis(data_standardised, max_clusters=12)
        
        # Detailed silhouette analysis for the optimal number
        #detailed_silhouette_analysis(data_standardised, optimal_clusters)
        
        # Compare different clustering methods
        #compare_clustering_methods(data_standardised, optimal_clusters)
        
        # Plot elbow curve for comparison
        #plot_elbow_curve(data_standardised)
        
        # Use 4 clusters as default
        num_clusters_to_use = 4
        
        # Perform clustering with 4 clusters
        perform_clustering(data, data_standardised, num_clusters_to_use, columns_to_cluster)
        visualize_clusters(data, data_standardised, num_clusters_to_use, columns_to_cluster)
        
        # Plot individual clusters in subplots
        plot_clusters_subplots(data, num_clusters_to_use)
        
        # Plot boxplots by cluster in subplots
        plot_cluster_boxplots_subplots(data, columns_to_cluster, num_clusters_to_use)

        # Explore cluster characteristics in detail
        explore_cluster_characteristics(data, columns_to_cluster, num_clusters_to_use)

        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Total processing time: Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("All plots have been saved as PNG files.")
        print("Detailed analysis log has been saved to 'clustering_analysis_log.txt'")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Restore stdout and close log file
        sys.stdout = logger.terminal
        logger.close()
        print("Analysis complete. Check 'clustering_analysis_log.txt' for detailed output.")

def load_fire_data(file_path):
    """Load and clean fire database."""
    # Read the CSV file
    data = pd.read_csv(file_path)
    print(f"Original dataset shape: {data.shape}")
    
    # Remove rows with missing Lat/Lon values
    data = data.dropna(subset=['Lat', 'Lon'])
    
    # Filter out negative longitudes (data errors for New Zealand)
    data = data[data['Lon'] > 0]
    print(f"After location filtering: {data.shape}")
    
    # Replace -9999 values with NaN for the clustering columns
    clustering_cols = ['Elev', 'Slope', 'Temp', 'RH',]
    
    # Check data availability for each column
    print("Data availability for clustering columns:")
    for col in clustering_cols:
        if col in data.columns:
            # Count non-missing values (not -9999 and not NaN)
            valid_count = len(data[(data[col] != -9999) & (data[col].notna())])
            total_count = len(data)
            print(f"{col}: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}% valid)")
        else:
            print(f"{col}: Column not found")
    
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
        return None
    
    # Remove rows with missing values only in the valid columns
    data_clean = data.dropna(subset=valid_cols)
    print(f"After removing missing values in valid columns: {data_clean.shape}")
    
    return data_clean, valid_cols

def plot_fire_locations(data):
    """Plot latitude and longitude as points."""
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAND, alpha=0.3)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)
    
    # Plot fire locations
    ax.scatter(data['Lon'], data['Lat'], alpha=0.6, s=10, c='red', 
               edgecolors='none', transform=ccrs.PlateCarree())
    
    # Set extent to New Zealand
    ax.set_extent([165, 180, -48, -34], crs=ccrs.PlateCarree())
    
    # Add gridlines
    ax.gridlines(draw_labels=True, alpha=0.3)
    
    plt.title('Fire Incident Locations in New Zealand')
    
    # Add statistics
    plt.text(0.02, 0.98, f'Total incidents: {len(data)}', 
             transform=ax.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('fire_locations.png', dpi=300, bbox_inches='tight')
    #plt.show()

def plot_fire_variables(data, columns_to_cluster):
    """Plot the fire variables on subplots in one row."""
    n_cols = len(columns_to_cluster)
    
    fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 3), sharey=True)
    if n_cols == 1:
        axes = [axes]
    
    for i, col in enumerate(columns_to_cluster):
        if col in data.columns:
            axes[i].hist(data[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{col} Distribution')
            axes[i].set_xlabel(col)
            if i == 0:  # Only set ylabel for the first subplot
                axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fire_variables_distribution.png', dpi=300, bbox_inches='tight')
    #plt.show()

def standardise_data(data, columns_to_cluster, plot_standardised=False):
    """Standardise the data for clustering."""
    # Load StandardScaler object
    scaler = StandardScaler()
    
    # Extract the section of the dataframe that we will be scaling
    data_for_scaling = data[columns_to_cluster]
    
    # Apply the scaling
    data_standardised = scaler.fit_transform(data_for_scaling)
    
    # Display the scaled data
    if plot_standardised:
        plot_standardised_data(data_standardised, columns_to_cluster)
    
    return data_standardised

def plot_standardised_data(data_standardised, columns_to_cluster):
    """Plot standardised data distributions in one row."""
    n_cols = len(columns_to_cluster)
    
    fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 3), sharey=True)
    if n_cols == 1:
        axes = [axes]
    
    for i, col in enumerate(columns_to_cluster):
        axes[i].hist(data_standardised[:, i], bins=30, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'Standardised {col}')
        axes[i].set_xlabel(f'Standardised {col}')
        if i == 0:  # Only set ylabel for the first subplot
            axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('standardised_data_distribution.png', dpi=300, bbox_inches='tight')
    #plt.show()

def plot_elbow_curve(data_standardised):
    """Plot the elbow curve for optimal number of clusters."""
    inertias = []
    max_clusters = 10
    
    for n_clusters in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(data_standardised)
        inertia = kmeans.inertia_
        inertias.append(inertia)
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_clusters + 1), inertias, marker='o')
    plt.title('Elbow Curve for Optimal Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.xticks(np.arange(1, max_clusters + 1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('elbow_curve.png', dpi=300, bbox_inches='tight')
    #plt.show()

def perform_clustering(data, data_standardised, num_clusters, columns_to_cluster):
    """Apply KMeans clustering with specified number of clusters."""
    # Create KMeans clustering object
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    
    # Fit the model and predict clusters
    kmeans.fit(data_standardised)
    membership = kmeans.predict(data_standardised)
    
    # Save membership into the dataframe
    data['cluster'] = membership
    
    # Plot clusters on map with NZ contours
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=1.5)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAND, alpha=0.2, color='lightgray')
    ax.add_feature(cfeature.OCEAN, alpha=0.3, color='lightblue')
    
    # Plot clusters
    cluster_colors = sns.color_palette('deep', n_colors=len(data['cluster'].unique()))
    
    for cluster, color in zip(data['cluster'].unique(), cluster_colors):
        cluster_data = data[data['cluster'] == cluster]
        fire_count = len(cluster_data)
        fire_percent = (fire_count / len(data)) * 100
        ax.scatter(cluster_data['Lon'], cluster_data['Lat'], 
                  c=[color], label=f'Cluster {cluster}\n({fire_count} fires, {fire_percent:.1f}%)', s=15, alpha=0.7,
                  transform=ccrs.PlateCarree())
    
    # Set extent to New Zealand
    ax.set_extent([165, 180, -48, -34], crs=ccrs.PlateCarree())
    
    # Add gridlines
    ax.gridlines(draw_labels=True, alpha=0.3)
    
    plt.title(f'Fire Incident Clusters (Number of Clusters: {num_clusters})')
    plt.legend(loc=1)  # loc=2 is upper left
    plt.tight_layout()
    plt.savefig(f'clusters_map_{num_clusters}_clusters.png', dpi=300, bbox_inches='tight')
    #plt.show()
    
    # Print cluster statistics
    print(f"\nCluster Statistics for {num_clusters} clusters:")
    for cluster in sorted(data['cluster'].unique()):
        cluster_data = data[data['cluster'] == cluster]
        print(f"Cluster {cluster}: {len(cluster_data)} incidents ({len(cluster_data)/len(data)*100:.1f}%)")

def visualize_clusters(data, data_standardised, num_clusters, columns_to_cluster):
    """Visualize cluster characteristics."""
    # Create cluster means plot
    cluster_means = []
    for cluster in range(num_clusters):
        cluster_mask = data['cluster'] == cluster
        cluster_data = data_standardised[cluster_mask]
        cluster_mean = np.mean(cluster_data, axis=0)
        cluster_means.append(cluster_mean)
    
    cluster_means = np.array(cluster_means)
    
    # Plot cluster characteristics
    fig, ax = plt.subplots(figsize=(12, 8))
    x_pos = np.arange(len(columns_to_cluster))
    
    for i in range(num_clusters):
        ax.plot(x_pos, cluster_means[i], marker='o', linewidth=2, 
                label=f'Cluster {i}', markersize=5)
    
    ax.set_xlabel('Variables')
    ax.set_ylabel('Standardised Mean Values')
    ax.set_title('Cluster Characteristics (Standardised Values)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(columns_to_cluster, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'cluster_characteristics_{num_clusters}_clusters.png', dpi=300, bbox_inches='tight')
    #plt.show()

def explore_cluster_characteristics(data, columns_to_cluster, num_clusters):
    """Explore and visualize what data characteristics inform each cluster."""
    
    # Calculate cluster statistics
    cluster_stats = {}
    for cluster in range(num_clusters):
        cluster_data = data[data['cluster'] == cluster]
        stats = {}
        for col in columns_to_cluster:
            stats[col] = {
                'mean': cluster_data[col].mean(),
                'std': cluster_data[col].std(),
                'median': cluster_data[col].median(),
                'min': cluster_data[col].min(),
                'max': cluster_data[col].max()
            }
        cluster_stats[cluster] = stats
    
    # Create box plots for each variable by cluster
    n_cols = len(columns_to_cluster)
    
    # Force single row layout for 4 variables
    if n_cols == 4:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=False)
    else:
        # Keep original logic for other numbers of variables
        fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 4), sharey=False)
    
    if n_cols == 1:
        axes = [axes]
    
    for i, col in enumerate(columns_to_cluster):
        # Prepare data for box plot
        cluster_data_list = []
        cluster_labels = []
        for cluster in range(num_clusters):
            cluster_subset = data[data['cluster'] == cluster][col].dropna()
            if len(cluster_subset) > 0:
                cluster_data_list.append(cluster_subset)
                cluster_labels.append(f'C{cluster}')
        
        # Create box plot
        bp = axes[i].boxplot(cluster_data_list, labels=cluster_labels, patch_artist=True)
        
        # Color the boxes
        colors = sns.color_palette('deep', n_colors=num_clusters)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[i].set_title(f'{col} by Cluster')
        axes[i].set_xlabel('Cluster')
        if i == 0:
            axes[i].set_ylabel('Value')
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'cluster_boxplots_{num_clusters}_clusters.png', dpi=300, bbox_inches='tight')
    #plt.show()
    
    # Create a heatmap of cluster means (normalized)
    cluster_means_df = pd.DataFrame()
    for cluster in range(num_clusters):
        cluster_data = data[data['cluster'] == cluster]
        means = cluster_data[columns_to_cluster].mean()
        cluster_means_df[f'Cluster {cluster}'] = means
    
    # Normalize the means for better visualization
    cluster_means_normalized = cluster_means_df.div(cluster_means_df.max(axis=1), axis=0)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(cluster_means_normalized, annot=True, cmap='RdYlBu_r', 
                center=0.5, fmt='.2f', cbar_kws={'label': 'Normalized Mean Value'})
    plt.title(f'Cluster Characteristics Heatmap (Normalized Means)')
    plt.ylabel('Variables')
    plt.xlabel('Clusters')
    plt.tight_layout()
    plt.savefig(f'cluster_heatmap_{num_clusters}_clusters.png', dpi=300, bbox_inches='tight')
    #plt.show()
    
    # Print detailed cluster statistics
    print(f"\n{'='*60}")
    print(f"DETAILED CLUSTER ANALYSIS ({num_clusters} clusters)")
    print(f"{'='*60}")
    
    for cluster in range(num_clusters):
        cluster_data = data[data['cluster'] == cluster]
        print(f"\nCluster {cluster} ({len(cluster_data)} fires, {len(cluster_data)/len(data)*100:.1f}%):")
        print("-" * 50)
        
        for col in columns_to_cluster:
            mean_val = cluster_data[col].mean()
            std_val = cluster_data[col].std()
            
            # Compare to overall mean
            overall_mean = data[col].mean()
            diff_from_overall = mean_val - overall_mean
            percent_diff = (diff_from_overall / overall_mean) * 100 if overall_mean != 0 else 0
            
            print(f"{col:8}: {mean_val:8.2f} ± {std_val:6.2f} "
                  f"({percent_diff:+6.1f}% from overall)")
    
    # Identify most distinctive features for each cluster
    print(f"\n{'='*60}")
    print("MOST DISTINCTIVE FEATURES BY CLUSTER")
    print(f"{'='*60}")
    
    overall_means = data[columns_to_cluster].mean()
    
    for cluster in range(num_clusters):
        cluster_data = data[data['cluster'] == cluster]
        cluster_means = cluster_data[columns_to_cluster].mean()
        
        # Calculate relative differences
        relative_diffs = {}
        for col in columns_to_cluster:
            if overall_means[col] != 0:
                rel_diff = abs((cluster_means[col] - overall_means[col]) / overall_means[col])
                relative_diffs[col] = rel_diff
        
        # Sort by most distinctive (largest relative difference)
        sorted_features = sorted(relative_diffs.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nCluster {cluster} - Most distinctive features:")
        for i, (feature, rel_diff) in enumerate(sorted_features[:3]):  # Top 3
            direction = "higher" if cluster_means[feature] > overall_means[feature] else "lower"
            print(f"  {i+1}. {feature}: {rel_diff*100:.1f}% {direction} than average")

def plot_clusters_subplots(data, num_clusters):
    """Create a figure with subplots showing individual clusters on maps."""
    
    # Force single row layout for 4 clusters
    if num_clusters == 4:
        ncols = 4
        nrows = 1
    else:
        # Keep original logic for other cluster numbers
        if num_clusters <= 6:
            ncols = 3
            nrows = 2
        else:
            ncols = 3
            nrows = int(np.ceil(num_clusters / 3))
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 5) if num_clusters == 4 else (15, 5*nrows))
    
    cluster_colors = sns.color_palette('deep', n_colors=num_clusters)
    
    for cluster_id in range(num_clusters):
        # Create subplot with map projection
        ax = plt.subplot(nrows, ncols, cluster_id + 1, projection=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.LAND, alpha=0.2, color='lightgray')
        ax.add_feature(cfeature.OCEAN, alpha=0.3, color='lightblue')
        
        # Plot current cluster
        cluster_data = data[data['cluster'] == cluster_id]
        fire_count = len(cluster_data)
        fire_percent = (fire_count / len(data)) * 100
        
        ax.scatter(cluster_data['Lon'], cluster_data['Lat'], 
                  c=[cluster_colors[cluster_id]], s=8, alpha=0.7,
                  transform=ccrs.PlateCarree())
        
        # Set extent to New Zealand
        ax.set_extent([165, 180, -48, -34], crs=ccrs.PlateCarree())
        
        # Add gridlines (simplified for subplots)
        ax.gridlines(alpha=0.3)
        
        # Set title with cluster info
        ax.set_title(f'Cluster {cluster_id}\n({fire_count} fires, {fire_percent:.1f}%)', 
                    fontsize=10)
    
    # Hide any unused subplots
    total_subplots = nrows * ncols
    for i in range(num_clusters, total_subplots):
        ax = plt.subplot(nrows, ncols, i + 1)
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'individual_clusters_subplots_{num_clusters}_clusters.png', dpi=300, bbox_inches='tight')
    #plt.show()

def plot_cluster_boxplots_subplots(data, columns_to_cluster, num_clusters):
    """Create a figure with subplots showing boxplots for each cluster."""
    
    # Force single row layout for 4 clusters
    if num_clusters == 4:
        ncols = 4
        nrows = 1
    else:
        # Keep original logic for other cluster numbers
        if num_clusters <= 6:
            ncols = 3
            nrows = 2
        else:
            ncols = 3
            nrows = int(np.ceil(num_clusters / 3))
    
    # Create figure with subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5) if num_clusters == 4 else (15, 4*nrows), sharey=False)
    
    # Flatten axes array for easier indexing
    if nrows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()
    
    cluster_colors = sns.color_palette('deep', n_colors=num_clusters)
    
    for cluster_id in range(num_clusters):
        ax = axes_flat[cluster_id]
        
        # Get data for current cluster
        cluster_data = data[data['cluster'] == cluster_id]
        fire_count = len(cluster_data)
        fire_percent = (fire_count / len(data)) * 100
        
        # Prepare data for boxplot
        box_data = []
        valid_columns = []
        
        for col in columns_to_cluster:
            col_data = cluster_data[col].dropna()
            if len(col_data) > 0:
                box_data.append(col_data)
                valid_columns.append(col)
        
        if len(box_data) > 0:
            # Create boxplot
            bp = ax.boxplot(box_data, labels=valid_columns, patch_artist=True)
            
            # Color the boxes with cluster color and create legend handles
            legend_handles = []
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(cluster_colors[cluster_id])
                patch.set_alpha(0.7)
                # Create legend handle for each variable
                if i == 0:  # Only create one handle per cluster for the legend
                    legend_handles.append(plt.Rectangle((0,0),1,1, 
                                                      facecolor=cluster_colors[cluster_id], 
                                                      alpha=0.7,
                                                      label=f'Cluster {cluster_id}'))
            
            # Add legend to upper left (loc=2)
            if legend_handles:
                ax.legend(handles=legend_handles, loc=2, fontsize=8)
        
        # Customize subplot
        ax.set_title(f'Cluster {cluster_id}\n({fire_count} fires, {fire_percent:.1f}%)', 
                    fontsize=10)
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # Add mean values as text (moved to upper right to avoid legend overlap)
        if len(cluster_data) > 0:
            means_text = "Means:\n"
            for col in valid_columns:
                mean_val = cluster_data[col].mean()
                means_text += f"{col}: {mean_val:.1f}\n"
            
            ax.text(0.98, 0.98, means_text, transform=ax.transAxes, 
                   verticalalignment='top', horizontalalignment='right', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide any unused subplots
    total_subplots = nrows * ncols
    for i in range(num_clusters, total_subplots):
        axes_flat[i].set_visible(False)
    
    plt.suptitle(f'Cluster Boxplots by Variables ({num_clusters} clusters)', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(f'cluster_boxplots_subplots_{num_clusters}_clusters.png', dpi=300, bbox_inches='tight')
    #plt.show()

def silhouette_analysis(data_standardised, max_clusters=10):
    """
    Perform silhouette analysis to find optimal number of clusters.
    
    Parameters:
    data_standardised: Standardized data for clustering
    max_clusters: Maximum number of clusters to test
    
    Returns:
    optimal_clusters: Recommended number of clusters
    silhouette_scores: List of silhouette scores for each k
    """
    
    silhouette_scores = []
    cluster_range = range(2, max_clusters + 1)
    
    print(f"\n{'='*60}")
    print("SILHOUETTE ANALYSIS FOR OPTIMAL CLUSTER NUMBER")
    print(f"{'='*60}")
    
    for n_clusters in cluster_range:
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        cluster_labels = kmeans.fit_predict(data_standardised)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(data_standardised, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        
        print(f"For {n_clusters} clusters: Average silhouette score = {silhouette_avg:.4f}")
    
    # Find optimal number of clusters
    optimal_idx = np.argmax(silhouette_scores)
    optimal_clusters = cluster_range[optimal_idx]
    best_score = silhouette_scores[optimal_idx]
    
    print(f"\nOptimal number of clusters: {optimal_clusters} (silhouette score: {best_score:.4f})")
    
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
    plt.savefig('silhouette_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return optimal_clusters, silhouette_scores

def detailed_silhouette_analysis(data_standardised, n_clusters):
    """
    Perform detailed silhouette analysis for a specific number of clusters.
    Shows silhouette plot with individual sample scores.
    
    Parameters:
    data_standardised: Standardized data for clustering
    n_clusters: Number of clusters to analyze
    """
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    cluster_labels = kmeans.fit_predict(data_standardised)
    
    # Calculate silhouette scores
    silhouette_avg = silhouette_score(data_standardised, cluster_labels)
    sample_silhouette_values = silhouette_samples(data_standardised, cluster_labels)
    
    print(f"\n{'='*60}")
    print(f"DETAILED SILHOUETTE ANALYSIS FOR {n_clusters} CLUSTERS")
    print(f"{'='*60}")
    print(f"Average silhouette score: {silhouette_avg:.4f}")
    
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
    plt.savefig(f'detailed_silhouette_analysis_{n_clusters}_clusters.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Identify poorly clustered samples
    poor_samples = np.where(sample_silhouette_values < 0)[0]
    if len(poor_samples) > 0:
        print(f"\nPoorly clustered samples (negative silhouette): {len(poor_samples)} "
              f"({len(poor_samples)/len(sample_silhouette_values)*100:.1f}%)")
        
        # Show distribution of poor samples by cluster
        poor_by_cluster = {}
        for sample_idx in poor_samples:
            cluster = cluster_labels[sample_idx]
            poor_by_cluster[cluster] = poor_by_cluster.get(cluster, 0) + 1
        
        print("Poor samples by cluster:")
        for cluster, count in sorted(poor_by_cluster.items()):
            cluster_size = np.sum(cluster_labels == cluster)
            percentage = (count / cluster_size) * 100
            print(f"  Cluster {cluster}: {count}/{cluster_size} ({percentage:.1f}%)")
    else:
        print("\nAll samples have positive silhouette scores - good clustering!")
    
    return silhouette_avg, sample_silhouette_values

def compare_clustering_methods(data_standardised, n_clusters):
    """
    Compare different clustering methods using silhouette scores.
    """
    from sklearn.cluster import DBSCAN, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    
    print(f"\n{'='*60}")
    print(f"COMPARING CLUSTERING METHODS ({n_clusters} clusters)")
    print(f"{'='*60}")
    
    methods = {}
    
    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    kmeans_labels = kmeans.fit_predict(data_standardised)
    kmeans_score = silhouette_score(data_standardised, kmeans_labels)
    methods['K-Means'] = (kmeans_labels, kmeans_score)
    
    # Hierarchical Clustering
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    hier_labels = hierarchical.fit_predict(data_standardised)
    hier_score = silhouette_score(data_standardised, hier_labels)
    methods['Hierarchical'] = (hier_labels, hier_score)
    
    # Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_clusters, random_state=0)
    gmm_labels = gmm.fit_predict(data_standardised)
    gmm_score = silhouette_score(data_standardised, gmm_labels)
    methods['Gaussian Mixture'] = (gmm_labels, gmm_score)
    
    # Print comparison
    print("Method Comparison (Silhouette Scores):")
    print("-" * 40)
    for method, (labels, score) in methods.items():
        print(f"{method:18}: {score:.4f}")
    
    # Find best method
    best_method = max(methods.items(), key=lambda x: x[1][1])
    print(f"\nBest method: {best_method[0]} (score: {best_method[1][1]:.4f})")
    
    # Visualize comparison
    fig, axes = plt.subplots(1, len(methods), figsize=(5*len(methods), 4))
    if len(methods) == 1:
        axes = [axes]
    
    for i, (method, (labels, score)) in enumerate(methods.items()):
        colors = plt.cm.nipy_spectral(labels.astype(float) / n_clusters)
        axes[i].scatter(data_standardised[:, 0], data_standardised[:, 1], 
                       c=colors, alpha=0.7, s=30)
        axes[i].set_title(f'{method}\nSilhouette: {score:.4f}')
        axes[i].set_xlabel('First Standardized Feature')
        if i == 0:
            axes[i].set_ylabel('Second Standardized Feature')
    
    plt.tight_layout()
    plt.savefig(f'clustering_methods_comparison_{n_clusters}_clusters.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return methods

if __name__ == "__main__":
    main()
