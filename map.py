import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read the CSV file
df = pd.read_csv('FOD_FENZ_updated2024-07-11(in).csv')

# Display basic information about the dataset
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows of Lat/Lon data:")
print(df[['Lat', 'Lon']].head())

# Check for missing values in Lat/Lon columns
print(f"\nMissing values in Lat: {df['Lat'].isna().sum()}")
print(f"Missing values in Lon: {df['Lon'].isna().sum()}")

# Remove rows with missing Lat/Lon values
df_clean = df.dropna(subset=['Lat', 'Lon'])
print(f"Rows after removing missing Lat/Lon: {len(df_clean)}")

# Filter out negative longitudes (data errors for New Zealand)
df_clean = df_clean[df_clean['Lon'] > 0]
print(f"Rows after removing negative longitudes: {len(df_clean)}")

# Create the plot
plt.figure(figsize=(12, 8))
plt.scatter(df_clean['Lon'], df_clean['Lat'], alpha=0.6, s=10, c='red', edgecolors='none')

# Customize the plot
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Fire Incidents in New Zealand - Geographic Distribution')
plt.grid(True, alpha=0.3)

# Set aspect ratio to be equal for better geographic representation
plt.axis('equal')

# Add some basic statistics to the plot
plt.text(0.02, 0.98, f'Total incidents: {len(df_clean)}', 
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

# Display coordinate ranges
print(f"\nCoordinate ranges:")
print(f"Latitude: {df_clean['Lat'].min():.4f} to {df_clean['Lat'].max():.4f}")
print(f"Longitude: {df_clean['Lon'].min():.4f} to {df_clean['Lon'].max():.4f}")



