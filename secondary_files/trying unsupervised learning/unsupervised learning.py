<<<<<<< HEAD
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the dataset with features (after feature extraction)
df = pd.read_csv('scaled_features.csv')

# Separate the features (X) and the target labels (y)
X = df.drop('Genre', axis=1).values  # Features only, exclude the Genre column

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means Clustering (assume 5 clusters for the 5 genres)
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)

# Get the predicted cluster labels
clusters = kmeans.labels_

# Add the cluster labels to the original dataframe
df['Cluster'] = clusters

# Visualizing the clustering result using 2D PCA (Principal Component Analysis)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the results
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
plt.title('K-Means Clustering of Music Genres (Unsupervised)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()

# Save the updated DataFrame with clusters to a new CSV file
df.to_csv('classified_music_genres.csv', index=False)

# Print the first few rows of the updated DataFrame
print(df.head())
=======
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the dataset with features (after feature extraction)
df = pd.read_csv('scaled_features.csv')

# Separate the features (X) and the target labels (y)
X = df.drop('Genre', axis=1).values  # Features only, exclude the Genre column

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means Clustering (assume 5 clusters for the 5 genres)
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)

# Get the predicted cluster labels
clusters = kmeans.labels_

# Add the cluster labels to the original dataframe
df['Cluster'] = clusters

# Visualizing the clustering result using 2D PCA (Principal Component Analysis)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the results
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
plt.title('K-Means Clustering of Music Genres (Unsupervised)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()

# Save the updated DataFrame with clusters to a new CSV file
df.to_csv('classified_music_genres.csv', index=False)

# Print the first few rows of the updated DataFrame
print(df.head())
>>>>>>> d611a1fd00bf27a9e031616257a59f9cf9a936fa
