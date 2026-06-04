# Import necessary libraries
from sklearn.manifold import TSNE
import numpy as np

# Sample data (eg points in high-dimensiona space)
X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [5, 6, 7], [5, 7, 8], [8, 9, 10]])

# Initialise and fit model
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
X_reduced = tsne.fit_transform(X)

print("Reduced Data:\n", X_reduced)
