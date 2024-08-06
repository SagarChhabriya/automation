
## [Pipeling with Pipeline object](/ML/pipeline.ipynb)
## Pipelining Using make_pipeline function

```python
# Perform the necessary imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScalar()
from sklearn.cluster import KMeans

# Create scaler: scaler
scaler = StandardScalar()

# Create KMeans instance: kmeans
kmeans = KMeans(n_cluster=4)

# Create pipeline: pipeline
pipeline = make_pipeline(scaler,kmeans)

```