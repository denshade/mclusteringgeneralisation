import numpy

from sklearn.cluster import KMeans

k = 2 # k anonymization on
nrElements = 10

# Generate random example values
vals = numpy.floor((numpy.random.rand(nrElements, 1) * 70) + 18).astype(int)

kmeans = KMeans(nrElements//k).fit(vals)
clusters = kmeans.cluster_centers_
clusters.sort(axis=0)
converted = list(map(lambda y: int(min(clusters, key=lambda x: abs(x-y))[0]), vals))

print("Example values:")
print(vals.tolist())
print("Clusters:")
print(clusters.tolist())
print("Converted example:")
print(converted)
