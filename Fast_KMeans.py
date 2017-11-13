import time
from KMeans import *


def init_assignments_and_bounds(data, centers):
    k = len(centers)
    assignments = []
    lbs = [[0 for j in range(len(centers))] for i in range(len(data))]
    ubs = []
    for i in range(len(data)):
        min_index = 0
        min_value = distance(data[i], centers[0])
        for j in range(1, k):
            new_distance = distance(data[i], centers[j])
            lbs[i][j] = new_distance
            if new_distance < min_value:
                min_index = j
                min_value = new_distance
        assignments.append(min_index)
        ubs.append(min_value)

    return assignments, lbs, ubs


def compute_center_distances(centers):
    center_distances = [[distance(centers[i], centers[j]) for j in range(len(centers))] for i in range(len(centers))]
    min_distances = [min(row) for row in center_distances]
    return center_distances, min_distances


def fast_k_means(data, k):
    centers = init_centers(data, k)
    assignments, lbs, ubs = init_assignments_and_bounds(data, centers)
    old_centers = []
    out_of_date = [True for i in range(len(data))]
    while not compare_centers(centers, old_centers):
        center_distances, min_distances = compute_center_distances(centers)
        for i in range(len(data)):
            if ubs[i] > min_distances[assignments[i]]:
                for j in range(k):
                    if j != assignments[i] and ubs[i] > lbs[i][j] and ubs[i] > 0.5 * center_distances[assignments[i]][j]:
                        d = ubs[i]
                        if out_of_date[i]:
                            d = distance(data[i], centers[assignments[i]])
                            out_of_date[i] = False

                        if d > lbs[i][j] or d > 0.5 * center_distances[assignments[i]][j]:
                            if distance(data[i], centers[j]) < d:
                                assignments[i] = j

        centroids = find_centroids(data, assignments, k)
        for i in range(len(data)):
            for j in range(len(centers)):
                lbs[i][j] = max(lbs[i][j] - distance(centers[j], centroids[j]), 0)
            ubs[i] = ubs[i] + distance(centers[assignments[i]], centroids[assignments[i]])

        out_of_date = [True for i in range(len(data))]
        old_centers = centers
        centers = centroids

    return centers, assignments


def test_time(algorithm, data, k):
    start = time.time()
    centers, assignments = algorithm(data, k)
    end = time.time()
    print(end - start)


# data = np.random.randint(-32, 32, size=(10000, 10))
data = np.random.rand(10000, 10)
test_time(fast_k_means, data, 10)
# test_time(k_means, data, 10)
