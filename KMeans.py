import math, random
import numpy as np
import matplotlib.pyplot as plt


def distance(point_1, point_2):
    return np.sum((np.asarray(point_1) - np.asarray(point_2)) ** 2)


def init_centers(data, k):
    dim = len(data[0])
    min_dict = {}
    max_dict = {}
    for point in data:
        for i in range(dim):
            max_dict[i] = point[i] if (i not in max_dict or point[i] > max_dict[i]) else max_dict[i]
            min_dict[i] = point[i] if (i not in min_dict or point[i] < min_dict[i]) else min_dict[i]

    centers = []
    for i in range(k):
        center_point = []
        for j in range(dim):
            center_point.append(random.uniform(min_dict[j], max_dict[j]))
        centers.append(center_point)

    return centers


def assign_centers(data, centers):
    k = len(centers)
    assignments = []
    for point in data:
        min_index = 0
        min_value = distance(point, centers[0])
        for j in range(1, k):
            new_distance = distance(point, centers[j])
            if new_distance < min_value:
                min_index = j
                min_value = new_distance
        assignments.append(min_index)

    return assignments


def find_centroids(data, assignments, k):
    centroids = {}
    nums = {}
    for i in range(len(data)):
        if assignments[i] not in centroids:
            centroids[assignments[i]] = data[i]
            nums[assignments[i]] = 1
        else:
            centroids[assignments[i]] = (np.asarray(centroids[assignments[i]]) + np.asarray([data[i]])).tolist()
            nums[assignments[i]] += 1

    output = []
    for i in range(k):
        if i not in centroids:
            output.append([0 for j in range(len(data[0]))])
        else:
            centroids[i] = (np.asarray(centroids[i]) / nums[i]).tolist()
            output.append(centroids[i])

    return output


def compare_centers(centers_1, centers_2):
    if len(centers_1) != len(centers_2):
        return 0
    length = len(centers_1)
    same = 0
    for i in range(length):
        same_for_i = 0
        for j in range(length):
            if centers_1[i] == centers_2[j]:
                same_for_i = 1
                break
        same = same or same_for_i
    return same


def k_means(data, k):
    centers = init_centers(data, k)
    old_centers = []
    while not compare_centers(centers, old_centers):
        assignments = assign_centers(data, centers)
        old_centers = centers
        centers = find_centroids(data, assignments, k)

    return centers, assign_centers(data, centers)

# data = np.random.randint(-8, 8, size=(70, 2))
# centers, assignments = k_means(data, 4)
# color_map = ['red', 'green', 'yellow', 'blue']
# colors = [color_map[assignment] for assignment in assignments]
# plt.scatter(np.asarray(data)[:, 0], np.asarray(data)[:, 1], c=colors, edgecolors='none')
# plt.show()
