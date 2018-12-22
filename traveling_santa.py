import numpy as np
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.special import comb
from scipy import stats
from sympy.ntheory.primetest import isprime
import sys
import random
import math
import time
import operator
import pandas as pd

from sklearn.cluster import DBSCAN
from tsp_solver.greedy import solve_tsp
from tsp_solver.greedy import optimize_solution
from scipy.spatial import distance_matrix


# Load the prime numbers we need in a set with the Sieve of Eratosthenes
def eratosthenes(n):
    P = [True for i in range(n+1)]
    P[0], P[1] = False, False
    p = 2
    l = np.sqrt(n)
    while p < l:
        if P[p]:
            for i in range(2*p, n+1, p):
                P[i] = False
        p += 1
    return P

def load_primes(n):
    return set(np.argwhere(eratosthenes(n)).flatten())

full_cities = pd.read_csv('./cities.csv')

# prime_cities is a set of prime cities
prime_cities = load_primes(full_cities.shape[0])

# so it actually plots path
matplotlib.rcParams['agg.path.chunksize'] = 10000



def plot_cities(cities):
    plt.plot(cities[:,0], cities[:,1], 'o', label='data')
    plt.axis('off')
    plt.show()

# Path plotting
def plot_path(cities, path, north_pole = None):
    coords = cities
    ordered_coords = coords[np.array(path)]
    codes = [Path.MOVETO] * len(ordered_coords)
    path = Path(ordered_coords, codes)
    
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)
    xs = ordered_coords[:,0]
    ys = ordered_coords[:,1]
    ax.plot(xs, ys,  lw=1., ms=10, c='blue')
    plt.axis('off')

    plt.scatter(coords[north_pole,0], coords[north_pole,1], c='red', s=10)
    
    plt.show()

def plot_path2(cities, path):
    coords = cities[['X', 'Y']].values
    ordered_coords = coords[np.array(path)]
    codes = [Path.MOVETO] * len(ordered_coords)
    path = Path(ordered_coords, codes)
    
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)
    xs, ys = zip(*ordered_coords)
    ax.plot(xs, ys,  lw=1., ms=10, c='blue')
    plt.axis('off')
    
    north_pole = cities[cities.CityId==0]
    plt.scatter(north_pole.X, north_pole.Y, c='red', s=10)
    
    plt.show()

def total_distance(cities, path):
    coord = cities[['X', 'Y']].values
    score = 0
    for i in range(1, len(path)):
        begin = path[i-1]
        end = path[i]
        distance = np.linalg.norm(coord[end] - coord[begin])
        if i%10 == 0:
            if begin not in prime_cities:
                distance *= 1.1
        score += distance
    return score

# Count the number of prime numbers reached
def count_primes_path(path):
    mask_iter = np.array([True if (i+1)%10==0 else False for i in range(len(path))])
    mask_primes = np.isin(path, list(prime_cities))
    return np.sum(mask_iter & mask_primes)


def path_to_connections(path):
    connections = []

    for i in range(len(path)):
        if i == 0:
            connections.append([path[1]])
            continue
        if i == len(path) - 1:
            connections.append([path[i-1]])
            continue

        connections.append([path[i-1], path[i+1]])

    return np.array(connections)    

# Greedy algorithm starting at startpoint and ending at endpoint
# cities is np.array of all city locations
def greedy_inner_cluster(cities, startpoint, endpoint, inner_cluster_index_to_actual_index, step_num):
    current = cities[startpoint]
    inner_path = [startpoint]
    
    visited_cities = set()
    visited_cities.add(startpoint)

    it = 0
    
    start = time.perf_counter()

    while len(inner_path) != len(cities):
        next = find_closest_point(cities, current, visited_cities, True, inner_cluster_index_to_actual_index, step_num)
        
        inner_path.append(next)
        visited_cities.add(next)
        current = cities[next]
        
        it += 1
        step_num += 1
        
        if it%1000 == 0:
            print('{} iterations, {} remaining cities, {} s elapsed.'.format(it, len(cities) - len(inner_path), time.perf_counter() - start), flush=True)
    
    return inner_path, step_num


# data is np.array of all points
# point is the coordinates of point to find closest point to
# ignore can be a set of points to not consider
def find_closest_point(data, point, ignore=None, promote_primes=False, inner_cluster_index_to_actual_index=None, step_num=None):
    min_dist = float('inf')
    closest_point = -1
    for i in range(len(data)):
        if ignore != None and i in ignore:
            continue
        dist = np.linalg.norm(data[i,:] - point)

        if step_num is not None and step_num % 10 == 0 and promote_primes:
            if inner_cluster_index_to_actual_index[i] not in prime_cities: 
                dist = dist * 1.1

        if dist < min_dist:
            min_dist = dist
            closest_point = i
    return closest_point

def cluster_path():
    ID = full_cities.CityId.values
    coord = full_cities[['X', 'Y']].values

    read_from_file = False
    if len(sys.argv) > 1:
        read_from_file = True

    # read cluster assignments from file
    if read_from_file:
        cluster_assignment_file = open(sys.argv[1], 'r')
        clusters = {}
        cluster_means = []
        cluster_starts = []
        cluster_ends = []
        clusters[-1] = set()
        current_cluster = set()
        index = 0
        for line in cluster_assignment_file:
            line = line.strip('\n')
            if line[0] == '*':
                line = line.split()
                clusters[index] = current_cluster
                cluster_means.append(np.array([float(line[1]), float(line[2])]))
                cluster_starts.append(int(line[3]))
                cluster_ends.append(int(line[4]))
                current_cluster = set()
                index += 1
                continue
            current_cluster.add(int(line))
            if int(line) == 0:
                north_pole_cluster = index

        cluster_means = np.array(cluster_means)

    else:
        clustering = DBSCAN(eps=6, min_samples=2).fit(coord)
        print(clustering, flush=True)
        labels = clustering.labels_ # -1 is noisy point, has the cluster label for every point

        core_samples = clustering.core_sample_indices_# the points where there are >= min_samples within eps
        components = clustering.components_ # just the X and Y values of each core sample
        # multiple core samples can be in the same cluster
        # each cluster is made up of all points reachable from each core sample
        # where reachable means there's a series of core points it can jump between to get there

        print("num core samples", len(core_samples), flush=True)

        # get the clusters
        clusters = {}
        for i in range(len(labels)):
            label = labels[i]
            if label not in clusters:
                clusters[label] = set()
            clusters[label].add(i)

            if i == 0:
                north_pole_cluster = label

    print("num clusters", len(clusters), flush=True)
    print("largest cluster size", max(len(j) for (i,j) in clusters.items() if i != -1), flush=True)
        
    # get the cluster means
    if not read_from_file:
        cluster_means = []
        cluster_labels = sorted(clusters.keys())
        cluster_labels.remove(-1)
        for i in cluster_labels:
            i_mean = np.array([0,0], dtype="float64")
            clust = clusters[i]
            for j in clust:
                i_mean += coord[j]
            i_mean = i_mean / len(clust)
            cluster_means.append(i_mean)
        cluster_means = np.array(cluster_means)

    print("num noisy", len(clusters[-1]), flush=True)

    # assign each noisy point to its closest cluster
    if not read_from_file:
        start_noisy = time.perf_counter()
        index = 0
        for i in clusters[-1]:
            if index % 1000 == 0:
                print("outlier", index, '/', len(clusters[-1]), time.perf_counter() - start_noisy, flush=True)
            index += 1
            closest_cluster = find_closest_point(cluster_means, coord[i])
            clusters[closest_cluster].add(i)

            if i == 0:
                north_pole_cluster = closest_cluster

    # get the closest cluster to the north pole cluster
    if not isinstance(north_pole_cluster, int):
        north_pole_cluster = north_pole_cluster.item()
    closest_to_north_pole = find_closest_point(cluster_means, cluster_means[north_pole_cluster,:], set([north_pole_cluster]))

    # Compute the distance matrix
    dist_matrix = distance_matrix(cluster_means, cluster_means)

    # solve TSP for cluster means
    cluster_path_unopt = solve_tsp(dist_matrix, endpoints=(north_pole_cluster, closest_to_north_pole))
    print("unopt done", flush=True)
    cluster_path = optimize_solution(dist_matrix, path_to_connections(cluster_path_unopt), endpoints=(north_pole_cluster, closest_to_north_pole))
    print("opt done", flush=True)

    plot_path(cluster_means, cluster_path_unopt, north_pole_cluster)
    plot_path(cluster_means, cluster_path, north_pole_cluster)

    # get cluster start and end points
    if not read_from_file:
        cluster_starts = []
        cluster_ends = []
        # for each (cluster, next_cluster) in cluster_path, get the closest pair of end and start points between clusters
        # cluster_ends[i] is the index of the point in cluster_path[i] to end at
        # cluster_starts[i] is the index of the point in cluster_path[i] to start at
        # the last cluster_end will be the closest point to the north pole in closest_to_north_pole cluster
        for c in range(len(cluster_path)):
            print("getting cluster {} start and end".format(c), flush=True)

            clust_prev = cluster_path[c]
            if c == 0:
                cluster_starts.append(0)

            if c == len(cluster_path) - 1:
                min_dist = float('inf')
                final_end = None
                for x_i in clusters[clust_prev]:
                    dist = np.linalg.norm(coord[0,:] - coord[x_i,:])

                    if dist < min_dist:
                        min_dist = dist
                        final_end = x_i

                cluster_ends.append(final_end)
                continue

            clust_next = cluster_path[c+1]
            min_dist = float('inf')
            prev_end = None
            next_start = None
            for x_i in clusters[clust_prev]:
                if cluster_starts[-1] == x_i and len(clusters[clust_prev]) != 1:
                    # can't end where the cluster started (unless it only has 1 point)
                    continue
                for x_j in clusters[clust_next]:
                    dist = np.linalg.norm(coord[x_j,:] - coord[x_i,:])

                    if dist < min_dist:
                        min_dist = dist
                        prev_end = x_i
                        next_start = x_j

            cluster_ends.append(prev_end)
            cluster_starts.append(next_start)
    
    # output cluster assignments
    if not read_from_file:
        output_clusters_file = open('cluster_assignment.txt', 'w+')
        for i, c in clusters.items():
            if i == -1:
                continue
            for j in c:
                output_clusters_file.write(str(j) + "\n")

            index_of_cluster_i_in_cluster_path = cluster_path.index(i)

            output_clusters_file.write('*' + '\t' + str(cluster_means[i][0]) + '\t' + str(cluster_means[i][1]) + '\t' + str(cluster_starts[index_of_cluster_i_in_cluster_path]) + '\t' + str(cluster_ends[index_of_cluster_i_in_cluster_path]) + '\n')
            output_clusters_file.flush()

    path = []

    step_num = 1
    # for each cluster, greedily solve TSP starting and ending at start and end points
    for c in range(len(cluster_path)):
        print("cluster TSP", c, flush=True)
        cluster = clusters[cluster_path[c]]
        if len(cluster) == 1:
            path.append(cluster[0])
            step_num += 1
            continue

        cluster_points = []
        startpoint = None
        endpoint = None
        index = 0
        inner_cluster_index_to_actual_index = {}
        if cluster_starts[c] not in cluster or cluster_ends[c] not in cluster:
            print(cluster_starts[c])
            print(cluster_ends[c])
            print(cluster, flush=True)
        assert cluster_starts[c] in cluster
        assert cluster_ends[c] in cluster
        for point_index in cluster:
            cluster_points.append(coord[point_index,:])
            inner_cluster_index_to_actual_index[index] = point_index
            if point_index == cluster_starts[c]:
                startpoint = index
            elif point_index == cluster_ends[c]:
                endpoint = index
            index += 1
        cluster_points = np.array(cluster_points)

        if startpoint is None or endpoint is None:
            print(startpoint, endpoint, cluster_starts[c], cluster_ends[c], flush=True)

        assert startpoint is not None
        assert endpoint is not None

        if len(cluster_points) > 10000:
            print("len {}, solving gredily".format(len(cluster_points)), flush=True)
            inner_cluster_path, step_num = greedy_inner_cluster(cluster_points, startpoint, endpoint, inner_cluster_index_to_actual_index, step_num)
            path.extend([inner_cluster_index_to_actual_index[i] for i in inner_cluster_path])
        else:
            print("len {}, solving TSP".format(len(cluster_points)), flush=True)
            dist_matrix = distance_matrix(cluster_points, cluster_points)
            inner_cluster_path = solve_tsp(dist_matrix, endpoints=(startpoint, endpoint))
            step_num += len(inner_cluster_path)
            path.extend([inner_cluster_index_to_actual_index[i] for i in inner_cluster_path])


    # for each cluster, solve TSP starting and ending at start and end points
    # too memory intensive for cluster sizes unfortunately
    # for c in range(len(cluster_path)):
    #     print("cluster TSP", c, flush=True)
    #     cluster = clusters[cluster_path[c]]
    #     if len(cluster) == 1:
    #         continue

    #     cluster_points = []
    #     startpoint = None
    #     endpoint = None
    #     index = 0
    #     inner_cluster_index_to_actual_index = {}
    #     for point_index in cluster:
    #         cluster_points.append(coord[point_index,:])
    #         inner_cluster_index_to_actual_index[index] = point_index
    #         if point_index == cluster_starts[c]:
    #             startpoint = index
    #         elif point_index == cluster_ends[c]:
    #             endpoint = index
    #         index += 1
    #     cluster_points = np.array(cluster_points)
        
    #     dist_matrix = distance_matrix(cluster_points, cluster_points)
    #     inner_cluster_path = solve_tsp(dist_matrix, endpoints=(startpoint, endpoint))

    #     path.extend([inner_cluster_index_to_actual_index[i] for i in inner_cluster_path])

    # return to 0 at end
    path.append(0)
    return path


# changes path to try and hit primes every 10 steps
# def optimize_path_for_primes(path):
    # every ten steps...
        # if path not already at a prime city, find the nearest prime city
        # if 



if __name__ == '__main__':
    # plot_cities(full_cities)

	# dumbest
    # path = list(range(full_cities.shape[0])) + [0]

    # clustering
    path = cluster_path()

    # read path from csv
    # path = pd.read_csv('./path_clusters.csv')
    # path = [i[0] for i in path.values.tolist()]

    output_file = open('path.txt', 'w+')
    output_file.write("Path\n")
    for city in path:
        output_file.write(str(city)+'\n')
    output_file.flush()
    output_file.close()

    print("Distance of path:" , total_distance(full_cities, path), flush=True)
    print("Num primes hit:", count_primes_path(path), " out of ", len(prime_cities), flush=True)

    plot_path2(full_cities, path)