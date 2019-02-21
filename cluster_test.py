
import collections
import math
import json
import random
import numpy as np
import score_object
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.cluster import homogeneity_score
from scipy.stats import gamma, norm

report = {}

def generate_easily_clusterable_data(num_samples_per_cluster=None,
                                     n_clusters=None,
                                     n_dimensions=None,
                                     distrib=None,
                                     mu_inc=None,
                                     sigma=None):
    """
    Returns np array of shape (NUM_SAMPLES * n_clusters) X (n_dimensions + 1)
    Additional dimension is the label (index 0 to n_clusters - 1).
    """
    clusters_mus, clusters_sigmas = [], []
    start_mu, mu_increment = 0, mu_inc
    for i in range(n_clusters):
        clusters_mus.append(start_mu + i * mu_increment)

        # low standard deviation for easy distinction between distributions.
        clusters_sigmas.append(sigma)

    num_rounds = num_samples_per_cluster
    data = []

    print("Generating data with parameters: ")
    print("Cluster mus: ", clusters_mus)
    print("Cluster sigmas: ", clusters_sigmas)

    for _ in range(num_rounds):
        new_clusters_points = [[] for _ in range(n_clusters)]
        for i in range(n_clusters):
            cluster_i_sample = new_clusters_points[i]
            # create n_dimensions-dimensional vector/sample for ith cluster.
            for _ in range(n_dimensions):
                # using normal distribution
                x = norm.rvs(clusters_mus[i], clusters_sigmas[i])
                cluster_i_sample.append(x)
            # take note of what cluster this sample belongs to.
            cluster_i_sample.append(int(i))
            new_clusters_points[i] = cluster_i_sample

        for point in new_clusters_points:
            data.append(point)

    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    return np.array(data), np.array(shuffled_data)


def analyse_result_class_by_class(actual_labels, predicted_labels):
    pass


def main():
    global report
    # define constant parameters
    n_clusters, n_dimensions, n_samples_per_class = 3, 1000, 200
    start_mu, start_sigma = 0, 1
    num_results, NUM_RESULTS_CAP = 0, 2

    # define variable parameteres
    distributions = ["normal", "gamma"]
    mu_increments = [1, 0.5, 0.25, 0.05, 0.01]
    sigma_increments = [0.5, .25, 0.05, 0.01, 0.001]
    report = dict(zip(sigma_increments, []))
    score_encoder = score_object.ScoreEncoder()

    done = False
    for dist in distributions:
        if done:
            break
        for mu_increment in mu_increments:
            if done:
                break
            curr_sigma = start_sigma
            for sigma_increment in sigma_increments:
                print("***************TEST RUN**********************************")
                curr_sigma -= sigma_increment
                data, shuffled_data = generate_easily_clusterable_data(num_samples_per_cluster=n_samples_per_class,
                                                                       n_clusters=n_clusters,
                                                                       n_dimensions=n_dimensions,
                                                                       distrib=dist,
                                                                       mu_inc=mu_increment,
                                                                       sigma=curr_sigma
                                                                       )
                print("data has shape: ", data.shape)
                model = KMeans(n_clusters=n_clusters)

                actual_labels = shuffled_data[:, -1]
                predicted_labels = model.fit_predict(shuffled_data)
                score = homogeneity_score(actual_labels, predicted_labels)

                if sigma_increment in report:
                    report[sigma_increment].append(score_encoder.encode(score_object.ScoreObject(score=score, mu_increment=mu_increment, sigma=curr_sigma)))
                else:
                    report[sigma_increment] = [score_encoder.encode(score_object.ScoreObject(score=score, mu_increment=mu_increment, sigma=curr_sigma))]

                #if num_results == NUM_RESULTS_CAP:
                    #done = True
                    #break
                print(actual_labels[:10])
                print(predicted_labels[:10])
                print("model scored: ", score)
                print("***************TEST END**********************************")
                num_results += 1
    print(num_results, "results calculated.")
    
    with open("results.json", "w") as result_file:
        json.dump(report, result_file)

    
if __name__ == "__main__":
    main()

"""
Thursday 15th November, 2018
-Try less different values (distributions with smaller parameters)
-Investigate parameter estimation: 
-Find a way to test similarity of distributions (based on samples).
-Calculating confidence level of comparing samples from distribution.
-Can also try to see if people have tried to generate data to see the
--limit of the clustering

I'm planning to look at the mean and std for each column and plot histograms for both.

All the things that acn be varied:
- number of samples
- number of features per sample
- number of clusters
- type of probability distribution:
  - normal:
    - mean
    - standard deviation


"""
