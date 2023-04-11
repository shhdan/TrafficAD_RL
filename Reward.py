import pandas as pd
import numpy as np
import random
import sklearn.preprocessing
from sklearn.neighbors import KernelDensity


def reward_calculation_kde_clustering(input_file, output_file, neighbor_time_step, daily_time_step):
    '''
    :param input_file: the input raw traffic flow data in the format of ['daily_time_idx', 'flow']
    :param output_file: the output reward file in the format of
    :param neighbor_time_step:
    :param daily_time_step:
    :return:
    '''
    # read the flow time series data and normalization
    ts = pd.read_csv(input_file, usecols=[0, 1], header=None, names=['idx', 'value'])
    ts['value'] = ts['value'].astype(np.float32)

    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(np.array(ts['value']).reshape(-1, 1))
    ts['value'] = scaler.transform(np.array(ts['value']).reshape(-1, 1))

    # the dictionary for key:index (start from 0), value:flow
    index_value_dict = {}
    for i in range(len(ts)):
        idx = ts['idx'][i] - 1
        val = ts['value'][i]
        if idx in index_value_dict.keys():
            index_value_dict[idx].append([i, val])
        else:
            index_value_dict[idx] = [[i, val]]

    reward = [None] * len(ts)

    # begin the main loop to calculate the reward
    for i in range(daily_time_step):
        # the flow data at time step i
        pivot = index_value_dict[i]
        # the data to do clustering
        data = np.array(index_value_dict[i])
        for j in range(neighbor_time_step):
            idx = (i - (j + 1)) % daily_time_step
            data = np.concatenate((data, index_value_dict[idx]), axis=0)
        for j in range(neighbor_time_step):
            idx = (i + j + 1) % daily_time_step
            data = np.concatenate((data, index_value_dict[idx]), axis=0)
        # perform clustering on data set

        X = np.array(data)[:, 1].reshape(-1, 1)
        s = np.linspace(0, X.max(), 1000).reshape(-1, 1)
        e = sklearn_kde(X, s)
        from scipy.signal import argrelextrema
        mi, ma = argrelextrema(np.exp(e), np.less)[0], argrelextrema(np.exp(e), np.greater)[0]
        labels = [-1] * len(data)
        if len(s[mi]) == 0:
            for k in range(len(pivot)):
                idx = pivot[k][0]
                ratio = random.choice([-1, 1])
                reward[idx] = [ratio, -ratio]
        else:
            for j in range(len(X)):
                for idx_mi in range(len(s[mi])):
                    if idx_mi == 0:
                        if X[j] >= 0 and X[j] < s[mi][idx_mi]:
                            labels[j] = idx_mi
                            break
                    else:
                        if X[j] >= s[mi][idx_mi-1] and X[j] < s[mi][idx_mi]:
                            labels[j] = idx_mi
                            break
                if X[j] >= s[mi][len(s[mi])-1]:
                    labels[j] = len(s[mi])

            # calculate the number of element in each cluster
            label_num_dict = {}
            for k in range(len(data)):
                if labels[k] in label_num_dict.keys():
                    label_num_dict[labels[k]] += 1
                else:
                    label_num_dict[labels[k]] = 1

            avg_num_clu = len(data) / (len(s[mi]) + 1)

            # calculate the reward value for each data point, each reward in the format [x,y]
            # x: the reward for a=0
            # y: the reward for a=1
            for k in range(len(pivot)):
                lab = labels[k]
                delta = label_num_dict[lab] / avg_num_clu
                idx = pivot[k][0]
                if delta > 1:
                    # in the case this flow in a larger cluster
                    reward[idx] = [delta, -delta]
                else:
                    # in the case this flow in a smaller cluster
                    ratio = 1 / delta
                    reward[idx] = [-ratio, ratio]
    reward_file = open(output_file, '+w')
    for i in range(len(reward)):
        reward_file.write(str(round(reward[i][0], 3)) + ',' + str(round(reward[i][1], 3)) + '\n')
    reward_file.close()


def sklearn_kde(data, points):

    # Silverman bandwidth estimator
    n, d = data.shape
    bandwidth = (n * (d + 2) / 4.)**(-1. / (d + 4))
    # standardize data so that we can use uniform bandwidth
    mu, sigma = np.mean(data, axis=0), np.std(data, axis=0)
    data, points = (data - mu)/sigma, (points - mu)/sigma

    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth, rtol=1e-6, atol=1e-6)
    kde.fit(data)
    log_pdf = kde.score_samples(points)

    return np.exp(log_pdf)

def mean_calculation(from_file, to_file, time_slot):
    # read the flow time series data and normalization
    ts = pd.read_csv(from_file, usecols=[0, 1], header=None, names=['idx', 'value'])
    ts['value'] = ts['value'].astype(np.float32)


    # the dictionary for key:index (start from 0), value:flow
    index_value_dict = {}
    for i in range(len(ts)):
        idx = ts['idx'][i]
        val = ts['value'][i]
        if idx in index_value_dict.keys():
            index_value_dict[idx].append(val)
        else:
            index_value_dict[idx] = [val]

    mean_file = open(to_file, "+w")
    for i in range(time_slot):
        idx = i+1
        flow_values = index_value_dict[idx]
        flow_values = np.array(flow_values)
        mean = flow_values.mean()
        mean_file.write(str(int(idx)) + ',' + str(round(mean, 3)) + '\n')
    mean_file.close()
