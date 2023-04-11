import numpy as np
import matplotlib.pyplot as plt
import torch as T
import EnvTra
from Reward import *
from Agent import DQNAgent
import pandas as pd
from sklearn import metrics
import time
from datetime import datetime


# config
DATA_PATH = 'traffic_data'
FILE_NAME = 'real_world_data'
GAMMA = 0.99
EPSILON = 0.5
LEARNING_RATE = 0.0001
NUM_LAYERS = 2
INPUT_DIM = 3
INPUT_SEQ_LENGTH = 10
HIDDEN_DIM = 128
BUFFER_SIZE = 10000
EPS_MIN = 0.01
EPS_DEC = 5e-6
BATCH_SIZE = 32
NUM_EPOCH = 8



def post_processing_smoothing(results, k):
    forward = results['action']
    backward = results['action']
    for i in range(len(results)-2*k):
        zeros = 0
        ones = 0
        for j in range(2*k+1):
            if results['action'][i+j] == 0:
                zeros += 1
            else:
                ones += 1
        if zeros > k:
            forward[i+k] = 0
        else:
            forward[i+k] = 1
    for i in range(len(results)-2*k):
        zeros = 0
        ones = 0
        for j in range(2 * k + 1):
            if results['action'][len(results) - 1 - (i + j)] == 0:
                zeros += 1
            else:
                ones += 1
        if zeros > k:
            backward[len(results) - 1 - (i + k)] = 0
        else:
            backward[len(results) - 1 - (i + k)] = 1
    for i in range(len(results)):
        if forward[i] + backward[i] > 0:
            results['action'][i] = 1
        else:
            results['action'][i] = 0
    return results

def performance_evaluation(gound_truth, results):
    gt = []
    pred = []
    for i in range(len(gound_truth)):
        gt.append(gound_truth['anomaly'][i])
        pred.append(results['action'][i])
    acc = metrics.accuracy_score(gt, pred)
    precision, recall, F1, _ = metrics.precision_recall_fscore_support(gt, pred, average='binary')
    print(metrics.confusion_matrix(gt, pred))

    print('acc:', acc, 'precision:', precision, 'recall:', recall, 'F1 score:', F1)
    return 0

def TrafficAD(with_ground_truth):

    if with_ground_truth:
        ground_truth = pd.read_csv(f'{DATA_PATH}/{FILE_NAME}.csv', header=None, usecols=[0,1,2], names=
                               ['id', 'value', 'anomaly'])

    env = EnvTra.EnvTafficRepo(DATA_PATH, f'{FILE_NAME}.csv',
                               f'{FILE_NAME}_mean.csv', f'{FILE_NAME}_reward.csv')

    agent = DQNAgent(gamma=GAMMA, epsilon=EPSILON, lr=LEARNING_RATE, n_layers=NUM_LAYERS,
                     input_dims=INPUT_DIM, input_steps=INPUT_SEQ_LENGTH, hidden_dims=HIDDEN_DIM,
                     n_actions=env.action_space_n, mem_size=BUFFER_SIZE, eps_min=EPS_MIN,
                     batch_size=BATCH_SIZE, eps_dec=EPS_DEC)

    load_checkpoint = False
    if load_checkpoint:
        agent.load_models()

    n_steps = 0
    scores, eps_history, steps_array = [], [], []
    final_result = pd.DataFrame(columns=['idx','value','action'])
    start_time = time.time()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)

    for i in range(NUM_EPOCH):
        done = False
        observation = env.reset()

        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_

            if n_steps % 10000 == 0:
                print(n_steps, score)
            n_steps += 1

        scores.append(score)
        steps_array.append(n_steps)
        eps_history.append(agent.epsilon)
        current_results = env.get_detected_result()

        # if the detected results for all epochs need to be stored, modify the follow line to append the current_results
        final_result = current_results

    end_time = time.time()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("End Time =", current_time)
    print("--- The time for anomaly detection = %s seconds ---" % (end_time - start_time))

    final_result = post_processing_smoothing(final_result, 10)
    final_result.to_csv(f'{DATA_PATH}/{FILE_NAME}_results.csv')

    if with_ground_truth:
        # performance evaluation
        performance_evaluation(ground_truth, final_result)

    return final_result

if __name__ == '__main__':
    # generate mean flow data and reward file
    reward_calculation_kde_clustering(f'{DATA_PATH}/{FILE_NAME}.csv', f'{DATA_PATH}/{FILE_NAME}_reward.csv', 2, 480)
    mean_calculation(f'{DATA_PATH}/{FILE_NAME}.csv', f'{DATA_PATH}/{FILE_NAME}_mean.csv', 480)
    with_ground_truth = False
    TrafficAD(with_ground_truth)






