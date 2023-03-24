import numpy as np
import matplotlib.pyplot as plt
import torch as T
import EnvTra
from Agent import DQNAgent
from Env import *
from sklearn import metrics
import time
from datetime import datetime


def smooth_result(results, k):
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
            #results['action'][i+k] = 0
            forward[i+k] = 0
        else:
            #results['action'][i+k] = 1
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
            # results['action'][i+k] = 0
            backward[len(results) - 1 - (i + k)] = 0
        else:
            # results['action'][i+k] = 1
            backward[len(results) - 1 - (i + k)] = 1
    for i in range(len(results)):
        if forward[i] + backward[i] > 0:
            results['action'][i] = 1
        else:
            results['action'][i] = 0
    return results


def play():
    data_path = 'synthetic_short_sequence'
    file_name = 'sequence_20_2'
    data_raw_file = file_name + '.csv'
    data_avg_file = file_name + '_mean.csv'
    data_reward_file = file_name + '_reward.csv'
    #data_raw_file = str(link_id) + '.csv'
    #data_avg_file = str(link_id) + '_average.csv'
    #data_reward_file = str(link_id) + '_reward.csv'
    log_file = data_path + '/' + data_raw_file[:-4] + '_log'
    #log = open(log_file, '+w')
    out_file_app_be = '_result_bs'
    out_file_app_af = '_result_as'

    print(log_file)
    #print(data_raw_file, thre_non_anomaly, thre_anomaly)

    ground_truth = pd.read_csv(data_path + '/'+data_raw_file, header=None, usecols=[0,1,2], names=
                               ['id', 'value', 'anomaly'])
    #ground_truth = pd.read_csv(data_path + '/' + data_raw_file, header=None, usecols=[0, 1], names=
    #['id', 'value'])
    env = EnvTra.EnvTafficRepo(data_path, data_raw_file, data_avg_file, data_reward_file)

    best_score = -np.inf
    load_checkpoint = False
    n_epoch = 12

    agent = DQNAgent(gamma=0.99, epsilon=0.5, lr=0.0001, n_layers=2,
                     input_dims=4, input_steps=10, hidden_dims=128,
                     n_actions=env.action_space_n, mem_size=10000, eps_min=0.01,
                     batch_size=32, eps_dec=5e-6)

    if load_checkpoint:
        agent.load_models()


    n_steps = 0
    scores, eps_history, steps_array = [], [], []
    final_result = pd.DataFrame(columns=['idx','value','action'])
    max_obj_score = -1

    start_time = time.time()
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)

    for i in range(n_epoch):
        done = False
        observation = env.reset()

        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)


            #log.write(str(n_steps) + ',' + str(round(observation[19][0],3)) + ',' + str(action) + ',' + str(reward) + '\n')
            score += reward
            #print(action, reward)

            agent.store_transition(observation, action,
                                       reward, observation_, done)
            agent.learn()
            observation = observation_

            if n_steps % 10000 == 0:
                #log.flush()
                print(n_steps, score)
            n_steps += 1

        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode: ', i, 'score: ', score,
              ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
              'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)

        current_results = env.objective_function_score()
        #performance_evaluation(ground_truth, current_results)
        #if obj_score > max_obj_score:
        #    max_obj_score = obj_score
        #    final_result = current_results
        #tmp = final_result.append(current_results, ignore_index=True)
        final_result = current_results
    end_time = time.time()
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("End Time =", current_time)
    print("--- %s seconds ---" % (end_time - start_time))

    final_result.to_csv(data_path + '/' + data_raw_file[:-4] + out_file_app_be +'_sr.csv')
    print('Before post-processing performance')
    performance_evaluation(ground_truth, final_result)
    final_result=smooth_result(final_result, 10)
    final_result.to_csv(data_path + '/' + data_raw_file[:-4] + out_file_app_af + '_sr.csv')
    print('After smoothing the results performance')
    performance_evaluation(ground_truth, final_result)


    #log.close()
if __name__ == '__main__':
    play()
    #print(T.__version__)
    #print(T.cuda.is_available())





