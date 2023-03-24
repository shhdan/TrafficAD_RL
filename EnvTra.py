import pandas as pd
import numpy as np
import os
import sklearn.preprocessing

NOT_ANOMALY = 0
ANOMALY = 1
action_space = [NOT_ANOMALY, ANOMALY]
IDX_MAX = 480
N_STEPS = 10



def RewardFucClu(clu_rewards, timeseries_curser, action):
    '''
    :param clu_rewards: the pre-computed reward series
    :param timeseries_curser: the current time step
    :param action: the current action
    :return: the current reward
    '''
    return clu_rewards[timeseries_curser][action]




def StateFuc(timeseries, average, timeseries_curser, action, previous_state=[]):
    '''
    :param timeseries: the traffic flow time series
    :param average: the historical mean of each time step
    :param timeseries_curser: current time step
    :param action: the action taken based on the previous state
    :param previous_state: the previous state
    :return: the current state formed by N_STEPS of [flow, historical_mean_flow, action]. Note that the action for the
    current time step is set to be '-1'.
    '''

    # initialize the first state; we assume the first N_STEPS of flow values are NOT_ANOMALY
    if timeseries_curser == N_STEPS:
        state = []
        for i in range(timeseries_curser):
            his_mean = average['value'][timeseries['idx'][i] - 1]
            state.append([timeseries['value'][i], his_mean, NOT_ANOMALY])

        state.pop(0)
        his_mean = average['value'][timeseries['idx'][timeseries_curser] - 1]
        state.append([timeseries['value'][timeseries_curser], his_mean, -1])

        return np.array(state, dtype='float32')

    if timeseries_curser > N_STEPS:
        # copy the previous state
        # replace the action (with value '-1') with actual action taken in the previous time step
        his_mean = average['value'][timeseries['idx'][timeseries_curser] - 1]
        his_mean_pre = average['value'][timeseries['idx'][timeseries_curser - 1] - 1]
        state0 = np.concatenate((previous_state[1:N_STEPS-1],
                                 [[timeseries['value'][timeseries_curser-1], his_mean_pre, action]]))
        state = np.concatenate((state0,
                                [[timeseries['value'][timeseries_curser], his_mean, -1]]))

        return np.array(state, dtype='float32')



class EnvTafficRepo():
    # init the class instance
    def __init__(self, repodir, file_raw, file_avg, file_reward):

        self.repodir = repodir
        self.file_raw = file_raw
        self.file_avg = file_avg
        # the file with pre-computed reward
        self.file_reward = os.path.join(self.repodir, file_reward)
        # the file with the raw traffic flow
        self.timeseries_raw_file = os.path.join(self.repodir, self.file_raw)
        # the file with the pre-computed historical mean
        self.timeseries_avg_file = os.path.join(self.repodir, self.file_avg)

        self.action_space_n = len(action_space)
        self.timeseries_raw = []
        self.timeseries_curser = -1
        self.timeseries_curser_init = N_STEPS
        self.timeseries_states = []
        self.timeseries_avg = []
        self.detected_result = pd.DataFrame(columns=['idx','value','action'])
        self.cluster_reward = []


        self.statefnc = StateFuc
        self.rewardfnc = RewardFucClu

        ts = pd.read_csv(self.timeseries_raw_file, usecols=[0,1], header=None, names=['idx','value'])
        avg = pd.read_csv(self.timeseries_avg_file, usecols=[0,1], header=None, names=['idx','value'])
        self.cluster_reward = pd.read_csv(self.file_reward, usecols=[0,1], header=None).to_numpy()


        ts['value'] = ts['value'].astype(np.float32)
        avg['value'] = avg['value'].astype(np.float32)

        scaler = sklearn.preprocessing.MinMaxScaler()
        scaler.fit(np.array(ts['value']).reshape(-1, 1))
        scaler.fit(np.array(avg['value']).reshape(-1, 1))
        ts['value'] = scaler.transform(np.array(ts['value']).reshape(-1, 1))
        avg['value'] = scaler.transform(np.array(avg['value']).reshape(-1, 1))

        self.timeseries_raw = ts
        self.timeseries_avg = avg
        self.datasetrng = len(self.timeseries_raw)

    # reset the instance
    def reset(self):
        self.timeseries_curser = self.timeseries_curser_init
        self.detected_result = pd.DataFrame(columns=['idx', 'value', 'action'])

        # return the first state, containing the first element of the time series
        self.timeseries_states = self.statefnc(self.timeseries_raw, self.timeseries_avg, self.timeseries_curser, 0)

        # store the first N_STEPS of results
        for i in range(self.timeseries_curser):
            list = pd.Series([self.timeseries_raw['idx'][i], self.timeseries_raw['value'][i], 0],
                             index=self.detected_result.columns)
            self.detected_result = self.detected_result.append(list, ignore_index=True)
        return self.timeseries_states


    # take a step and gain a reward
    def step(self, action):
        # 0. append the result for the current time step
        list = pd.Series([self.timeseries_raw['idx'][self.timeseries_curser],
                          self.timeseries_raw['value'][self.timeseries_curser], action],
                         index=self.detected_result.columns)
        self.detected_result = self.detected_result.append(list, ignore_index=True)

        # 1. get the reward of the action
        reward = self.rewardfnc(self.cluster_reward,
                                self.timeseries_curser, action)

        # 2. get the next state and the done flag after the action
        self.timeseries_curser += 1

        if self.timeseries_curser >= self.timeseries_raw['value'].size:
            done = 1
            state = self.timeseries_states
        else:
            done = 0
            state = self.statefnc(self.timeseries_raw, self.timeseries_avg, self.timeseries_curser, action, self.timeseries_states)

        self.timeseries_states = state

        return state, reward, done, []


    def get_detected_result(self):
        '''
        :return: detected_result: in the format of[[idx_0, value_1, action_1], [idx_2, value_2, action_2], ...]
        '''
        return self.detected_result
