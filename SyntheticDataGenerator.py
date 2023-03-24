import pandas as pd
import numpy as np
import random

def get_avg_std_from_real(file, boundary, to_file):
    raw = pd.read_csv(file, header=None).to_numpy()
    selected_raw = raw[boundary[0]:boundary[1]]
    idx_dict = {}

    for i in range(len(selected_raw)):
        if int(selected_raw[i][0]) in idx_dict.keys():
            idx_dict[int(selected_raw[i][0])].append(selected_raw[i][1])
        else:
            idx_dict[int(selected_raw[i][0])] = [selected_raw[i][1]]
    print(idx_dict)
    mean_std=[]
    for i in range(480):
        data=idx_dict[i+1]
        mean=np.mean(data)
        std=np.std(data)
        mean_std.append([mean,std])
    print(mean_std)

    out = open(to_file, '+w')
    for i in range(len(mean_std)):
        out.write(str(int(i+1)) + ',' + str(round(mean_std[i][0], 3)) + ',' + str(round(mean_std[i][1], 3)) + '\n')
    out.close()

def generate_normal_flow(mean_std_file, to_file, days):
    mean_std = pd.read_csv(mean_std_file, header=None, usecols=[1,2]).to_numpy()
    #print(mean_std)
    flow_by_day=[]
    for i in range(480):
        flow = np.random.normal(loc=mean_std[i][0], scale=mean_std[i][1], size=days)
        flow_by_day.append(flow)

    out = open(to_file, '+w')
    for i in range(days):
        for j in range(480):
            if flow_by_day[j][i] < 0:
                flow_by_day[j][i] = 0
            out.write(str(int(j+1)) + ',' + str(round(flow_by_day[j][i], 3)) + '\n')
    out.close()

def add_anomaly(normal_file, anomaly_file, days):

    origin_data = pd.read_csv(normal_file, header=None).to_numpy()
    format_data = []
    unit = []
    for i in range(len(origin_data)):
        if origin_data[i][0] == 480:
            unit.append([origin_data[i][1], 0])
            format_data.append(unit)
            unit = []
        else:
            unit.append([origin_data[i][1], 0])
    #print(format_data)
    d = int(days/5)
    anomaly_day = np.random.choice(range(days), d, replace=False)
    print(anomaly_day)
    for i in range(len(anomaly_day)):
        sequence_length = random.randrange(10, 20)
        print(sequence_length)
        start_idx = random.randrange(120, 300)
        while(start_idx + sequence_length > 440):
            start_idx = random.randrange(120, 300)
        print(start_idx)
        for j in range(sequence_length):
            d = anomaly_day[i]
            error_per = random.randrange(40, 90)
            format_data[d][start_idx+j][0] = format_data[d][start_idx+j][0] - \
                                          format_data[d][start_idx+j][0] * error_per/100
            format_data[d][start_idx+j][1] = 1
            #if format_data[d][start_idx+j] < 0:
            #    format_data[d][start_idx+j] = 0

    out = open(anomaly_file, '+w')
    for i in range(len(format_data)):
        for j in range(len(format_data[i])):
            out.write(str(int(j+1)) + ',' + str(round(format_data[i][j][0], 3)) + ',' +
                      str(int(format_data[i][j][1])) + '\n')
    out.close()


if __name__ == '__main__':
    #get_avg_std_from_real('traffic_data/2922872.csv', [480*2, 480*20], 'traffic_data/2922872_syt_meanstd.csv')
    #generate_normal_flow('traffic_data/2922872_syt_meanstd.csv', 'synthetic_short_sequence/sequence_20.csv', 40)
    add_anomaly('synthetic_short_sequence/sequence_20.csv', 'synthetic_short_sequence/sequence_20_2.csv', 40)
