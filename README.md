# TrafficAD_RL

This study develops an autonomous artificial intelligence (AI) agent to detect anomalies in traffic flow time series data, 
which can learn anomaly patterns from data without supervision, requiring no ground-truth labels for model training or knowledge
 of a threshold for anomaly definition. Specifically, our model is based on reinforcement learning, where an agent is built by a
 Long-Short-Term-Memory (LSTM) model and Q-learning algorithm to incorporate sequential information in time series data into policy
 optimization. The key contribution of our model is the development of a novel unsupervised reward learning algorithm that 
 automatically learns the reward for an action taken by the agent based on the distribution of data, without requiring a manual 
 specification of a reward function. To test the performance of our model, we conduct a comprehensive set of experimental study 
 on both real-world data from Brisbane city, Australia, and synthetic data simulated according to the distribution of real-world data. 
 We compare the performance of our model against three state-of-the-art models, and the experimental results show that our model 
 outperforms the other models in different parameter settings, with around 90\% precision, 80\% recall, and 85\% F1 score.

## Data Set
Please refer to the sample data in ./traffic_data for the format of the input data with/without ground truth labels.

## Requirement
pytorch 1.12 \
pandas 1.2 \
numpy 1.21 \
python 3.7

## Remarks
This cite the following paper when using the codes in this repo

@article{he2023autonomous, \
  title={Autonomous anomaly detection on traffic flow time series with reinforcement learning}, \
  author={He, Dan and Kim, Jiwon and Shi, Hua and Ruan, Boyu}, \
  journal={Transportation Research Part C: Emerging Technologies}, \
  volume={150}, \
  pages={104089}, \
  year={2023}, \
  publisher={Elsevier} \
}