import numpy as np
import torch as T
from DQN import DeepQNetwork
from Buffer import ReplayBuffer

class DQNAgent(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, input_steps, hidden_dims,
                 n_layers, mem_size, batch_size, eps_min=0.01, eps_dec=1e-4,
                 replace=500, algo=None, env_name=None, chkpt_dir='tmp/dqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.input_steps = input_steps
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, input_steps, input_dims, n_actions)

        self.q_eval = DeepQNetwork(lr=self.lr, n_actions=self.n_actions, hidden_dims=self.hidden_dims,
                                    input_dims=self.input_dims, input_steps=self.input_steps, n_layers = self.n_layers,
                                    name='q_eval', chkpt_dir=self.chkpt_dir)

        self.q_next = DeepQNetwork(lr=self.lr, n_actions=self.n_actions, hidden_dims=self.hidden_dims,
                                    input_dims=self.input_dims, input_steps=self.input_steps, n_layers = self.n_layers,
                                    name='q_next', chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation):
        r = np.random.random()
        #print(r,self.epsilon)
        if r > self.epsilon:
            state = T.tensor([observation],dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        #print(action)
        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = \
                                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        T.autograd.set_detect_anomaly(True)
        loss.backward(retain_graph=True)
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

    def update_eps(self, epsilon):
        self.epsilon = epsilon
