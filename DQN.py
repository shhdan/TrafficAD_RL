import numpy as np
import torch
import torch as T
import torch.nn as nn
import torch.optim as optim
import os


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_steps, input_dims, n_actions, hidden_dims, n_layers, name, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.input_steps = input_steps
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.n_actions = n_actions
        self.n_layers = n_layers
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir, name)

        self.lstm = nn.LSTM(self.input_dims, self.hidden_dims, self.n_layers, batch_first=True)
        self.ln = nn.Linear(self.hidden_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        print(self.device)
        self.to(device=self.device)

    def forward(self, state):
        lstm_out, self.hidden_cell = self.lstm(state)
        actions = self.ln(lstm_out.index_select(1, T.tensor([1]).to(self.device)).squeeze(1))

        return actions

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))












