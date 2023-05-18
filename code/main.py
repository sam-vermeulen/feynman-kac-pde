from util.domain import RectDomain, CircleDomain
from processes.gaussian_increments import GaussianIncrements
from util.collector import BatchCollector
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()

        self.ffw = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output)
        )

    def forward(self, x):
        return self.ffw(x)

class TDFeynman():
    def __init__(self, model, optimizer, collector):
        self.model = model
        self.collector = collector
        self.optimizer = optimizer
        
    def train(self):
        def g_estimate(old, new):
            return torch.zeros_like(new)
        
        def f_estimate(old, new):
            return self.collector.process.time_step * torch.ones_like(new)

        def loss_fn(old, new, u_new):
            return ((1/2) * (u_new - self.model(old) - (1/2)*f_estimate(old, new))**2).mean()

        logs = defaultdict(list)
        with tqdm(total=self.collector.max_exits, unit='exits') as pbar:
            for i, (old_points, points, exited, clipped_points) in enumerate(self.collector):
                u_new = torch.where(exited[:, None], g_estimate(old_points, points), self.model(points).detach())
                loss = loss_fn(old_points, points, u_new)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if i % 1 == 0:
                    logs['loss'].append(loss.item())
                
                pbar.update(torch.count_nonzero(exited).item())

        return logs

if __name__ == '__main__':
    print("CUDA Device available:", torch.cuda.is_available())

    batch_shape = (2**16,)
    max_exits = 2**19

    domain = RectDomain()
    process = GaussianIncrements(time_step=0.001)
    model = Model(2, 128, 1)
    
    collector = BatchCollector(domain=domain, process=process, batch_size=batch_shape, max_exits=max_exits)
    optimizer = torch.optim.Adam(model.parameters())

    td_pde = TDFeynman(model, optimizer, collector)
    logs = td_pde.train()

    plt.plot(logs['loss'])
    plt.show()

