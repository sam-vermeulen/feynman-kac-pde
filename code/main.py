from util.domain import RectDomain, CircleDomain
from processes.gaussian_increments import GaussianIncrements
from util.collector import BatchCollector
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

from pde import CartesianGrid, solve_laplace_equation

def finite_difference_laplace_soln():
  grid = CartesianGrid([[0, 1]] * 2, 100)
  bcs = [[{'value': 0}, {'value': 0}], [{'value': 0}, {'value': 1}]]
  result = solve_laplace_equation(grid, bcs)
  return result

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
        
    def train(self, verbose=True):
        def g_estimate(old, new, clipped):
            values = torch.where(torch.isclose(clipped[:, 1], self.collector.domain.boundaries[1, 1]), 1, 0)
            
            return values
        
        def f_estimate(old, new):
            values = torch.zeros(new.shape[:-1])
            return values

        def loss_fn(old, new, u_new):
            loss = ((1/2) * (u_new - self.model(old) - (1/2)*f_estimate(old, new)[:, None])**2)
            return loss.mean()

        logs = defaultdict(list)
        with tqdm(total=self.collector.max_exits, unit='exits', disable=not verbose) as pbar:
            for i, (old_points, points, exited, clipped_points) in enumerate(self.collector):
                u_new = torch.where(exited[:, None], g_estimate(old_points, points, clipped_points)[:, None], self.model(points).detach())
                
                loss = loss_fn(old_points, points, u_new)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if i % 10 == 0:
                    logs['loss'].append(loss.item())
                    
                pbar.update(torch.count_nonzero(exited).item())

        return logs
    
if __name__ == '__main__':
    print("CUDA Device available:", torch.cuda.is_available())

    batch_shape = (2**16,)
    max_exits = 2**18

    domain = RectDomain()
    process = GaussianIncrements(time_step=0.001)
    model = Model(2, 128, 1)
    
    collector = BatchCollector(domain=domain, process=process, batch_size=batch_shape, max_exits=max_exits)
    optimizer = torch.optim.Adam(model.parameters())

    td_pde = TDFeynman(model, optimizer, collector)
    logs = td_pde.train()

    plt.plot(logs['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss throughout training')

    x = np.linspace(0, 1, 100, dtype=np.float32)
    y = np.linspace(0, 1, 100, dtype=np.float32)

    xv, yv = np.meshgrid(x, y)
    xv = xv.reshape(-1,)
    yv = yv.reshape(-1,)
    coords = np.stack([xv, yv], axis=-1)
    pred = model(torch.from_numpy(coords)).reshape(100, 100)

    plt.imshow(pred.detach().numpy(), origin='lower', extent=(0, 1, 0, 1))
    plt.title('NN Prediction')
    plt.colorbar()
    plt.show()

    u_val = finite_difference_laplace_soln()
    u_val.plot()

    err = pred.detach().numpy() - u_val.data.T
    plt.imshow(err, origin='lower', extent=(0, 1, 0, 1))
    plt.title('Error')
    plt.colorbar()
    plt.show()

    print("RMSE", np.sqrt((err**2).mean()))

