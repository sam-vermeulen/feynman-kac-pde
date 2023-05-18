import numpy as np
import torch



class GaussianIncrements():
    def __init__(self, time_step):
        self.time_step = time_step
        self.scale = torch.sqrt(torch.tensor(time_step))

    def apply_noise(self, points):
        points += self.scale * torch.randn(size=points.shape)
        return points