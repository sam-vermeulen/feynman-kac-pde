import numpy as np
from abc import ABC, abstractmethod
import torch

class Domain(ABC):
    def __init__(self, name=''):
        self.name = name
    
    @abstractmethod
    def sample_points(self, n):
        """
        Sample random points within the domain

        :param n: number of points to sample
        :return: np.array of points
        """
        pass

    def resample_points(self, points, where):
        return torch.where(where[:, None], self.sample_points(points.shape[:-1]), points)

    @abstractmethod
    def points_inside(self, points):
        """
        Checks if points are within the domain

        :param points: the points to check
        :return: np.array of booleans
        """
        pass

    @abstractmethod
    def exit_point(self, start, end):
        """
        Returns intersections points of domain boundary given start and end of line

        :param start: the start of the points to check
        :param end: the end of the points to check
        :return: np.array of intersection points
        """
        pass

class RectDomain(Domain):
    def __init__(self, boundaries=[[0, 1], [0, 1]], name='Rectangle'):
        super().__init__(name=name)

        self.boundaries = torch.tensor(boundaries, dtype=torch.float32)

    def sample_points(self, n):
        x = torch.rand(size=n) / (self.boundaries[0, 1] - self.boundaries[0, 0])
        y = torch.rand(size=n) / (self.boundaries[1, 1] - self.boundaries[1, 0])

        return torch.column_stack([x, y])
    
    def points_inside(self, points):
        x = points[:, 0]
        y = points[:, 1]

        in_x = (x > self.boundaries[0, 0]) & (x < self.boundaries[0, 1])
        in_y = (y > self.boundaries[1, 0]) & (y < self.boundaries[1, 1])

        return in_x & in_y
    
    def exit_point(self, start, end):
    
        vec = end - start

        lower_bounds = self.boundaries[:, 0]
        upper_bounds = self.boundaries[:, 1]

        ratio_to_upper = (upper_bounds - start) / (vec+1e-32)
        ratio_to_lower = (lower_bounds - start) / (vec+1e-32)

        ratio = torch.maximum(ratio_to_lower, ratio_to_upper)
        ratio = torch.min(ratio, dim=1)

        intersection = start + ratio.values[:, None] * vec

        return intersection

class CircleDomain(Domain):
    def __init__(self, centre=[0, 0], radius=1, name='Circle'):
        super().__init__(name)

        self.centre = torch.tensor(centre)
        self.radius = radius

    def sample_points(self, n):
        theta = torch.rand(size=n) / (2 * torch.pi - 0)
        
        r = torch.rand(size=n) / (self.radius - 0)

        x = r * torch.cos(theta) + self.centre[0]
        y = r * torch.sin(theta) + self.centre[1]

        return torch.column_stack([x, y])
    
    def points_inside(self, points):
        return torch.linalg.norm(points - self.centre, axis=1) < self.radius

    def exit_point(self, start, end):
        """
        centered_start = start - self.centre

        vec = end - start
        centered_vec = centered_start

        a = torch.einsum('...i,...i', vec, vec)[..., None]
        b = 2 * torch.einsum('...i,...i', vec, centered_vec)[..., None]
        c = (torch.einsum('...i,...i', centered_vec, centered_vec) - self.radius**2)[..., None]
    
        discriminant = torch.sqrt(b**2 - 4 * a * c)

        t1 = (-b - discriminant)/(2*a+1e-128)
        t2 = (-b + discriminant)/(2*a+1e-128)
        t1 = torch.maximum(t1, torch.tensor([0]))
        t2 = torch.maximum(t2, torch.tensor([0]))

        intersect = torch.stack([start + t1 * vec, start + t2 * vec], axis=1)
        """
        raise NotImplementedError('Not yet implemented')
        return 
        

