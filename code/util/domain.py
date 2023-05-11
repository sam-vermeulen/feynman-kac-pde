import numpy as np
from abc import ABC, abstractmethod

class Domain(ABC):
    def __init__(self, name='', rng=None):
        self.name = name
        self.rng = rng

    @property
    def rng(self):
        if self._rng is None:
            return np.random.default_rng()
        return self._rng
    
    @rng.setter
    def rng(self, value):
        if value is None:
            self._rng = None
        elif isinstance(value, (np.random.RandomState, np.random.Generator)):
            self._rng = value
        else:
            raise TypeError('rng must be of type `numpy.random.Generator`')
    
    @abstractmethod
    def sample_points(self, n):
        """
        Sample random points within the domain

        :param n: number of points to sample
        :return: np.array of points
        """
        pass

    @abstractmethod
    def points_inside(self, points):
        """
        Checks if points are within the domain

        :param points: the points to check
        :return: np.array of booleans
        """
        pass

    @abstractmethod
    def line_intersection(self, start, end):
        """
        Returns intersections points of domain boundary given start and end of line

        :param start: the start of the points to check
        :param end: the end of the points to check
        :return: np.array of intersection points
        """
        pass

class RectDomain(Domain):
    def __init__(self, boundaries=[[0, 1], [0, 1]], name='Square', rng=None):
        super().__init__(name=name, rng=rng)

        self.boundaries = np.array(boundaries)

    def sample_points(self, n):
        x = self.rng.uniform(self.boundaries[0, 0], self.boundaries[0, 1], size=n)
        y = self.rng.uniform(self.boundaries[1, 0], self.boundaries[1, 1], size=n)

        return np.column_stack([x, y])
    
    def points_inside(self, points):
        x = points[:, 0]
        y = points[:, 1]

        in_x = (x > self.boundaries[0, 0]) & (x < self.boundaries[0, 1])
        in_y = (y > self.boundaries[1, 0]) & (y < self.boundaries[1, 1])

        return in_x & in_y
    
    def line_intersection(self, start, end):
        f_bounds = self.boundaries[None, ...]

        f = (self.boundaries[None, ...] - start[:, :, None])/((end - start + 1e-128)[:, :, None])

        f[:, :, 0] = np.maximum(f_bounds[:, :, 0], f[:, :, 0])
        f[:, :, 1] = np.minimum(f_bounds[:, :, 1], f[:, :, 1])

        f = np.min(f, axis=1)
        vec = end - start
        intersect = start[:, None, :] + np.einsum('...i,...j->...ij', f, vec)

        return intersect

class CircleDomain(Domain):
    def __init__(self, centre=[0, 0], radius=1, name='Circle', rng=None):
        super().__init__(name, rng)

        self.centre = np.array(centre)
        self.radius = radius

    def sample_points(self, n):
        theta = self.rng.uniform(0, 2*np.pi, size=n)
        r = self.rng.uniform(0, self.radius, size=n)

        x = r * np.cos(theta) + self.centre[0]
        y = r * np.sin(theta) + self.centre[1]

        return np.column_stack([x, y])
    
    def points_inside(self, points):
        centered = points - self.centre
        return np.linalg.norm(centered, axis=1) < self.radius

    def line_intersection(self, start, end):
        centered_start = start - self.centre

        vec = end - start
        centered_vec = centered_start

        a = np.einsum('...i,...i', vec, vec)[..., None]
        b = 2 * np.einsum('...i,...i', vec, centered_vec)[..., None]
        c = (np.einsum('...i,...i', centered_vec, centered_vec) - self.radius**2)[..., None]
    
        discriminant = np.sqrt(b**2 - 4 * a * c)

        t1 = (-b - discriminant)/(2*a+1e-128)
        t2 = (-b + discriminant)/(2*a+1e-128)
        t1 = np.maximum(t1, 0)
        t2 = np.maximum(t2, 0)

        i1 = start + t1 * vec
        i2 = start + t2 * vec

        intersect = np.stack([start + t1 * vec, start + t2 * vec], axis=1)

        return intersect
        

