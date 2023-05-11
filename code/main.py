from util.domain import RectDomain, CircleDomain
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    domain = CircleDomain()

    start = np.array([
        [0.2, 0.2],
        [0.3, 0.0]
    ])

    end = np.array([
        [1.0, 1.0],
        [1.0, 0.0]
    ])

    points_in_domain = domain.sample_points(100)
    
    
    #points = domain.line_intersection(start, end)
    print(points_in_domain)