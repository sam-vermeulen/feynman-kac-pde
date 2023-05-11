import numpy as np

class BrownianMotion():
    def __init__(self, scale=1.0, T=1.0, rng=None):
        self.n = None
        self.times = None
        self.T = T
        self.scale = scale

        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng
    
    def sample(self, n, batch_size=1):
        """
        Generate discrete time sample of a Brownian Motion

        :param n: number of steps
        :return: numpy array
        """
        
        self.n = n
        self.times = np.linspace(0, self.T, n)
        delta_t = self.T / self.n
        gaussian_noise = self.rng.normal(loc=0, scale=np.sqrt(delta_t), size=(batch_size, self.n - 1))
        bm = np.cumsum(self.scale * gaussian_noise, axis=-1)
        bm = np.insert(bm, 0, [0], axis=-1)
        
        return bm
