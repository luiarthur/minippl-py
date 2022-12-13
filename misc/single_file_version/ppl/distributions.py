import numpy as np

class Distribution:
    def logpdf(self, x):
        pass

    def sample(self):
        pass

class Normal(Distribution):
    def __init__(self, loc=0, scale=1):
        self.loc = loc
        self.scale = scale

    def logpdf(self, x):
        z = (x - self.loc) / self.scale
        const = -0.5 * np.log(2 * np.pi) - np.log(self.scale)
        return -(z * z) / 2 + const

    def sample(self):
        return np.random.normal(self.loc, self.scale)

class Uniform(Distribution):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def logpdf(self, x):
        return -np.log(self.upper - self.lower)

    def sample(self):
        return np.random.uniform(self.lower, self.upper)
