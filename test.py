from ppl.core.model import AbstractModel
from ppl.core.handlers import trace, condition
import ppl.distributions as dist

class NormalModel(AbstractModel):
    def model(self, x):
        mu = self.rv("mu", dist.Normal())
        sigma = self.rv("sigma", dist.Uniform(0, 1))
        self.rv("x", dist.Normal(mu, sigma), obs=x)

model = NormalModel()
model.run(x = 10)

t1 = trace(condition(model, {"mu": 999})).get(x = 10)
t2 = trace(condition(model, {"sigma": 0.01})).get(x = None)
t3 = trace(condition(model, {"mu": 11, "sigma": 0.01})).get(x = 10)

for i, t in enumerate([t1, t2, t3]):
    d = {
        name: msg.value
        for name, msg in t.items()
    }
    print(f"t{i + 1}:", d)
