import numpy as np
import ppl.distributions as dist
from ppl.core import (
    trace, condition,
    sample, logpdf
)

# Define a simple model.
def model(x):
    # Prior for mean of x.
    mu = sample("mu", dist.Normal(0, 10))

    # Prior for SD of x.
    sigma = sample("sigma", dist.Uniform(0, 1))

    # Likelihood.
    sample("x", dist.Normal(mu, sigma), obs=x)

# Set a random seed.
np.random.seed(0)

# This returns nothing, as expected.
model(x = 10)

# Under different contexts (trace, condition), the following return different
# things.
t0 = trace(model).get(x = None)
t1 = trace(model).get(x = 10)
t2 = trace(condition(model, {"mu": 999})).get(x = 10)
t3 = trace(condition(model, {"sigma": 0.01})).get(x = None)
t4 = trace(condition(model, {"mu": 11, "sigma": 0.01})).get(x = 10)

# Print the values of the messages only.
for i, t in enumerate([t0, t1, t2, t3, t4]):
    d = {
        name: msg.value
        for name, msg in t.items()
    }
    print(f"t{i + 1}:", d)

print()

# Here, we generate 2000 draws from a Normal distribution (mean=3, scale=0.5).
data = np.random.normal(3, 0.5, 2000)

# We compute the log joint density of this Bayesian model sigma = 0.5, and
# various values of mu. The log joint density should be highest at mu = 3.
profile = {
    mu: logpdf(
        model,
        {"mu": mu, "sigma": 0.5},
        x = data
    )
    for mu in np.linspace(2.8, 3.2, 15)
}

# Print the log joint density for vaious values of mu.
# Should be highest at around 3.
for mu, lpdf in profile.items():
    print(f"mu: {mu:.3f}, {lpdf:.3f}")

def plot_log_joint_density_as_function_of_mu():
    import matplotlib.pyplot as plt
    plt.plot(profile.keys(), profile.values())
    plt.show()

# Uncomment the line below to see a plot of the log joint density as a function
# of mu.
# plot_log_joint_density_as_function_of_mu()
