import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm, t

from acy import tcplObj
from mad import mad

er_est = -3
offset = 2

def g(p=None, conc=None, resp=None, fname=None, errfun="dt4", err=None):
    
    # objective function is the sum of log-likelihood of response given the model at each concentration
    # scaled by variance (err)
    if errfun == "dt4":
        # degree of freedom paramter = 4 for Student’s t probability density function
        return np.sum(t.logpdf((resp - 1) / err, df=4) - np.log(err))
    elif errfun == "dnorm":
        return np.sum(norm.logpdf((resp - 1) / err) - np.log(err))

def f(x = 1, conc=None, resp=None):
    return (x - 2) * x * (x + 2)**2
conc = []
fit = minimize_scalar(f,  bounds=(-3, -1), method='bounded', args=(conc, resp, fname)
# fit = minimize_scalar(f, bounds=(-3, -1), method='bounded')

print(fit)