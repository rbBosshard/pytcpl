import numpy as np
import pandas as pd
from tcplfit2_core import tcplfit2_core
# Define the model functions (fitcnst, fithill, fitgnls, etc.) and acy function

# Sample input data
conc = pd.DataFrame([0.1, 0.5, 1.0, 2.0, 5.0])
resp = pd.DataFrame([0.2, 0.8, 1.2, 2.5, 5.5])
cutoff = 0.5
force_fit = False
bidirectional = True
verbose = True
do_plot = True
fitmodels = ["hill", "poly2", "pow", "exp2", "exp3"]

# Call the function
result = tcplfit2_core(conc, resp, cutoff, force_fit, bidirectional, verbose, do_plot, fitmodels)

# Print the output dictionary
for model in result["modelnames"]:
    print(model, ":\n", result[model])
