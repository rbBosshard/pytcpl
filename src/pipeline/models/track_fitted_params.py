import os

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from src.pipeline.pipeline_constants import LOG_DIR_PATH
from src.pipeline.models.models import get_model


def track_fitted_params(fit_params):
    """
    Track and visualize the fitted parameters from curve-fitting results.

    This function collects fitted parameters from the curve-fitting results and generates histograms for each parameter
    of each fit model. It calculates and reports the median, minimum, and maximum values for each parameter.

    Args:
        fit_params (list): A list of dictionaries containing fitted parameters for different fit models.

    Outputs:
        Generates histograms for each parameter and writes statistics to an output file.
    """
    parameters = {}
    for result in fit_params:
        for model, params in result.items():
            parameters.setdefault(model, []).append(list(params['pars'].values()))

    def plot_histograms(params, key):
        plt.figure(figsize=(10, 6))
        plt.suptitle(f'Fit model parameters histograms: {key}')
        param_names = get_model(key)('params')
        num_params = len(param_names)
        for i, param_name in enumerate(param_names):
            plt.subplot(1, num_params, i + 1)
            plt.hist(params[:, i], bins=50)
            plt.title(f"{param_name}")
            plt.xlabel('Value')
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(LOG_DIR_PATH, "curve_fit_parameter_tracks", f"{key}.png"))
        plt.close()

    matplotlib.use('Agg')

    with open(os.path.join(LOG_DIR_PATH, "curve_fit_parameter_tracks", f"stats.out"), "w") as file:
        for key, param_list in parameters.items():
            param_array = np.array(param_list)
            plot_histograms(param_array, key)
            median, minimum, maximum = np.median(param_array, 0), np.min(param_array, 0), np.max(param_array, 0)
            file.write(f"{key}:\n >> median {median}\n >> min {minimum}\n >> max {maximum}\n\n")
