import numpy as np

def hitloginner(conc, resp, top, cutoff, ac50=None):
    n_gt_cutoff = np.sum(np.abs(resp) > cutoff)

    # hitlogic - hit must have: at least one point above abs cutoff,
    # a defined top (implying there is a winning non-constant model),
    # and an abs. top greater than the cutoff
    hitcall = 0
    if n_gt_cutoff > 0 and top is not None and np.abs(top) > cutoff:
        hitcall = 1

    return hitcall