import numpy as np


def mc5_mthds(mthd, bmad):
    return {
        'pc20': 20,
        'pc50': 50,
        'pc70': 70,
        'log2_1.2': np.log2(1.2),
        'log10_1.2': np.log10(1.2),
        'log2_2': np.log2(2),
        'log10_2': np.log10(2),
        'neglog2_0.88': -1 * np.log2(0.88),
        'coff_2.32': 2.32,
        'fc0.2': 0.2,
        'fc0.3': 0.3,
        'fc0.5': 0.5,
        'pc05': 5,
        'pc10': 10,
        'pc25': 25,
        'pc30': 30,
        'pc95': 95,
        'bmad1': bmad,
        'bmad2': bmad * 2,
        'bmad3': bmad * 3,
        'bmad4': bmad * 4,
        'bmad5': bmad * 5,
        'bmad6': bmad * 6,
        'bmad10': bmad * 10,
        # 'maxmed20pct': lambda df: df['max_med'].aggregate(lambda x: np.max(x) * 0.20),  # is never used
    }.get(mthd)
