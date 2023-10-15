import numpy as np
import matplotlib.pyplot as plt

def constant(x):
    return np.zeros_like(x)

def linear(x, a):
    return a * x

def quadratic(x, a, b):
    return a * (x / b + (x / b)**2)

def power(x, a, p):
    return a * x**p

def hill(x, tp, ga, p):
    return tp / (1 + (ga / x)**p)

def gain_loss(x, tp, ga, la, p, q):
    return tp / ((1 + (ga / x)**p) * (1 + (x / la)**q))

def gain_loss_2(x, tp, ga, p, q):
    return (tp / (1 + (ga / x)**p)) * np.exp(-q * x)

def exponential2(x, a, b):
    return a * (np.exp(x / b) - 1)

def exponential3(x, a, b, p):
    return a * (np.exp((x / b)**p) - 1)

def exponential4(x, tp, ga):
    return tp * (1 - 2**(-x / ga))

def exponential5(x, tp, ga, p):
    return tp * (1 - 2**(-(x / ga)**p))


def plot_curve(subplot_position, curve_function, title, *parameters):
    plt.subplot(3, 4, subplot_position)
    plt.plot(np.log10(x), curve_function(x, *parameters))
    plt.title(title)
    plt.gca().set_yticklabels([])

x = np.logspace(-2, 2, 1000)

# Create plots for each model
plt.figure(figsize=(12, 8))

params = {
    'poly1': {'a': 1},
    'poly2': {'a': 0.8376534517925958, 'b': 200.00025976848568},
    'power': {'a': 0.3542301680340536, 'p': 0.3},
    'exp2': {'a': 0.9838224625966211, 'b': 200.00847772171898},
    'exp4': {'tp': 1.2897200087401082, 'ga': 1.1237571655438394},
    'exp5': {'tp': 1.2964252419708209, 'ga': 1.132704718398446, 'p': 1.1111464091067835},
    'hill': {'tp': 1.5, 'ga': 8, 'p': 3},
    'sigmoid': {'tp': 2, 'ga': 10, 'p': 2, 'q': 0.01},
    'gnls': {'tp': 1, 'ga': 1, 'p': 1, 'la': 1, 'q': 1}
}

plot_curve(1, linear, 'Poly1', params['poly1']['a'])
plot_curve(2, quadratic, 'Poly2', params['poly2']['a'], params['poly2']['b'])
plot_curve(3, power, 'Power', params['power']['a'], params['power']['p'])
plot_curve(4, exponential2, 'Exp2', params['exp2']['a'], params['exp2']['b'])
plot_curve(10, exponential3, 'Exp3', params['exp5']['tp'], params['exp5']['ga'], params['exp5']['p'])
plot_curve(5, exponential4, 'Exp4', params['exp4']['tp'], params['exp4']['ga'])
plot_curve(6, exponential5, 'Exp5', params['exp5']['tp'], params['exp5']['ga'], params['exp5']['p'])
plot_curve(7, hill, 'Hill', params['hill']['tp'], params['hill']['ga'], params['hill']['p'])
plot_curve(8, gain_loss_2, 'Sigmoid', params['sigmoid']['tp'], params['sigmoid']['ga'], params['sigmoid']['p'], params['sigmoid']['q'])
plot_curve(9, gain_loss, 'GNLS', params['gnls']['tp'], params['gnls']['ga'], params['gnls']['p'], params['gnls']['la'], params['gnls']['q'])

plt.ylim(0, 100)

# Set x-axis labels to -2 to 2
plt.xticks([0.01, 0.1, 1, 10, 100], ['-2', '-1', '0', '1', '2'])

plt.tight_layout()
plt.show()