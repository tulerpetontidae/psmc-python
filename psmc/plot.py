import matplotlib.pyplot as plt
import numpy as np 
from matplotlib.ticker import ScalarFormatter


def plot_history(psmc, data=None, sim_history=None, th=20, n0_sim = 1e4, axs=None, color='dodgerblue', label='Estimated history'):
    """
    Plot the estimated population history using PSMC and optionally overlay the real history from a simulation.

    Args:
        psmc: A PSMC object with the estimated demographic history.
        data: An input data array used to estimate the parameters (batch, seq_length).
        sim_history: A 2D numpy array with the real demographic history from a simulation, in the format (time, population size).
                     If not provided, the real history will not be plotted.
        th: A threshold for the confidence intervals of the estimated history, above which the confidence intervals will be colored red.
        n0_sim: The N0 effective population size of the simulated population.

    Returns:
        A Matplotlib figure with the population history plot.

    """
    xs = psmc.t * 2 * 25 * psmc.N0 / 100
    xs[-1] = 1e10
    ys = psmc.map_lam(psmc.lam) * psmc.N0 / 100
    ys = np.append(ys, ys[-1])
    if data is not None:
        es = psmc.sigma * (data.shape[0] * data.shape[1]) / psmc.C_sigma

    
    if axs is None:
        fig, axs = plt.subplots(1,1, figsize=(6,4), dpi=150)
    
    if sim_history is not None:
        sim_xs = sim_history[:,0]  * 4 * 25 * n0_sim
        sim_xs = np.append(sim_xs, 1e8) 
        sim_ys = sim_history[:,1] * n0_sim
        sim_ys = np.append(sim_ys, sim_ys[-1]) 

        axs.step(sim_xs, sim_ys, where='post', linestyle='--', color='k', label = "Real history", alpha=0.9)
    if data is not None:
        for i in range(len(xs)-1):
            plt.axvspan(xs[i], xs[i+1], alpha=0.1, edgecolor='none', facecolor=('none' if es[i]>th else 'tomato'))
    axs.step(xs, ys, where='post', linestyle='-', lw=2, color=color, label = label)

    if data is not None:
        axs.axvspan(0, 0, alpha=0.1, edgecolor='none', facecolor='tomato', label = 'Potential overfitting')
    axs.set_xscale('log')
    axs.set_ylim(0, 5e4)
    axs.set_xlim(1e3, 1e7)
    axs.set_title('Population History')
    axs.set_xlabel('Years, $(g=25, \mu=2.5 × 10^{-8})$')
    axs.set_ylabel('Effective population size')

    axs.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    axs.yaxis.offsetText.set_visible(True)
    axs.ticklabel_format(style='sci', axis='y', scilimits=(4,4))

    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.legend() 