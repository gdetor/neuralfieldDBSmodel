import numpy as np
import matplotlib.pylab as plt

if __name__ == '__main__':
    x = np.load('bruteforcepms.npy')

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111)

    ax.hist(x, bins=25, align='mid', color=(0.5, 0., 0.), alpha=0.7)
    ax.set_ylim([0, 1000])
    ax.set_xlabel('Frequency (Hz)', fontsize=18, weight='bold')
    ax.set_ylabel('Hits', fontsize=18, weight='bold')

    ticksx, tickslx = np.arange(0, 30, 5), np.arange(0, 30, 5)
    ticksy, ticksly = np.arange(0, 1250, 250), np.arange(0, 1250, 250)
    ax.set_xticks(ticksx)
    ax.set_yticks(ticksy)
    ax.set_xticklabels(tickslx, fontsize=15, weight='bold')
    ax.set_yticklabels(ticksly, fontsize=15, weight='bold')

    ax.get_xaxis().set_tick_params(which='both', direction='out')
    ax.get_yaxis().set_tick_params(which='both', direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_tick_params(which='major', size=8, width=1.2)
    # ax.xaxis.set_tick_params(which='minor', width=1.2, size=5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_tick_params(which='major', size=8, width=1.2)

    plt.savefig('Figure110.pdf')
    plt.show()
