import numpy as np
import matplotlib.pylab as plt


if __name__ == '__main__':
    T = 600
    damagePercent = [1, 3, 5, 7, 8, 9, 10, 13, 15, 20]
    percent = len(damagePercent)

    # res1, res2 = np.zeros((percent, )), np.zeros((percent, ))
    # for i, l in enumerate(damagePercent):
    #     x1 = np.load('protocolDelayssolution'+str(l)+'.npy')
    #     n = x1.shape[1]
    #     dx = 1.0/float(n//4)
    #     fr = x1.sum(axis=1) * dx / 1.0

    #     res1[i] = fr[T:].max() - fr[T:].min()
    #     res2[i] = fr[100:400].max() - fr[100:400].min()

    res1, res2 = np.zeros((3, percent)), np.zeros((3, percent))
    for j, k in enumerate([2, 6, 12]):
        for l, i in enumerate(damagePercent):
            x1 = np.load('protocolDelayssolution'+str(k)+'_'+str(i)+'.npy')
            n = x1.shape[1]
            dx = 1.0/float(n//6)
            fr = x1[:, :n//6].sum(axis=1) * dx

            res1[j, l] = fr[T:].max() - fr[T:].min()
            res2[j, l] = fr[100:400].max() - fr[100:400].min()
            print res1[j, l] * 1000.0, res2[j, l]*1000.0

    n = np.array(damagePercent)
    fig = plt.figure(figsize=(8, 7.5))
    ax = fig.add_subplot(111)

    al = [0.45, 0.65, 1.0]
    K = [2, 6, 12]
    for i in range(res1.shape[0]):
        ax.plot(n, res1[i], 'k-o', lw=2, alpha=al[i])
        ax.plot(n, res2[i], 'r', ls='--', lw=2)

    ax.text(9.0, 0.1, r"$k_c=$"+str(K[2]),
            ha='left',
            va='top',
            fontsize=12,
            weight='bold')

    ax.text(11.0, 0.08, r"$k_c=$"+str(K[1]),
            ha='left',
            va='top',
            fontsize=12,
            weight='bold')

    ax.text(13.0, 0.07, r"$k_c=$"+str(K[0]),
            ha='left',
            va='top',
            fontsize=12,
            weight='bold')

    ax.set_xlim([0.5, 20.3])
    ax.set_ylim([-0.001, 0.20])

    ticks = ax.set_xticks(damagePercent)
    tickl = ('1', '3', '5', '7', '8', '9', '10', '13', '15', '20')
    ax.set_xticklabels(tickl, fontsize=15, weight='bold')
    ax.set_xlabel('Delay (ms)',
                  fontsize=14,
                  weight='bold')

    ax.set_yticks(np.arange(0.0, 0.25, 0.05))
    ticks = [str(i) for i in np.arange(0, 250, 50)]
    ax.set_yticklabels(ticks, fontsize=15, weight='bold')
    ax.set_ylabel('Maximum Oscillations Amplitude',
                  fontsize=14,
                  weight='bold')

    ax.get_xaxis().set_tick_params(which='both', direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_tick_params(which='major', width=1.2, size=7)
    ax.xaxis.set_tick_params(which='major', size=8, width=1.2)
    ax.xaxis.set_tick_params(which='minor', width=1.2, size=5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_tick_params(which='major', size=8, width=1.2)
    ax.yaxis.set_tick_params(which='minor', width=1.2, size=5)

    plt.savefig('Figure9.pdf')
    plt.show()
