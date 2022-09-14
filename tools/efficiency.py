import numpy as np
import matplotlib.pylab as plt


if __name__ == '__main__':
    T = 800
    percent = 5

    res1, res2 = [], []
    for k in range(2, 8, 2):
        for i in range(0, 10, 2):
            print(k, i)
            x1 = np.load('protocolEfficiencysolution1'+str(i)+str(k)+'.npy')
            n = x1.shape[1]
            dx = 1.0/float(n//4)
            fr = x1.sum(axis=1) * dx / 1.0

            print(fr[T:].max(), fr[T:].min())

            res1.append(np.abs(np.abs(fr[T:].max()) - np.abs(fr[T:].min())))
            res2.append(np.abs(np.abs(fr[100:400].max()) -
                               np.abs(fr[100:400].min())))
    res1, res2 = np.array(res1).reshape(3, 5), np.array(res2)

    n = np.arange(0, 10, 2)

    fig = plt.figure(figsize=(8, 7.5))
    ax = fig.add_subplot(111)

    al = [0.45, 0.65, 1.0]
    K = [2, 4, 6]
    for i in range(res1.shape[0]):
        ax.plot(n, res1[i], 'k-o', lw=2, alpha=al[i])
        ax.axhline(res2[i], c='r', ls='--', lw=2)
    ax.text(7.0, res1[0, 4]+0.0012, r"$k_c=$"+str(K[0]),
            ha='left',
            va='top',
            fontsize=12,
            weight='bold')

    ax.text(7.0, res1[1, 4]+0.0014, r"$k_c=$"+str(K[2]),
            ha='left',
            va='top',
            fontsize=12,
            weight='bold')

    ax.text(7.0, res1[2, 4]-0.001, r"$k_c=$"+str(K[1]),
            ha='left',
            va='top',
            fontsize=12,
            weight='bold')

    ax.set_xlim([-0.1, 8.5])
    ax.set_ylim([0, 0.025])

    ax.set_xticks(np.arange(0, 10, 2))
    ticks = [str(i) for i in range(0, 125, 25)]
    ax.set_xticklabels(ticks, fontsize=15, weight='bold')
    ax.set_xlabel(r'Percentage of $\alpha(r)$ degeneracy (%)',
                  fontsize=14,
                  weight='bold')

    ax.set_yticks(np.arange(0.0, 0.035, 0.005))
    ticks = [str(i) for i in np.arange(0, 40, 5)]
    ax.set_yticklabels(ticks, fontsize=15, weight='bold')
    ax.set_ylabel('Maximum Oscillations Amplitude (MOA) after DBS',
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

    plt.savefig('Figure5.pdf')
    plt.show()
