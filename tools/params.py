import numpy as np
import matplotlib.pylab as plt


def plotFreqVel(filenames):
    fig = plt.figure(figsize=(25, 8.5))

    xlabels = [r'$K_{12}$', r'$c_{1}$']
    ylabels = [r'$K_{21}$', r'$c_{2}$']
    points = [np.array([29, 37]), np.array([17, 8])]
    ticks = [np.arange(0, 110, 25.0), np.arange(0, 110, 25)]
    ticksl = [np.arange(0, 110.0, 25.0), np.arange(0, 2.5, 0.5)]

    for i, j in enumerate(filenames):
        x = np.load(j)
        n = x.shape[0]

        ax = fig.add_subplot(1, 2, i+1)
        pts = points[i]  # np.array([[5, 30], [5, 38]])

        im = ax.imshow(x, interpolation='bicubic', cmap=plt.cm.Blues,
                       origin='lower', vmin=0, vmax=x.max())
        plt.colorbar(im)
        ax.set_xticks(ticks[i])
        ax.set_yticks(ticks[i])
        ax.set_xticklabels(ticksl[i], fontsize=15, weight='bold')
        ax.set_yticklabels(ticksl[i], fontsize=15, weight='bold')
        ax.set_xlabel(xlabels[i], fontsize=23, weight='black')
        ax.set_ylabel(ylabels[i], fontsize=23, weight='black')

        if i == 1:
            ax.contour(x, colors=['red'], levels=[14], ls=2, linewidths=2)
            # ax.contour(x, colors=['yellow'], levels=[31], linewidths=3)
        else:
            ax.contour(x, colors=['red'], levels=[14],
                       linewidths=2)
        ax.scatter(pts[0], pts[1], c='k', s=40, marker='x')
        ax.set_xlim([0, n])
        ax.set_ylim([0, n])
        ax.text(pts[0], pts[1],
                str(np.round(x[pts[0], pts[1]], 1)),
                va='top',
                ha='left',
                fontsize=15,
                weight='bold',
                color='black')

        ax.get_xaxis().set_tick_params(which='both', direction='out')
        ax.get_yaxis().set_tick_params(which='both', direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_tick_params(which='major', size=8, width=1.2)
        # ax.xaxis.set_tick_params(which='minor', width=1.2, size=5)
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)

        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_tick_params(which='major', size=8, width=1.2)


if __name__ == '__main__':
    fnames = ['weightsw12w21.npy', 'velocity.npy']

    plotFreqVel(fnames)
    plt.savefig('Figure2.pdf')
    plt.show()
