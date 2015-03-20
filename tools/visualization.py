import numpy as np
import matplotlib.pylab as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

majorLocator = MultipleLocator(2)
majorFormatter = FormatStrFormatter('%d')
minorLocator = MultipleLocator(0.5)


def prettyPlot1D(x, y):
    """ Target neurons rate evolution over time plot function """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, 'c', lw=2)
    ax.plot(y, 'm', lw=2)


def prettyPlot2D(x, index):
    """ All neurons rates over time plot function """
    t, n = x.shape[0], x.shape[1]
    tmp = np.zeros((t, 1 + n//2))
    tmp[:, :n//4] = x[:, :n//4]
    tmp[:, n//4+1:] = x[:, 3*n//4:]

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111)
    im = ax.imshow(x, interpolation='nearest', cmap=plt.cm.hot,
                   origin='upper', aspect='auto', vmin=0, vmax=x.max())

    ax.get_xaxis().set_tick_params(which='both', direction='out')
    ax.xaxis.set_major_locator(majorLocator)
    ax.xaxis.set_major_formatter(majorFormatter)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.set_xticks([-0.5, 15, 31.5])
    ax.set_xticklabels(['0', '0.5', '1'], fontsize=15, weight='bold')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_tick_params(width=1)
    ax.xaxis.set_tick_params(size=7)

    ax.get_yaxis().set_tick_params(which='both', direction='out')
    ax.set_yticks([0, 500, 1000])
    ax.set_yticklabels(['0', '10', '20'], fontsize=15, weight='bold')
    ax.yaxis.set_tick_params(size=7)
    ax.yaxis.set_ticks_position('left')

    ax.set_xlabel('Space (mm)', fontsize=20, weight='bold')
    ax.set_ylabel('Time (s)', fontsize=20, weight='bold')

    plt.colorbar(im)
    # plt.savefig('soltrace'+index+'.pdf')


def loadData(experiments, filenames):
    """ Loads the data from npy files """
    data = []
    for j in range(len(experiments)):
        for i in range(len(filenames)):
            print experiments[j], filenames[i]
            data.append(np.load(experiments[j]+filenames[i]))
    return np.array(data)


def plotFiringRate(x1, x2, index):
    """ Firing rates plot function """
    majorLocator = MultipleLocator(0.01)
    minorLocator = MultipleLocator(0.005)
    n = x1.shape[1]
    dx = 1.0/float(n//4)
    fr1 = x1.sum(axis=1) * dx / 1.0
    fr2 = x2.sum(axis=1) * dx / 1.0

    print fr1.max() - np.abs(fr1.min())
    print fr2.max() - np.abs(fr2.min())

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.plot(fr1, 'k', lw=2, zorder=0)
    ax.plot(fr2, 'r', lw=2, alpha=0.8, zorder=5)
    ax.axvline(500, c='k', ls='--', lw=2.7)

    ax.set_xlabel('Time (ms)', fontsize=18, weight='bold')
    ax.set_ylabel('Frequency (sp/ms)', fontsize=18, weight='bold')

    ax.get_xaxis().set_tick_params(which='both', direction='out')
    ax.get_yaxis().set_tick_params(which='both', direction='out')
    ax.yaxis.set_major_locator(majorLocator)
    ax.yaxis.set_minor_locator(minorLocator)
    ax.set_xticks([0, 500, 1000])
    ax.set_xticklabels(['0', '500', '1000'], fontsize=15, weight='bold')
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_tick_params(which='major', width=1.2, size=7)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_tick_params(which='major', size=8, width=1.2)
    ax.yaxis.set_tick_params(which='minor', width=1.2, size=5)
    ax.set_yticks([-0.05, 0.0, 0.15])
    ax.set_yticklabels(['-0.05', '0.0', '0.15'], fontsize=15, weight='bold')

    # plt.savefig('firing-rates'+index+'.pdf')


def plotControlSignal(x, index):
    majorLocator = MultipleLocator(0.01)
    minorLocator = MultipleLocator(0.01)

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111)
    ax.plot(x, lw=1.5, alpha=0.6)

    ax.set_xlabel('Time (ms)', fontsize=18, weight='bold')
    ax.set_ylabel(r'${\bf u(r,t)}$', fontsize=18, weight='bold')

    ax.get_xaxis().set_tick_params(which='both', direction='out')
    ax.get_yaxis().set_tick_params(which='both', direction='out')
    ax.yaxis.set_major_locator(majorLocator)
    ax.yaxis.set_minor_locator(minorLocator)
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_tick_params(which='major', width=1.2, size=7)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_tick_params(which='major', size=7, width=1.2)
    ax.yaxis.set_tick_params(which='minor', width=1.2, size=3)

    ax.set_yticks([-0.05, 0.0, 0.30])
    ax.set_yticklabels(['-0.05', '0.0', '0.30'], fontsize=15, weight='bold')
    ax.set_xticks([0, 500, 1000])
    ax.set_xticklabels(['0', '500', '1000'], fontsize=15, weight='bold')

    # plt.savefig(index+'control.pdf')


if __name__ == '__main__':
    print 'Plotting results!'

    path = '../data/'
    filenames = ['solution1.npy', 'solution2.npy']

    experiments = [path+'protocolA']
    # experiments = ['protocolB']
    # experiments = ['protocolC']
    # experiments = ['protocolD']
    # experiments = ['protocolD2']

    data = loadData(experiments, filenames)

    ii = 0
    for i in range(0, len(experiments)+1, 2):
        plotFiringRate(data[i], data[i+1], experiments[ii])
        # prettyPlot2D(data[i]+data[i+1], experiments[i])
        control = np.load(path+experiments[i]+'control.npy')
        plotControlSignal(control, experiments[i])
        ii += 1

    plt.show()
