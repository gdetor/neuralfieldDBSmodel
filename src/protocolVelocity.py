from __future__ import division
import sys
import numpy as np
import configparser
from numpy.fft import rfft
from numpy import polyfit, arange
from scipy.signal import blackmanharris, detrend


def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.
    f is a vector and x is an index for that vector.

    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.

    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.

    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]

    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)

    """
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)


def parabolic_polyfit(f, x, n):
    """Use the built-in polyfit() function to find the peak of a parabola
    f is a vector and x is an index for that vector.

    n is the number of samples of the curve used to fit the parabola.
    """
    a, b, c = polyfit(arange(x-n//2, x+n//2+1), f[x-n//2:x+n//2+1], 2)
    xv = -0.5 * b/a
    yv = a * xv**2 + b * xv + c
    return (xv, yv)


def getlist(option, sep=',', chars=None):
    """Return a list from a ConfigParser option. By default,
       split on a comma and strip whitespaces."""
    return [float(chunk.strip(chars)) for chunk in option.split(sep)]


class dnf:
    def __init__(self, configFileName):
        config = configparser.RawConfigParser()
        config.read(configFileName)

        self.n = config.getint('Model', 'numNeurons')
        self.m, self.k = self.n//6, 5*self.n//6

        self.x_inf = config.getfloat('Model', 'x_inf')
        self.x_sup = config.getfloat('Model', 'x_sup')
        self.tau1 = config.getfloat('Model', 'tau1')
        self.tau2 = config.getfloat('Model', 'tau2')
        self.K = getlist(config.get('Model', 'SynapticStrength'))
        self.S = getlist(config.get('Model', 'SynapticVariance'))

        self.Wcx = config.getfloat('Model', 'synapticstrengthCx')
        self.Wstr = config.getfloat('Model', 'synapticstrengthStr')

        self.switch = config.getint('DBS', 'switch')
        self.Kc = config.getfloat('DBS', 'Kc')
        self.tau = config.getint('DBS', 'tau')

        self.len = (self.x_sup - self.x_inf)
        self.dx = self.len/float(self.n)
        self.mean = self.len/2

        self.norm()
        self.printParams()

    def printParams(self):
        print('Model Parameters')
        print('----------------')
        print("Number of neurons: {}".format(self.n))
        print("Domain Omega: [{}, {}]".format(self.x_inf, self.x_sup))
        print("Length of domain: {}".format(self.len))
        print("Synaptic decay times: {} and {}".format(self.tau1, self.tau2))
        print("Synaptic stregnths: {}, {}, {}".format(self.K[0], self.K[1],
                                                      self.K[2]))
        print("Synaptic variances: {}, {}, {}".format(self.S[0], self.S[1],
                                                      self.S[2]))
        print('DBS Parameters')
        print('--------------')
        print("DBS Gain (Kc): {}".format(self.Kc))
        print("DBS control signal time constant: {}".format(self.tau))
        print("-------------------------------------------------------------")

    def S1(self, x):
        """ Sigmoid function of population #1 """
        # return 1.0/(1.0 + np.exp(-x)) - 0.5
        return 0.3/(1 + np.exp(-4.*x/0.3) * 283./17.)

    def S2(self, x):
        """ Sigmoid function of population #2 """
        # return 1.0/(1.0 + np.exp(-x)) - 0.5
        return 0.4/(1 + np.exp(-4.*x/0.4) * 325./75.)

    def gaussian(self, x, sigma=1.0):
        ''' Gaussian function '''
        return (1.0/(np.sqrt(2*np.pi)*sigma))*np.exp(-.5*(x/sigma)**2)
        # return np.exp(-0.5*(x/sigma)**2)

    def g(self, x, sigma):
        return np.exp(-.5*(x/sigma)**2)

    def build_distances(self, nodes, mean, x_inf, x_sup):
        """ Computes all the possible distances between units """
        X, Y = np.meshgrid(np.linspace(x_inf, x_sup, nodes),
                           np.linspace(x_inf, x_sup, nodes))
        D = abs((X-mean) - (Y-mean))
        return D

    def norm(self):
        """ Computes the norms """
        N = 5000
        dx = self.len/float(N)
        d = self.build_distances(N, 0.0, 0.0, 1.0)
        norm = np.zeros((len(self.K), ))

        for i in range(len(self.K)):
            tmp = (self.K[i] * self.g(d, self.S[i]))**2
            norm[i] = tmp.sum() * dx * dx
        print("Norm W22: {}".format(norm[2]))

    def build_kernels(self):
        """ Build the synaptic connectivity matrices """
        n = self.n
        # Compute all the possible distances
        dist = [self.build_distances(n, 0.917, 0.0, 1.0),
                self.build_distances(n, 0.083, 0.0, 1.0),
                self.build_distances(n, 0.912, 0.83, 1.0)]

        # Create a temporary vector containing gaussians
        g = np.empty((len(self.K), n, n))
        for j in range(len(self.K)):
            for i in range(n):
                # g[j, i] = self.K[j] * self.gaussian(dist[i], self.S[j])
                g[j, i] = self.K[j] * self.g(dist[j][i], self.S[j])
            g[j, self.m:self.k] = 0.0

        # GPe to STN connections
        W12 = np.zeros((n, n))
        W12[:self.m, self.k:] = g[0, self.k:, self.k:]

        # STN to GPe connections
        W21 = np.zeros((n, n))
        W21[self.k:, :self.m] = g[1, :self.m, :self.m]

        # GPe to GPe connections
        W22 = np.zeros((n, n))
        W22[self.k:, self.k:] = g[2, self.k:, self.k:]
        np.fill_diagonal(W22, 0.0)

        return W12, W21, W22, dist

    def initial_conditions(self, time):
        """ Set the initial conditions """
        n = self.n
        self.X1 = np.zeros((time, n))
        self.X2 = np.zeros((time, n))

    def run(self, tf, dt, c1, c2):
        np.random.seed(62)
        """ Run a simulation """
        n, m, k = self.n, self.m, self.k

        # Total simulation time
        simTime = int(tf/dt)

        # Returns the three synaptic connections kernels
        W12, W21, W22, delays = self.build_kernels()

        # Compute delays by dividing distances by axonal velocity
        delays12 = np.floor(delays[0]/c2).astype('i')
        delays21 = np.floor(delays[1]/c1).astype('i')
        delays22 = np.floor(delays[2]/c2).astype('i')
        maxDelay = int(max(delays12[0].max(), delays21[0].max(),
                           delays22[0].max()))

        # Set the initial conditions and the history
        self.initial_conditions(simTime)

        # Initialize the cortical and striatal inputs
        Cx = 0.5
        Str = 0.4

        # Presynaptic activities
        pre12, pre21, pre22 = np.empty((m,)), np.empty((m,)), np.empty((m,))

        # Simulation
        for i in range(maxDelay, simTime):
            # Take into account the history of rate for each neuron according
            # to its axonal delay
            for idxi, ii in enumerate(range(m)):
                mysum = 0.0
                for jj in range(k, n):
                    mysum += (W12[ii, jj] *
                              self.X2[i-delays12[ii, jj], jj])*self.dx
                pre12[idxi] = mysum

            for idxi, ii in enumerate(range(k, n)):
                mysum = 0.0
                for jj in range(0, m):
                    mysum += (W21[ii, jj] *
                              self.X1[i-delays21[ii, jj], jj])*self.dx
                pre21[idxi] = mysum

            for idxi, ii in enumerate(range(k, n)):
                mysum = 0.0
                for jj in range(k, n):
                    mysum += (W22[ii, jj] *
                              self.X2[i-delays22[ii, jj], jj])*self.dx
                pre22[idxi] = mysum

            # Forward Euler step
            self.X1[i, :m] = (self.X1[i-1, :m] + (-self.X1[i-1, :m] +
                              self.S1(-pre12 + Cx)) * dt/self.tau1)
            self.X2[i, k:] = (self.X2[i-1, k:] + (-self.X2[i-1, k:] +
                              self.S2(pre21 - pre22 - Str))*dt/self.tau2)
        dx = 1.0/float(m)
        fr = self.X1.sum(axis=1) * dx / 1.0

        signal = detrend(fr)
        windowed = signal * blackmanharris(len(signal))
        f = rfft(windowed)
        i = np.argmax(np.abs(f))
        # true_i = parabolic(np.log(np.abs(f)), i)[0]
        return i


if __name__ == '__main__':
    if len(sys.argv) == 3:
        config = configparser.RawConfigParser()
        config.read(sys.argv[1])

        tf = config.getfloat('Time', 'tf')
        dt = config.getfloat('Time', 'dt')

        sim = dnf(sys.argv[1])

        n = 100
        c1 = np.linspace(0.01, 0.95, n)
        c2 = np.linspace(0.01, 0.95, n)

        ampl = np.empty((n, n))
        for i, j in enumerate(c1):
            for k, l in enumerate(c2):
                ampl[i, k] = (sim.run(tf, dt, j, l))
        np.save('velocity', ampl)

    else:
        print("Parameters file {} does not exist!".format(sys.argv[1]))
