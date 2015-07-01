import sys
import time as tm
import numpy as np
import ConfigParser


def getlist(option, sep=',', chars=None):
    """Return a list from a ConfigParser option. By default,
       split on a comma and strip whitespaces."""
    return [float(chunk.strip(chars)) for chunk in option.split(sep)]


class dnf:
    def __init__(self, configFileName):
        config = ConfigParser.RawConfigParser()
        config.read(configFileName)

        self.n = config.getint('Model', 'numNeurons')
        self.m, self.k = self.n//6, 5*self.n//6

        self.x_inf = config.getfloat('Model', 'x_inf')
        self.x_sup = config.getfloat('Model', 'x_sup')
        self.tau1 = config.getfloat('Model', 'tau1')
        self.tau2 = config.getfloat('Model', 'tau2')
        self.axonalVelGS = config.getfloat('Model', 'axonalvelocityGS')
        self.axonalVelSG = config.getfloat('Model', 'axonalvelocitySG')
        self.axonalVelGG = config.getfloat('Model', 'axonalvelocityGG')
        self.K = getlist(config.get('Model', 'SynapticStrength'))
        self.S = getlist(config.get('Model', 'SynapticVariance'))

        self.Wcx = config.getfloat('Model', 'synapticstrengthCx')
        self.Wstr = config.getfloat('Model', 'synapticstrengthStr')

        self.switch = config.getint('DBS', 'switch')
        self.Kc = config.getfloat('DBS', 'Kc')
        self.tau = config.getint('DBS', 'tau')

        self.l = (self.x_sup - self.x_inf)
        self.dx = self.l/float(self.n)
        self.mean = self.l/2

        self.norm()
        self.printParams()

    def printParams(self):
        print 'Model Parameters'
        print '----------------'
        print "Number of neurons: {}".format(self.n)
        print "Domain Omega: [{}, {}]".format(self.x_inf, self.x_sup)
        print "Length of domain: {}".format(self.l)
        print "Synaptic decay times: {} and {}".format(self.tau1, self.tau2)
        print "Axonal transmission velocity G->S: {}".format(self.axonalVelGS)
        print "Axonal transmission velocity S->G: {}".format(self.axonalVelSG)
        print "Axonal transmission velocity G->G: {}".format(self.axonalVelGG)
        print "Synaptic stregnths: {}, {}, {}".format(self.K[0], self.K[1],
                                                      self.K[2])
        print "Synaptic variances: {}, {}, {}".format(self.S[0], self.S[1],
                                                      self.S[2])
        print 'DBS Parameters'
        print '--------------'
        print "DBS Gain (Kc): {}".format(self.Kc)
        print "DBS control signal time constant: {}".format(self.tau)
        print "-------------------------------------------------------------"

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
        dx = self.l/float(N)
        d = self.build_distances(N, 0.0, 0.0, 1.0)
        norm = np.zeros((len(self.K), ))

        for i in range(len(self.K)):
            tmp = (self.K[i] * self.g(d, self.S[i]))**2
            norm[i] = tmp.sum() * dx * dx
        print "Norm W22: {}".format(norm[2])

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

    def run(self, tf, dt, procDelay):
        np.random.seed(62)
        """ Run a simulation """
        n, m, k = self.n, self.m, self.k

        # Total simulation time
        simTime = int(tf/dt)

        # Returns the three synaptic connections kernels
        W12, W21, W22, delays = self.build_kernels()

        # Compute delays by dividing distances by axonal velocity
        delays12 = np.floor(delays[0]/self.axonalVelGS)
        delays21 = np.floor(delays[1]/self.axonalVelSG)
        delays22 = np.floor(delays[2]/self.axonalVelGG)
        maxDelay = int(max(delays12[0].max(), delays21[0].max(),
                           delays22[0].max()))

        # Set the initial conditions and the history
        self.initial_conditions(simTime)

        # Initialize the cortical and striatal inputs
        Cx = 0.026 * self.Wcx
        Str = 0.002 * self.Wstr

        # Presynaptic activities
        pre12, pre21, pre22 = np.empty((m,)), np.empty((m,)), np.empty((m,))

        # DBS signals
        # A is a gaussian that defines spatialy the stimulation zone
        x = np.linspace(self.x_inf, self.x_sup, n)
        tmp = self.g(x-0.5, .09)
        A = np.zeros((m, ))
        A = tmp[(n-10)//2:(n+10)//2]

        # Xref is the reference signal
        Xref = np.zeros((m, ))

        # Simulation
        U, U_ = np.zeros((m, )), np.zeros((simTime, m))
        t0 = tm.time()
        for i in range(maxDelay, simTime):
            if i*dt > 500:
                U_[i] = self.Kc * (self.X1[i-procDelay, :m] - Xref)
                U = self.switch * A * U_[i]

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
                              self.S1(-pre12 + Cx - U)) * dt/self.tau1)
            self.X2[i, k:] = (self.X2[i-1, k:] + (-self.X2[i-1, k:] +
                              self.S2(pre21 - pre22 - Str))*dt/self.tau2)
        t1 = tm.time()
        print "Simulation time: {} sec".format(t1-t0)
        return U_


if __name__ == '__main__':
    if len(sys.argv) == 3:
        config = ConfigParser.RawConfigParser()
        config.read(sys.argv[1])

        tf = config.getfloat('Time', 'tf')
        dt = config.getfloat('Time', 'dt')

        sim = dnf(sys.argv[1])

        damagePercent = [1, 3, 5, 7, 8, 9, 10, 13, 15, 20]
        for i in damagePercent:
            sim.run(tf, dt, i)
            np.save(sys.argv[2]+"solution12_"+str(i), sim.X1)

    else:
        print "Parameters file {} does not exist!".format(sys.argv[1])
