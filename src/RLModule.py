# RLModule is responsible to decide the action
import abc
import numpy as np


# The base class of RL
class RLBase(object):
    __metaclass__ = abc.ABCMeta

    # The discount factor
    gamma = 0.9

    # The action set (prices goes from the smallest to the largest)
    ActionSet = [0.1, 1, 5, 10]

    @abc.abstractmethod
    def decide(self, start = False):
        """Decide the action a_t"""
        return

    @abc.abstractmethod
    def observe(self, a, r, s, start = False, terminal = False):
        """Observe <reward_t, state_t+1> """
        return


# The simple RL module
class Simple(RLBase):
    z=[0,0,0,0]

    def decide(self, start = False):
        return self.ActionSet[1]

    def observe(self, a, r, s, start = False, terminal = False):
        pass


# Gaussian Process SARSA
class GpSarsa(RLBase):

    def __init__(self, len_state: int):
        # The noisy level of the Gaussian process
        self.sigma = 0.02
        # Observation history
        self.Hist = []  # <State, Action>
        self.len_state = len_state
        self.rHist = []  # Reward
        # The covariance matrix
        self.Cov = []
        # Current State
        self.S = []
        # H matrix
        self.invH = []
        # Gaussian Process Parameters
        self.Alpha = None
        self.C = None

    def decide(self):
        """We use Thompson sampling to decide the next-step action."""
        if len(self.Hist) == 0:
            '''At the first step, choose the lowest price'''
            return RLBase.ActionSet[0]
        else:
            '''Firstly, update the Gaussian process model'''
            self.gpRegression()
            '''Secondly, compute the prediction of the Gaussian process'''
            q = []
            z = self.S.copy()
            z.append(self.ActionSet[0])
            for a in RLBase.ActionSet:
                z[-1] = a
                q.append(self.gpPredict(z))
            '''Thirdly, select the action with the largest Q value'''
            pos = q.index(max(q))
            return RLBase.ActionSet[pos]

    def observe(self, a, r, s, start = False, terminal = False):
        """Observe the environment change after action a"""
        '''Add the data to the observation history'''
        if len(self.S)>0:
            self.Hist.append(list(self.S)+[a])
            self.rHist.append(r)
        '''Update the current state'''
        self.S = list(s.copy())

    # noinspection PyUnresolvedReferences
    def kernel(self, z1: np.array, z2: np.array) -> float:
        x1 = np.mean(z1[0: self.len_state: 2])
        x2 = np.mean(z1[1: self.len_state: 2])
        x3 = np.mean(z2[0: self.len_state: 2])
        x4 = np.mean(z2[1: self.len_state: 2])
        dist_w = np.exp(-1.0*((x1 - x3)**2 + (x2 - x4)**2)) # Here, 4 = 2^2
        x5 = z1[self.len_state] - z2[self.len_state]
        dist_A = np.exp(-1.0*(x5**2)) # Here, 0.25 = 0.5^2
        return dist_A*dist_w
        # diff = (z1 - z2)**2
        # '''The difference between different workers'''
        # distW = np.exp(-4*(diff[0:-1:2]+diff[1:-1:2]))
        # '''The difference between actions'''
        # distA = np.exp(-0.01*diff[-1])
        # noinspection PyTypeChecker
        # return np.sum(distW)*distA
        #return np.exp(-0.5*np.sum(np.abs(diff)))

    def gpRegression(self):
        """Compute the Gaussian Process model"""
        ''''Compute the covariance between new observation and old ones'''
        k_cov = [self.kernel(np.asarray(self.Hist[-1]), np.asarray(z)) for z in self.Hist]
        '''Add these data to the covariance table'''
        for (c1, c2) in zip(self.Cov, k_cov):
            c1.append(c2)
        self.Cov.append(k_cov)
        '''Compute the inverse of the covariance'''
        T = len(self.Cov)
        matCov = np.asarray(self.Cov)
        invCov = np.linalg.inv(matCov+self.sigma*self.sigma*np.identity(T))
        '''Update the inverse H matrix'''
        for i in range(T-1):
            self.invH[i].append(RLBase.gamma**(T-i-1))
        self.invH.append([0]*(T-1)+[1])
        matInvH = np.asarray(self.invH)
        '''Compute alpha and C matrix'''
        self.Alpha = np.dot(np.dot(invCov, matInvH), self.rHist)
        self.C = invCov
        return

    def gpPredict(self, z):
        """Use the Gaussian Process model to predict Q(z=<s,a>)"""
        '''Compute the mean value and variance'''
        k_cov = [self.kernel(np.asarray(z), np.asarray(el)) for el in self.Hist]
        vec_k_cov = np.asarray(k_cov)
        k0 = self.kernel(np.asarray(z), np.asarray(z))
        meanVal = vec_k_cov.dot(self.Alpha)
        varVal = k0 - vec_k_cov.dot(np.dot(self.C,vec_k_cov))
        '''Generate a sample from the prediction'''
        return np.random.normal(meanVal, np.sqrt(varVal))


# Gaussian Process SARSA
class EpGpSarsa(RLBase):
    def __init__(self):
        # The noisy level of the Gaussian process
        self.sigma = 0.2
        self.kernel_sigma = np.array([2,0.1,0.05,0.05])
        self.explore_prob = 0.2
        # Observation history
        self.Hist = []  # <State, Action, Action>
        self.R = []  # Reward
        # The covariance matrix
        self.Cov = []
        # Current State
        self.z = None
        # H matrix
        self.H = []
        # Gaussian Process Parameters
        self.A = None
        self.C = None
        # Add r flag
        r_flag = False


    def kernel(self, z1: np.array, z2: np.array) -> float:
        d = self.kernel_sigma * (z1-z2)
        dd = np.sum(d**2)
        return np.exp(-1.0*dd)

    def observe(self, a, r, s, start = False, terminal = False):
        # Add to the history
        if start is False:
            self.Hist.append(list(self.z) + [a])
            self.R.append(r)

        # Update Current State
        if terminal is False:
            self.z = list(s.copy()) + [a]
        else:
            self.z.clear()
        # Update the H mat
        if start is True:
            for row in self.H:
                row.append(0)
        else:
            self.H.append([0]*len(self.H)+[1])
            self.gpRegression()
            if terminal is False:
                for row in self.H[0:-1]:
                    row.append(0)
                self.H[-1].append(-self.gamma)

    def gpRegression(self):
        """Compute the Gaussian Process model"""
        # Compute the covariance between new observation and old ones
        k_cov = [self.kernel(np.asarray(self.Hist[-1]), np.asarray(z)) for z in self.Hist]
        # Add these data to the covariance table
        for (c1, c2) in zip(self.Cov, k_cov):
            c1.append(c2)
        self.Cov.append(k_cov)
        # Compute the inverse of the covariance
        T = len(self.H)
        K = np.asmatrix(self.Cov)
        '''Compute alpha and C matrix'''
        HH = np.asmatrix(self.H)
        Temp = HH.T * np.linalg.inv(HH * K * HH.T + HH * HH.T * self.sigma * self.sigma)
        r = np.asmatrix(self.R)
        self.A = Temp * r.T
        self.C = Temp * HH

    def decide(self, start = False):
        """We use Thompson sampling to decide the next-step action."""
        # At the first step, choose the lowest price
        th = np.random.rand()
        if th<self.explore_prob:
            return np.random.choice(RLBase.ActionSet)
        else:
            if len(self.H) == 0:
                return RLBase.ActionSet[0]
            if start is True:
                self.z = [0.5, 0.0, RLBase.ActionSet[0]]
            # Calculate the prediction
            q = np.zeros(len(RLBase.ActionSet))
            x = self.z.copy() + [0]
            for i in range(len(RLBase.ActionSet)):
                x[-1] = RLBase.ActionSet[i]
                q[i] = self.gpPredict(x)
            print("Q: ", q)
            pos = np.argmax(q)
            return RLBase.ActionSet[pos]

    def gpPredict(self, z):
        """Use the Gaussian Process model to predict Q(z=<s,a>)"""
        # Compute the mean value and variance
        k_cov = [self.kernel(np.asarray(z), np.asarray(el)) for el in self.Hist]
        vec_k_cov = np.asmatrix(k_cov)
        k0 = self.kernel(np.asarray(z), np.asarray(z))
        meanVal = vec_k_cov * self.A
        varVal = k0 - vec_k_cov * self.C * vec_k_cov.T
        # print(meanVal, '\t', varVal)
        # Generate a sample from the prediction
        # return np.random.normal(meanVal, np.sqrt(varVal))
        return meanVal


# True Online SARSA
class TOSarsa(RLBase):

    # SARSA parameters
    explore_prob = 0.2
    p_alpha = 1.0
    p_lambda = 0.8

    def __init__(self, T: int):
        # The kernel parameter
        self.kernel_sigma = np.array([10, 0.5, 0.5, 0.5])
        # The state space basis
        state_basis = np.arange(0.5, 1.05, 0.05)
        # The space basis
        basis_dim1 = state_basis.shape[0]
        basis_dim2 = T+1
        basis_dim3 = len(self.ActionSet)
        basis_dim4 = len(self.ActionSet)
        bsize = basis_dim1*basis_dim2*basis_dim3*basis_dim4
        space_basis_temp = np.zeros(shape=(basis_dim1, basis_dim2, basis_dim3, basis_dim4, 4))
        for i in range(basis_dim1):
            for j in range(basis_dim2):
                for k in range(basis_dim3):
                    for t in range(basis_dim4):
                        space_basis_temp[i, j, k, t] = [state_basis[i], j, self.ActionSet[k], self.ActionSet[t]]
        self.space_basis = space_basis_temp.reshape((bsize,4))

        # The system state < S, a >
        self.z = np.zeros(4)
        self.z_p = np.zeros(4)
        # The system feature vector
        self.phi = None
        self.phi_p = None
        # The action reward
        self.r = 0
        # The theta variable
        self.theta = np.zeros(bsize)
        # The Q_old
        self.Q_old = 0.0
        # The e vector
        self.e = np.zeros(bsize)
        # The end point
        self.end_point = T

    def feature(self, z1: np.array):
        d = np.multiply(self.kernel_sigma, z1 - self.space_basis)
        dd = np.sum(np.square(d), axis=1)
        f = np.exp(-1.0*dd)
        return f/np.sum(f)

    def decide(self, start = False):
        """Decide the action a_t"""
        if start == True:
            # Initialize the state vector
            self.z = np.array([0.5, 0, self.ActionSet[0], 0])
            # Decide the action
            a = self.action_strategy(self.z)
            self.z[-1] = a
            # Update the algorithm parameters
            self.phi = self.feature(self.z)
            self.e.fill(0.0)
            self.Q_old = 0.0
            return a
        else:
            # Decide the action
            a = self.action_strategy(self.z_p)
            self.z_p[-1] = a
            # Update the phi_p
            if self.z_p[1] < self.end_point:
                self.phi_p = self.feature(self.z_p)
            else:
                self.phi_p = np.zeros(self.phi.shape[0])
            # Update the Q
            Q = np.dot(self.theta, self.phi)
            Q_p = np.dot(self.theta, self.phi_p)
            delta = self.r + self.gamma*Q_p - Q
            self.e = self.p_lambda*self.gamma*self.e + self.phi - self.p_alpha*self.gamma*self.p_lambda*np.dot(self.e, self.phi)*self.phi
            self.theta += self.p_alpha*(delta+Q-self.Q_old)*self.e-self.p_alpha*(Q-self.Q_old)*self.phi
            # print(delta, '\t', np.sum(self.e), '\t', np.sum(self.theta), '\t', Q_p)
            self.Q_old = Q_p
            self.phi = self.phi_p
            self.z = self.z_p
            return a

    def observe(self, a, r, s, start = False, terminal = False):
        """Observe <reward_t, state_t+1> """
        self.z_p[0:2] = s
        self.z_p[2] = a
        self.r = r
        return

    def action_strategy(self, z):
        # Decide the action with epsilon-greedy strategy
        th = np.random.rand()
        if th<self.explore_prob:
            print("Exploring")
            return np.random.choice(RLBase.ActionSet)
        else:
            Q = np.zeros(len(self.ActionSet))
            for i, a in enumerate(self.ActionSet):
                z[-1] = a
                phi = self.feature(z)
                Q[i] = np.dot(self.theta, phi)
            pos = np.argmax(Q)
            print(Q)
            return RLBase.ActionSet[pos]


# Sparse Gaussian Process SARSA
class EpSGPS(RLBase):
    def __init__(self, end:int):
        # The noisy level of the Gaussian process
        self.sigma = 1.0
        self.kernel_sigma = np.array([2,0.1,0.1,0.1])
        self.explore_prob = 0.2
        self.sparse_th = 0.05
        self.endT = end
        # Observation history
        self.D = []  # <State, Action, Action>
        # The inverse covariance matrix
        self.invK = np.asmatrix(np.zeros(1))
        # The mean and variance matrix
        self.mu = np.asmatrix(np.zeros((1, 1)))
        self.CV = np.asmatrix(np.zeros((1, 1)))
        self.c = np.asmatrix(np.zeros((1, 1)))
        self.g = np.asmatrix(np.zeros((1, 1)))
        self.c_p = None
        self.g_p = None
        # The algorithm parameters
        self.d = 0
        self.re_v = 0
        # Current belief
        self.z = None

    def kernel(self, z1: np.array, z2: np.array) -> float:
        d = self.kernel_sigma * (z1-z2)
        dd = np.sum(d**2)
        return np.exp(-1.0*dd)

    def decide(self, start = False):
        # The beginning of each epi
        if start is True:
            self.z = [0.5, 0, self.ActionSet[0], 0]
            # The starting point
            if len(self.D) == 0:
                self.z[-1] = np.random.choice(self.ActionSet)
                self.D.append(self.z.copy())
                self.invK[0, 0] = self.kernel(np.asarray(self.D[0]), np.asarray(self.D[0]))
            else:
                self.z[-1] = self.strategy(self.z)
            # Initialize the parameters
            self.c = np.asmatrix(np.zeros((len(self.D),1)))
            self.d = 0
            self.re_v = 0
            k = self.var_vector(self.z)
            self.g = self.invK * k
            delta = float(self.kernel(np.asarray(self.z), np.asarray(self.z)) - k.T * self.g)
            if delta > self.sparse_th:
                self.add_one(delta, start=True)
        return self.z[-1]

    def strategy(self, z):
        if self.explore_prob < 0.0:
            q = np.zeros(len(self.ActionSet))
            for i, a in enumerate(self.ActionSet):
                z[-1] = a
                q[i] = self.gp_predict_Thompson(z)
            print("Q: ", q)
            pos = np.argmax(q)
            return RLBase.ActionSet[pos]
        else:
            th = np.random.rand()
            if th < self.explore_prob:
                q = np.zeros(len(self.ActionSet))
                for i, a in enumerate(self.ActionSet):
                    z[-1] = a
                    q[i] = self.gp_predict(z)
                print("Exploring Q: ", q)
                return np.random.choice(RLBase.ActionSet)
            else:
                q = np.zeros(len(self.ActionSet))
                for i, a in enumerate(self.ActionSet):
                    z[-1] = a
                    q[i] = self.gp_predict(z)
                print("Q: ", q)
                pos = np.argmax(q)
                return RLBase.ActionSet[pos]

    def gp_predict(self, x):
        """Use the Gaussian Process model to predict Q(z=<s,a>)"""
        # Compute the mean value and variance
        k_cov = self.var_vector(x)
        mean_val = self.mu.T * k_cov
        k0 = self.kernel(np.asarray(x), np.asarray(x))
        varVal = k0 - k_cov.T * self.CV * k_cov
        print(varVal)
        # print(meanVal, '\t', varVal)
        # Generate a sample from the prediction
        # return np.random.normal(meanVal, np.sqrt(varVal))
        return mean_val

    def gp_predict_Thompson(self, x):
        """Use the Gaussian Process model to predict Q(z=<s,a>)"""
        # Compute the mean value and variance
        k_cov = self.var_vector(x)
        mean_val = self.mu.T * k_cov
        k0 = self.kernel(np.asarray(x), np.asarray(x))
        varVal = k0 - k_cov.T * self.CV * k_cov
        print(varVal)
        # print(meanVal, '\t', varVal)
        # Generate a sample from the prediction
        return np.random.normal(mean_val, np.sqrt(varVal))

    def var_vector(self, x):
        k_cov = [self.kernel(np.asarray(x), np.asarray(el)) for el in self.D]
        return np.asmatrix(k_cov).T

    def add_one(self, delta, start=False, k=None, k_p=None, delta_k=None):
        self.D.append(self.z.copy())
        if start is True:
            self.invK += 1.0 / delta * self.g * self.g.T
            temp1 = -1.0 / delta * self.g
            self.invK = np.hstack((self.invK, temp1))
            temp2 = np.hstack((temp1.T, [[1.0 / delta]]))
            self.invK = np.vstack((self.invK, temp2))
            temp = np.zeros(len(self.D))
            temp[-1] = 1
            self.g = np.asmatrix(temp).T
            self.c = np.vstack((self.c, [0]))
        else:
            self.invK += 1.0 / delta * self.g_p * self.g_p.T
            temp1 = -1.0 / delta * self.g_p
            self.invK = np.hstack((self.invK, temp1))
            temp2 = np.hstack((temp1.T, [[1.0 / delta]]))
            self.invK = np.vstack((self.invK, temp2))
            temp = np.zeros(len(self.D))
            temp[-1] = 1
            self.g_p = np.asmatrix(temp).T
            h = np.vstack((self.g, [-self.gamma]))
            delta_ktt = self.g.T * (k - 2*self.gamma*k_p) + self.gamma * self.gamma * self.kernel(np.asarray(self.z), np.asarray(self.z))
            temp = self.CV * delta_k
            self.c_p = self.gamma * self.sigma * self.sigma * self.re_v * np.vstack((self.c, [0])) + h - np.vstack((temp, [0]))
            self.re_v = float(1.0 / ((1+self.gamma*self.gamma)*self.sigma*self.sigma + delta_ktt - delta_k.T * self.CV * delta_k + 2 * self.gamma * self.sigma * self.sigma*self.re_v * self.c.T * delta_k - (self.gamma**2) * (self.sigma**4) * self.re_v))
        self.mu = np.vstack((self.mu, [0]))
        self.CV = np.hstack((self.CV, np.zeros((self.CV.shape[0], 1))))
        self.CV = np.vstack((self.CV, np.zeros((1, self.CV.shape[1]))))

    def observe(self, a, r, s, start=False, terminal=False):
        # Update the observation
        delta = 0
        k = self.var_vector(self.z)
        k_p = None
        delta_k = None
        if terminal is False:
            self.z[0:2] = s
            self.z[2] = a
            self.z[-1] = self.strategy(self.z)
            k_p = self.var_vector(self.z)
            self.g_p = self.invK * k_p
            delta = float(self.kernel(np.asarray(self.z), np.asarray(self.z)) - k_p.T * self.g_p)
            delta_k = k - self.gamma * k_p
        else:
            self.g_p = np.asmatrix(np.zeros((len(self.D), 1)))
            delta_k = k
        # Update the r vector
        self.d = self.gamma * self.sigma * self.sigma * self.d * self.re_v + r - delta_k.T * self.mu
        # Update the basis
        if delta > self.sparse_th and terminal is False:
            self.add_one(delta, start=False, k=k, k_p=k_p, delta_k=delta_k)
        else:
            h = self.g - self.gamma * self.g_p
            self.c_p = self.gamma * self.sigma * self.sigma * self.re_v * self.c + h - self.CV * delta_k
            if terminal is False:
                self.re_v = float(1.0 / ((1+self.gamma*self.gamma)*self.sigma*self.sigma + delta_k.T * (self.c_p + self.gamma * self.sigma * self.sigma * self.re_v * self.c) - (self.gamma**2) * (self.sigma**4) * self.re_v))
            else:
                self.re_v = float(1.0 / (self.sigma*self.sigma + delta_k.T * (self.c_p + self.gamma * self.sigma * self.sigma * self.re_v * self.c) - (self.gamma**2) * (self.sigma**4) * self.re_v))
        # Update the mean and variance
        self.mu += self.c_p * self.d * self.re_v
        self.CV += self.re_v * self.c_p * self.c_p.T
        self.c = self.c_p
        self.g = self.g_p
