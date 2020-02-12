import numpy as np
from scipy.stats import norm

try:
    import cupy as cp
    cupy_flag = True
except:
    import numpy as cp
    cupy_flag = False
    print('Could not import cupy. Running numpy instead.')

# import warnings

class additive_poisson_process:
    """
    
    """
    def __init__(self, samples, dt, t_max, order=1, kernel_estimator=True, h=1, device='gpu'):
        """
        Parameters:
        -----------
        X:numpy.ndarray
            1D array of the observations.
        T: numpy.ndarray
            1D array of the time.
        dt: int
            Descerete time step of the model.
        t_max: int
            Max time in the model
        h: float
            Bandwidth for Kernel Density Estimation (KDE)
        
        Returns:
        --------
        None
        """

        if len(samples.shape) < 2:
            samples = samples[:,np.newaxis]

        # Store input values
        self.dt = dt
        self.N = int(t_max / dt)
        self.samples = samples
        self.order = order
        self.device = device
        if not cupy_flag:
            self.device = 'cpu'

        # Preprocess data
        self._get_nodes()
        self.bin_times = (np.arange(1,self.N+1) + np.arange(self.N)) * self.dt / 2
        self.p_emp_vec = self._KDE(self.samples, self.dt, h, kernel_estimator=kernel_estimator)

        # Initalize model
        self._store_index()
        self.eta_emp_vec = self._compute_eta(self.p_emp_vec)

        self.theta_vec = np.zeros((self.N, self.C_vec.shape[0]))
        self.psi = 0

    def _count_observations_in_bin(self, samples, dt, C_vec):
        # Store half bin time interval
        dt2 = dt/2
        C_event_count_list = list()
        C_time_list = list()
        for C in C_vec:
            event_count_list = list()
            time_list = list()
            for b in self.bin_times:
                # Count number of observation in each bin
                count = np.sum(np.prod((samples[:,C] >= (b - dt2)) & (samples[:,C] < (b + dt2)), axis=1))
                if count != 0:
                    event_count_list.append(count)
                    time_list.append(b)
            C_event_count_list.append(event_count_list)
            C_time_list.append(time_list)
        return C_event_count_list, C_time_list


    def _KDE(self, samples, dt, h, kernel_estimator=True):


        self.C_event_count_list, self.C_time_list = self._count_observations_in_bin(samples, dt, self.C_vec)
        self.count = np.array([np.sum(event) for event in self.C_event_count_list])

        if kernel_estimator:
            ft = np.array([
                np.sum([norm.pdf(self.bin_times, t, h) * c
                        for c, t in zip(event_count_list, time_list)], axis=0)
                        for event_count_list, time_list in zip(self.C_event_count_list, self.C_time_list)
            ])
        else:
            def delta_bin_times(t):
                bin_times = np.zeros(len(self.bin_times))
                bin_times[np.argmax(self.bin_times > t)] = 1
                return bin_times
            ft = np.array([
                np.sum([(delta_bin_times(t)) * c
                        for c, t in zip(event_count_list, time_list)], axis=0)
                        for event_count_list, time_list in zip(self.C_event_count_list, self.C_time_list)
            ])
        del_count = 0
        for i, x in enumerate(ft):
            if ((x == 0) == True).all():
                self.C_vec = np.delete(self.C_vec, i - del_count, axis=0)
                ft = np.delete(ft, i - del_count, axis=0)
                del_count += 1
        ft = np.array([f for f in ft]).T
        ft /= np.sum(ft)
        return ft

    def _activate_nodes(self, C, ci, N, depth):
        """
        A recursive function to generate only the nodes which
        are required.
        Parameters:
        -----------
        None
        Returns:
        --------
        None
        """
        if depth == 0:
            self.C_vec.append(C)
            return

        for c in range(ci, N-depth+1):
            C_current = np.array(C, dtype=bool)
            C_current[c] = True
            C_current = self._activate_nodes(C_current, c+1, N, depth-1)

    def _get_nodes(self):
        """
        Initalise nodes for the model.
        Parameters:
        -----------
        None
        Returns:
        --------
        None
        """
        N = self.samples.shape[1]
        self.C_vec = list()
        for d in np.arange(1,self.order+1):
            self._activate_nodes(np.zeros(N, dtype=bool), 0, N, d)
        self.C_vec = np.array(self.C_vec, dtype=bool)

    def _store_index(self):
        """

        """
        self.eta_idx = np.array([[
            np.sum(self.C_vec & C, axis=1) == np.sum(C)
            for C in self.C_vec]
            for n in range(self.N)
        ])

        self.theta_sum_idx = np.array([[
            np.sum(self.C_vec & C, axis=1) == np.sum(self.C_vec, axis=1)
            for C in self.C_vec]
            for n in range(self.N)
        ])

        # self.fim_idx = _fim_idx_fn(self.eta_idx)
        self.fim_idx = np.array([[
            self.eta_idx[n1,i,:] & self.eta_idx[n2,j,:]
                        for n1 in range(self.N) for i in range(self.C_vec.shape[0])]
                        for n2 in range(self.N) for j in range(self.C_vec.shape[0])
        ])


        self.fim_n_val = np.array([[
            n1 if n1 > n2 else n2
                        for n1 in range(self.N) for i in range(self.C_vec.shape[0])]
                        for n2 in range(self.N) for j in range(self.C_vec.shape[0])
        ])

    def _compute_theta_sum(self, theta_sum_idx):
        """
        Parameters:
        -----------
        None
        
        Returns:
        --------
        theta_sum_vec: numpy.ndarray
            An array returning sum of theta.
        """
        theta_sum_vec = np.array([[
            np.sum(self.theta_vec[:n+1, theta_sum_idx[n,i,:]])
            for i in range(theta_sum_idx.shape[1])]
            for n in range(self.N)
        ])
        return theta_sum_vec

    def compute_partition_fuction(self, p_vec):
        """
        Computes partition function
        
        Parameters:
        -----------
        None
        
        Returns:
        --------
        None
        """
        self.psi += np.log(np.sum(p_vec))


    def _reconstruct(self, theta_sum_idx):
        """
        Computes the probability of each node in the model
        
        Parameters:
        -----------
        None
        
        Returns:
        --------
        None
        """
        p_vec = np.exp(self._compute_theta_sum(theta_sum_idx) - self.psi)
        self.compute_partition_fuction(p_vec)
        p_vec = np.exp(self._compute_theta_sum(theta_sum_idx) - self.psi)

        return p_vec


    def _compute_eta(self, p_vec):
        """
        Computes the expectation parameter (eta) for each node.
        
        Parameters:
        -----------
        None
        
        Returns:
        ---------
        None
        """
        eta_vec = np.array([[
            np.sum(p_vec[n:, self.eta_idx[n,i,:]])
            for i in range(self.eta_idx.shape[1])]
            for n in range(self.N)
        ])
        return eta_vec
    
    def _construct_fisher_information_matrix(self):
        """

        """
        fim = np.array([[
            np.sum(self.p_vec[self.fim_n_val[i,j]:, self.fim_idx[i,j,:]])
                        for i in range(self.fim_n_val.shape[1])]
                        for j in range(self.fim_n_val.shape[0])
        ]) - np.outer(self.eta_vec, self.eta_vec)
        return fim

    def _step(self, natural_gradient=False, lr=1, stochastic_eta=False, noise_epsilon=1e-10):
        """
        A single step for each iteration

        Parameters:
        ------------
        natural_gradient: bool
            True: Uses natural gradient (not working yet)
            False: Uses gradient descent
        lr: int
            Learning rate for gradient descent
        
        Returns:
        ---------
        error: float
            Root mean square error
        """
        self.p_vec = self._reconstruct(self.theta_sum_idx)
        self.eta_vec = self._compute_eta(self.p_vec)
        # self.eta_emp = self._stochastic_eta() if stochastic_eta else self.eta_emp_vec
        if self.device == 'cpu':
            gradient = self.eta_emp_vec - self.eta_vec
        elif self.device == 'gpu':
            gradient = cp.array(self.eta_emp_vec) - cp.array(self.eta_vec)
        if natural_gradient:
            if self.device == 'cpu':
                fim = self._construct_fisher_information_matrix()
                fim += np.eye(len(fim)) * noise_epsilon
            elif self.device == 'gpu':
                fim = cp.array(self._construct_fisher_information_matrix())
                fim += cp.eye(len(fim)) * noise_epsilon
            try:
                if self.device == 'cpu':
                    self.theta_vec += np.linalg.solve(fim, gradient.flatten()).reshape(self.eta_emp_vec.shape)
                elif self.device == 'gpu':
                    self.theta_vec +=  cp.asnumpy(cp.linalg.solve(fim, gradient.flatten()).reshape(self.eta_emp_vec.shape))
            except:
                print('Could not invert fisher information matrix, running gradient descent instead')
                self.natural_gradient = False
                self.theta_vec += lr * gradient

        else:
            self.theta_vec += lr * gradient
        self.p_vec = self._reconstruct(self.theta_sum_idx)
        
        error = np.mean((gradient)**2)**0.5
        return error

    def train(self, N_iter, natural_gradient=False, lr=1, tol=1e-7, stochastic_eta=False, noise_epsilon=1e-10, verbose=False, verbose_step=1000):
        """
        Training function for the model.

        Parameters:
        ------------
        N_iter: int
            Maximum number of iterations
        natural_gradient: bool
            True: Uses natural gradient (not working yet)
            False: Uses gradient descent
        lr: int
            Learning rate for gradient descent
        verbose: bool
            Print debugging statements
        verbose_step: int
            Number of steps before printing debug statements
        """
        self.natural_gradient = natural_gradient
        for i in range(N_iter):
            error = self._step(natural_gradient=self.natural_gradient, lr=lr, stochastic_eta=stochastic_eta, noise_epsilon=noise_epsilon)

            if verbose and (i % verbose_step == 0):
                print('=========================================')
                print('iteration:', i)
                # print('p_vec:\n', self.p_vec)
                # print('np.sum(p_vec):', np.sum(self.p_vec))
                # print('eta:\n', self.eta_vec)
                # print('eta_emp:\n', self.eta_emp_vec)
                # print('theta_vec:\n', self.theta_vec)
                # print('gradient:\n', self.eta_emp_vec - self._compute_eta(self.p_vec))
                # print('theta_sum:\n', self._compute_theta_sum(self.theta_sum_idx))
                # print('psi:', self.psi)
                print('error:', error)
                print('=========================================')

            if error < tol:
                break
        if i >= N_iter:
            print('Maximum iteration reached. Did not converge.')

    def get_count(self, C_vec=None):
        C_vec = np.atleast_2d(C_vec)
        C_event_count_list, C_time_list = self._count_observations_in_bin(self.samples, self.dt, C_vec)
        count = np.array([np.sum(event) for event in C_event_count_list])
        return count

    def get_intensity(self, C_vec=None):
        """
        Get intensity of the Poisson Point Process

        Parameters:
        ------------
        None

        Returns:
        --------
        intensity: np.ndarray
            Intensity of the Poisson Point Process
        """
        if C_vec is None:
            C_vec = self.C_vec

        C_vec = np.atleast_2d(C_vec)

        theta_sum_idx = np.array([[
            np.sum(self.C_vec & C, axis=1) == np.sum(self.C_vec, axis=1)
            for C in C_vec]
            for n in range(self.N)
        ])


        C_vec = np.atleast_2d(C_vec)
        theta_sum = self._compute_theta_sum(theta_sum_idx)
        count = self.get_count(C_vec)
        count[count == 0] = 1
        intensity = np.array([
            np.squeeze((np.exp(theta_sum[:,i]) / np.sum(np.exp(theta_sum[:,i]), axis=0)) * count[i])
            for i, C in enumerate(C_vec)
        ]).T
        return intensity

    def _get_event_time(self, C):
        idx = np.where(np.sum(self.C_vec == np.array(C), axis=1) == self.C_vec.shape[1])[0]
        if len(idx) != 0:
            event_time = self.C_time_list[idx[0]]
        else:
            event_time = np.array([])
        return event_time

    def get_event_time(self, C_vec):
        C_vec = np.atleast_2d(C_vec)
        event_times = [self._get_event_time(C) for C in C_vec]
        return event_times

def convert_spiketimes_to_discreteincrements(samples, t_init, t_max, dt):
    N_intervals = int((t_max - t_init) / dt) + 1
    t_intervals = np.linspace(t_init, t_max, N_intervals)
    X = np.array([
        np.sum((samples >= t_intervals[n]) & (samples < t_intervals[n+1]), axis=0, dtype=bool)
            for n in range(N_intervals-1)
    ])
    return X, t_intervals[:-1]
    
if __name__ == '__main__':
    np.random.seed(1)
    X = np.random.binomial(1,0.3,100)
    T = np.array(range(len(X)))
    IGPPP_model = IGPPP(X,T,1,100, 5)
    IGPPP_model.train(100, verbose=False, verbose_step=10)
    plt.plot(IGPPP_model.get_intensity())
    plt.plot(T[X==True], np.zeros(len(T[X==True])), 'o')
    plt.show()