import numpy as np
from numba import jit, njit
from scipy.optimize import minimize
from psmc.utils import maxmul

from tqdm import tqdm

@njit
def rand_choice_nb(arr, prob):
    """
    Returns a random choice from an array based on given probabilities.

    Parameters:
    -----------
    arr: numpy.ndarray
        The array from which to make a random choice.
    prob: numpy.ndarray
        An array of probabilities for each element in `arr`. The probabilities
        should sum to 1.

    Returns:
    --------
    numpy.ndarray
        A single element randomly chosen from `arr` based on the given probabilities.

    """
    cum_prob = np.cumsum(prob)
    return arr[np.searchsorted(cum_prob, np.random.random(), side="right")]

class PSMC:
    def __init__(self, t_max, n_steps, theta0, rho0, mu=2.5 * 1e-8, pattern=None):
        
        self.n_steps = n_steps
        self.mu = mu
        self.pattern=pattern

        self.theta = theta0 #learned parameter
        self.rho = rho0 #learned parameter
        self.lam = np.ones(self.n_free_params) #learned parameter
        self.t_max = t_max #learned parameter, apparently
        
        self.t = self.compute_t()
        self.C_pi, self.C_sigma, self.p_kl, self.em, self.sigma = self.compute_params()

        # store params of expectation step
        self.loglike_stored = None
        self.log_xi_stored = None
        self.log_gamma_stored = None                
        
    @property
    def N0(self):
        return self.theta / (4 * self.mu)
    
    @property
    def n_free_params(self):
        if self.pattern==None:
            return 3 + self.n_steps
        else:
            return 3 + np.sum([int(x.split('*')[0]) for x in self.pattern.split('+')])
    
    def compute_t(self, alpha=0.1):
        """
        Computes the time points for a PSMC model. For details look at the paper methods section.

        Parameters:
        -----------
        alpha : float, optional
            A scaling factor for the time intervals.

        Returns:
        --------
        numpy.ndarray
            An array of time points.
        """
        beta = np.log(1 + self.t_max / alpha) / self.n_steps
        t = []
        for k in range(self.n_steps):
            t.append(alpha * (np.exp(beta * k) - 1))
        t.append(self.t_max)
        t.append(1e300)
        return np.array(t)
    
    def map_lam(self, lam_grouped):
        pattern = self.pattern
        if pattern != None:
            lam = []
            counter = 0
            pattern = np.array([x.split('*') for x in pattern.split('+')]).astype(int)
            for s in range(pattern.shape[0]):
                ts, gs = pattern[s]
                for t in range(ts):
                    for g in range(gs):
                        lam.append(lam_grouped[counter])
                    counter += 1
            lam.append(lam_grouped[-1])
            return np.array(lam)
        else:
            return lam_grouped
            
    # PSMC model functions 
        
    def compute_params(self):
        """
        Computes parameters for the PSMC model. For details look at the paper methods section.
        Largely directly translated from the original C code (for the sace of efficiency).

        Returns:
        --------
        tuple
            A tuple of computed parameters, including:
            - C_pi: a scalar value used in calculation of other parameters
            - C_sigma: a scalar value used in calculation of other parameters
            - p_kl: a matrix of transition probabilities between states k and l
            - e: emission probabilities for each state
            - sigma: prior probabilities for each state

        """
        n, t, lam, theta, rho = self.n_steps, self.t, self.lam, self.theta, self.rho
        lam = self.map_lam(lam)
        
        # Initialize arrays
        alpha = np.zeros(n+2)
        tau = np.zeros(n+1)
        beta = np.zeros(n+1)
        sigma = np.zeros(n+1)
        q_aux = np.zeros(n)
        q = np.zeros(n + 1) * np.nan
        e = np.zeros((2, n+1))
        p_kl = np.zeros((n+1, n+1))

        # Calculate theta
        for k in range(n+1):
            tau[k] = t[k+1] - t[k]

        # Calculate alpha
        alpha[0] = 1.
        for k in range(1, n+1):
            alpha[k] = alpha[k-1] * np.exp(- tau[k-1] / lam[k-1])
        alpha[k+1] = 0.

        # Calculate beta
        beta[0] = 0.
        for k in range(1, n+1):
            beta[k] = beta[k-1] + lam[k-1] * (1.0 / alpha[k] - 1.0 / alpha[k-1])

        # Calculate C_pi and C_sigma
        C_pi = 0.
        for i in range(n+1):
            C_pi += lam[i] * (alpha[i] - alpha[i + 1])

        C_sigma = 1./ (C_pi * rho) + 0.5

        for m in range(n):
            q_aux[m] = (alpha[m] - alpha[m+1]) * (beta[m] - lam[m] / alpha[m]) + tau[m]

        sum_t = 0. 
        # Just in case initialize them as 0
        k = 0 
        new_m = 0 
        m = 0 

        # Loop for calculating sigma, q and e
        for k in range(n + 1):
            ak1 = alpha[k] - alpha[k+1]
            lak = lam[k]

            # Calculate pi_k, sigma_k
            cpik = ak1 * (sum_t + lak) - alpha[k+1] * tau[k]
            pik = cpik / C_pi
            sigma[k] = (ak1 / (C_pi * rho) + pik / 2.0) / C_sigma

            # Calculate avg_t, the average time point where mutation happens
            avg_t = - np.log(1.0 - pik / (C_sigma * sigma[k])) / rho
            if np.isnan(avg_t) or avg_t < sum_t or avg_t > sum_t + tau[k]:  # in case something bad happens
                print("SOMETHING BAD IS HAPPENING")
                avg_t = sum_t + (lak - tau[k] * alpha[k+1] / (alpha[k] - alpha[k+1]))

            # Calculate q_{kl}
            tmp = ak1 / cpik

            for m in range(k):
                q[m] = tmp * q_aux[m]  # l < k
            q[k] = (ak1**2 * (beta[k] - lak/alpha[k]) + 2*lak*ak1 - 2*alpha[k+1]*tau[k]) / cpik  # k = l

            new_m = m + 1
            if k < n:
                tmp = q_aux[k] / cpik
                for m in range(new_m, n+1):
                    q[m] = (alpha[m] - alpha[m+1]) * tmp # k > l

            # Calculate p_{kl}
            tmp = pik / (C_sigma * sigma[k])
            for m in range(n+1):
                p_kl[k, m] = tmp * q[m]
            p_kl[k,k] = tmp * q[k] + (1.0 - tmp)

            # Calculate e_{2,k}
            e[0, k] = np.exp(-theta * (avg_t + 0))
            e[1, k] = 1 - np.exp(-theta * (avg_t + 0))

            sum_t += tau[k]

        return C_pi, C_sigma, p_kl, e, sigma
    
    def param_recalculate(self):
        """
        Recalculates model parameters.

        Returns:
        --------
        tuple
            A tuple of computed parameters, including:
            - C_pi: a scalar value used in calculation of other parameters
            - C_sigma: a scalar value used in calculation of other parameters
            - p_kl: a matrix of transition probabilities between states k and l
            - em: emission probabilities for each state
            - sigma: prior probabilities for each state

        """
        self.C_pi, self.C_sigma, self.p_kl, self.em, self.sigma = self.compute_params()
    
    ## General HMM functionality
    
    def prior_matrix(self):
        """array (k_states)"""
        return self.sigma
    
    def transition_matrix(self):
        """array (k_states x k_states)"""
        return self.p_kl
    
    def emission_matrix(self):
        """array (2 observed_states x k_states)"""
        return self.em
    
    def emission_likelihood(self, x):
        """array (batch x string x k_states)"""
        return self.em[x]
    
    def normalize(self, a, axis=-1):
        c = np.sum(a, axis=axis)
        b = a / c[:, None]
        
        return b, c
    
        
    def compute_alpha(self, x):
        pi = self.prior_matrix()
        A = self.transition_matrix()
        b = self.emission_likelihood(x)
        
        batch_size = x.shape[0]
        S_max = x.shape[1]
        
        alpha = np.zeros((batch_size, S_max, self.n_steps+1))
        c_norm = np.zeros((batch_size, S_max))
        
        alpha[:, 0, :], c_norm[:,0] = self.normalize(np.multiply(b[:,0,:], pi[None,:]), axis=-1)
        
        for s in tqdm(range(1, S_max)):
            alpha[:, s, :], c_norm[:,s] = self.normalize(np.multiply(b[:,s,:],
                                                                     np.dot(alpha[:, s-1, :], A)), axis=-1)
        return alpha, c_norm
    
    
    def compute_beta(self, x, c_norm):
        pi = self.prior_matrix()
        A = self.transition_matrix()
        b = self.emission_likelihood(x)
        
        batch_size = x.shape[0]
        S_max = x.shape[1]
        
        beta = np.zeros((batch_size, S_max, self.n_steps+1))
        
        beta[:, -1, :]= np.ones((batch_size, self.n_steps+1))     
        for s in tqdm(range(S_max-2, -1, -1)):
            beta[:, s, :] = np.dot(beta[:, s+1, :] * b[:,s+1,:], A.T) / c_norm[:,s+1,None]
        return beta

        
    def compute_xi_gamma(self, x):
        
        batch_size = x.shape[0]
        S_max = x.shape[1]
        
        alpha, cn = self.compute_alpha(x)
        beta = self.compute_beta(x, cn)
        
        pi = self.prior_matrix()
        A = self.transition_matrix()
        b = self.emission_likelihood(x)
        
        xi = np.zeros((self.n_steps+1, self.n_steps+1))
        for i in tqdm(range(1, S_max)): # To reduce memmory usage
            xi += np.sum(alpha[:,i-1,:,None] * b[:,i,None,:] *
                         A[None,:,:] * beta[:,i,None,:] / cn[:,i,None,None], 0)
        
        gamma = (alpha * beta)
        
        return xi, gamma, cn
    
    
    def viterbi(self, x, S):
        """
        Finds the most likely sequence of hidden states given the observed data x for each example in the batch.

        Parameters:
        -----------
        x : numpy.ndarray of shape (batch_size, T_max)
            An array of integers representing the observed data.
        S : numpy.ndarray of shape (batch_size)
            An array of integers representing the number of observed data points
            in each sample.

        Returns:
        --------
        tuple
            A tuple of 2 items:
            1. A list of the most likely hidden state sequences for each example in the batch.
            2. An array of the log probabilities of the best paths for each example in the batch.

        """

        batch_size = x.shape[0]; 
        S_max = x.shape[1]
        
        # Compute the log state priors and transition matrix
        log_state_priors = np.log(self.prior_matrix())
        log_transition_matrix = np.log(self.transition_matrix())
        
        # Initialize the delta and psi arrays
        log_delta = np.zeros((batch_size, S_max, self.n_steps))
        psi = np.zeros((batch_size, S_max, self.n_steps))
        
        # Compute the emission log-likelihoods
        emission_loglike = np.log(self.emission_likelihood(x))

        # Initialize delta at time 0
        log_delta[:, 0, :] = emission_loglike[:,0,:] + log_state_priors[None,None,:]
        
        # Run the Viterbi algorithm
        for s in tqdm(range(1, S_max)):
            max_val, argmax_val = maxmul(log_delta[:, s-1, :], log_transition_matrix)
            log_delta[:, s, :] = emission_loglike[:,s,:] + max_val
            psi[:, s, :] = argmax_val

        # Get the log probability of the best path
        log_max = log_delta.max(axis=2)
        best_path_scores = log_max[np.arange(batch_size), S - 1]

        # Compute the most likely state sequence for each example in the batch
        z_star = []
        for i in range(0, batch_size):
            z_star_i = [int(np.argmax(log_delta[i, S[i] - 1, :], axis=0).item())]
            for s in range(S[i] - 2, 0, -1):
                z_s = int(psi[i, s, z_star_i[-1]].item())
                z_star_i.append(z_s)
            z_star_i.reverse()
            z_star.append(z_star_i)
    
    def hidden_path_loglike(self, z, x):
        S_max = x.shape[1]
        
        log_state_priors = np.log(self.prior_matrix())
        log_transition_matrix = np.log(self.transition_matrix())
        
        log_delta = 0
        
        emission_loglike = np.log(self.emission_likelihood(x))
        

        log_delta = emission_loglike[:,0,z[0]] + log_state_priors[z[0]]
        for s in tqdm(range(1, S_max)):
            log_delta += emission_loglike[:,s,z[s]] + log_transition_matrix[z[s-1],z[s]]
            

        return log_delta
        
    
    ## EM Inference and sampling
    
    def Q_func(self, params, x):
        """
        Calculates the Q-function for the EM algorithm.

        Parameters:
        -----------
        params : numpy.ndarray [theta, rho, lam_0, lam_1, ... lam_n+1] 
            An array of parameters that determine the state of the model.
        x : numpy.ndarray of shape (batch_size, S_max)
            An array of integers representing the observed data.
        Returns:
        --------
        float
            The value of the Q-function evaluated at the given parameters and data.

        """
        # TODO : not yet implemented for multiple batches of different size
        
        # Update learnable parameters
        self.lam = params[3:]
        self.theta = params[0]
        self.rho = params[1]
        self.t_max = params[2]
        
        # Recalculate psmc parameters necessery to estimate new (pi, A, b)
        self.param_recalculate()
        
        batch_size = x.shape[0]; 
        S_max = x.shape[1]
        
        # Expectation step parameters
        xi = self.xi_stored
        gamma = self.gamma_stored
        
        # Calculate log probabilities of pi, A, b
        log_state_priors = np.log(self.prior_matrix())
        log_transition_matrix = np.log(self.transition_matrix())
        log_emission_matrix = np.log(self.emission_likelihood(x))

        # Calculate Q-function value
        q = (gamma[:,0,:] * log_state_priors[:]).sum() + \
        (xi * log_transition_matrix[:,:]).sum() + \
        (log_emission_matrix * gamma).sum()

        return -q
    
    def EM(self, params0, bounds, x, S, n_iter=20):
        """
        Runs the EM algorithm to estimate the parameters of the model.

        Parameters:
        -----------
        x : numpy.ndarray of shape (batch_size, S_max)
            An array of integers representing the observed data.
        S : numpy.ndarray of shape (batch_size)
            An array of integers representing the number of observed data points
            in each sample.
        verbose : bool, optional (default=False)
            If True, print iteration status messages.
        max_iter : int, optional (default=100)
            The maximum number of iterations to run the EM algorithm.
        ftol : float, optional (default=1e-5)
            The convergence threshold for the change in the Q-function value.

        Returns:
        --------
        numpy.ndarray
            An array of estimated model parameters.

        """
        
        batch_size = x.shape[0]; 
        S_max = x.shape[1]
        
        # Initialize loss list and parameters history list
        loss_list = []
        params_history = []
        # Set initial parameter values
        self.lam = params0[3:]
        self.theta = params0[0]
        self.rho = params0[1]
        self.t_max = params0[2]
        params_history.append(params0)

        # Run the EM algorithm
        for i in range(n_iter):
            print('EM iteration', i)

            # Recalculate model parameters
            self.param_recalculate()
            
            # Compute xi and gamma matrices (essentially an E-step)
            self.xi_stored, self.gamma_stored, cn = self.compute_xi_gamma(x)
            loglike_before_m = np.log(cn).sum()
            
            # Q-func maximization (M-step)
            optimized_params = minimize(self.Q_func,
                                        params0,
                                        args=(x),
                                        method='Nelder-Mead', #in original paper they use Powell method
                                        bounds=bounds,
                                        options={'maxiter':3*100,
                                                 'maxfev':3*100,
                                                 'fatol': 0.1})
            
            # Update learnable parameters
            self.lam = optimized_params['x'][3:]
            self.theta = optimized_params['x'][0]
            self.rho = optimized_params['x'][1]
            self.t_max = optimized_params['x'][2]
            
            # Recalculate model parameters
            self.param_recalculate()
            
            _, cn = self.compute_alpha(x)
            loglike_after_m = np.log(cn).sum()  

            params0 = [self.theta, self.rho, self.t_max] + list(self.lam)  
            loss_list.append((loglike_before_m, loglike_after_m))
            print(loglike_before_m, '-->', loglike_after_m, '\tΔ:',  loglike_before_m - loglike_after_m)
            params_history.append(params0)

        return loss_list, params_history
    
    
    @staticmethod
    @njit
    def jit_sampling(n_iter, pi, A, b, n_states):
            z_t = rand_choice_nb(np.arange(n_states+1), pi)

            z =[]
            x = []
            z.append(z_t)
            
            obs_choice = np.arange(2)
            
            for t in range(0, n_iter):
                p = b[:,z_t]
                x_t = rand_choice_nb(obs_choice, p) # np.random.choice(np.arange(2), p = p)
                x.append(x_t)
                sel = np.arange(n_states+1)
                psum = A[z_t,:] / A[z_t,:].sum()
                z_t = rand_choice_nb(sel, psum)
                if t < n_iter-1:
                    z.append(z_t)
            return z, x
        
    def sample(self, n_iter):
        self.param_recalculate()
        
        pi = self.prior_matrix()
        A = self.transition_matrix()
        b = self.emission_matrix()
        
        return self.jit_sampling(n_iter, pi, A, b, self.n_steps)