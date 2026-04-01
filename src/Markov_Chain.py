import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time

class Markov_reward_process:
    def __init__(self, transition_matrix: np.ndarray, expected_reward: np.ndarray, std_dev_noise: float = 0.1):
        check_rows_sum_to_one = np.allclose(transition_matrix.sum(axis=1), 1.)
        if not check_rows_sum_to_one:
            raise ValueError("The provided transition matrix is not stochastic.")
        check_columns_sum_to_one = np.allclose(transition_matrix.sum(axis=0), 1.)   
        #if not check_columns_sum_to_one:
        #    print("The provided transition matrix is not bistochastic.")

        self.transition_matrix = transition_matrix
        self.cdf_transition_matrix = np.cumsum(transition_matrix, axis=1)
        self.num_states = transition_matrix.shape[0]
        self.expected_reward = expected_reward
        self.std_dev_noise = std_dev_noise

        #self.stat_dist = np.ones(self.num_states) / self.num_states

        eigvals, eigvecs = np.linalg.eig(self.transition_matrix.T)
        stat_dist = np.real(eigvecs[:, np.isclose(eigvals, 1.)][:,0])
        self.stat_dist = stat_dist / np.sum(stat_dist)
        self.cumulative_stat_dist = np.cumsum(self.stat_dist)

        self.spectral_gap = None
        self.compute_spectral_gap()

    def sample_transition(self, current_state: np.ndarray) -> np.ndarray:
        next_state = (self.cdf_transition_matrix[current_state] >= np.random.rand(current_state.shape[0],1)).argmax(axis=1).astype(np.int64)
        #shape is (nb_copies,)
        return next_state
    
    def reward(self, state: np.ndarray) -> np.ndarray:
        reward = self.expected_reward[state] + self.std_dev_noise * np.random.randn(state.shape[0])
        #shape is (nb_copies,)
        return reward
    
    def generate_invariant_distribution(self, nb_samples) -> np.ndarray:
        #return np.random.uniform(0, self.num_states, size=batch_size).astype(int)
        samples = (self.cumulative_stat_dist[None,:] >= np.random.rand(nb_samples,1)).argmax(axis=1).astype(np.int64)
        #shape is (nb_copies,)
        return samples

    def transition_operator_on_functions(self, function_vect: np.ndarray) -> np.ndarray:
        #output = self.transition_matrix @ function_vect

        #function_vect should be of size (nb_parallel_copies, nb_copies_to_average, num_states)
        output = function_vect @ self.transition_matrix.T 
        #the output should be of size (nb_parallel_copies, nb_copies_to_average, num_states)
        return output
    
    def compute_spectral_gap(self):
        aux = np.diag(np.sqrt(self.stat_dist)) @ self.transition_matrix @ np.diag(1./np.sqrt(self.stat_dist))
        eigvals = np.linalg.eigvals(aux)
        eigvals = np.sort(np.abs(eigvals))
        print(f"Eigenvalues of the transition matrix: {eigvals}")
        self.spectral_gap = 1. - eigvals[-2]
        print(f"Spectral gap of the Markov chain: {self.spectral_gap:.4f}")

class Linear_Parametrization:
    def __init__(self,num_states: int,num_features: int, initial_theta, nb_parallel_copies: int,
                 nb_copies_to_average: int, do_average: bool):
        self.num_states = num_states
        self.num_features = num_features
        self.nb_parallel_copies = nb_parallel_copies
        self.nb_copies_to_average = nb_copies_to_average
        if initial_theta is None:
            self.theta = np.tile(np.random.uniform(-1., 1., size=(1,1,num_features)),(nb_parallel_copies,nb_copies_to_average,1))
        else:
            self.theta = np.tile(initial_theta.reshape(1,1,num_features),(nb_parallel_copies,nb_copies_to_average,1))
        #theta has size (nb_parallel_copies,nb_copies_to_average, num_features)
        self.do_average = do_average
        self.theta_average = np.zeros_like(self.theta)
        self.num_averages = 0.

        self.feature_matrix = None #size num_states x num_features

    def feature(self, states: np.ndarray) -> np.ndarray:
        #states admits a shape of (nb_copies_to_average)
        output = self.feature_matrix[states,:]
        #output admits a shape of (nb_copies_to_average, num_features)
        return output

    def evaluate(self, states: np.ndarray, with_average: bool = False) -> np.ndarray:
        #states admits a shape of (nb_copies_to_average)
        values = np.sum(self.feature(states).reshape(1,states.shape[0],-1) * (self.theta_average if with_average else self.theta),axis=-1)
        #shape of values and states are (nb_parallel_copies,nb_copies_to_average)
        return values

    def evaluate_all_states(self, with_average: bool = False) -> np.ndarray:
        values = (self.theta_average if with_average else self.theta) @ self.feature_matrix.T
        return values
        #the output should be of size (nb_parallel_copies,nb_copies_to_average,num_states)

    def update_theta(self, delta_theta: np.ndarray):
        self.theta += delta_theta
        if self.do_average:
            self.num_averages += 1.
            self.theta_average += (self.theta - self.theta_average) / self.num_averages
        

class Diagonal_Parametrization(Linear_Parametrization):
    def __init__(self, eigenvalues: np.ndarray, initial_theta = None, nb_parallel_copies = 1, nb_copies_to_average = 1, do_average: bool = False):
        super().__init__(num_states=eigenvalues.shape[0], num_features=eigenvalues.shape[0], 
                         initial_theta=initial_theta, nb_parallel_copies=nb_parallel_copies, nb_copies_to_average=nb_copies_to_average, do_average=do_average)
        self.eigenvalues = eigenvalues
        #self.diag = eigenvalues

        P, _, Q = np.linalg.svd(np.random.uniform(low=-1.,high=1.,size=(self.num_states,self.num_states)))
        #Q = np.linalg.qr(np.random.randn(self.num_states,self.num_states))[0]
        #Q = np.eye(self.num_states)
        #P = np.eye(self.num_states)
        self.feature_matrix = P @ np.diag(eigenvalues) @ Q.T
      
   

class TD0_Linear_Functional_Approximation:
    def __init__(self, markov_reward_proc: Markov_reward_process, parametrization: Linear_Parametrization,
                  gamma_vect: np.ndarray = np.array([0.99]), do_pctd: bool = False):
        self.markov_reward_proc = markov_reward_proc
        self.parametrization = parametrization
        self.gamma_vect = gamma_vect.reshape(-1,1,1)
        self.num_states = markov_reward_proc.num_states
        self.do_pctd = do_pctd

        self.mu = None
        self.Sigma = None
        self.Sigma_half = None
        self.H = None
        self.S = None
        self.Sigma = None
        self.Sigma_half = None
        self.H = None
        self.S = None
        self.compute_matrices()
        self.compute_solution() 

    def compute_matrices(self):
        self.mu = self.parametrization.feature_matrix.T @ self.markov_reward_proc.stat_dist
        self.Sigma = self.parametrization.feature_matrix.T @ np.diag(self.markov_reward_proc.stat_dist) @ self.parametrization.feature_matrix
        self.Sigma_half = sp.linalg.sqrtm(self.Sigma)
        #Sigma_half = np.diag(np.sqrt(self.parametrization.diag * self.markov_reward_proc.stat_dist))
        Sigma_1 = self.parametrization.feature_matrix.T @ np.diag(self.markov_reward_proc.stat_dist) @ self.markov_reward_proc.transition_matrix @ self.parametrization.feature_matrix
        self.H = (self.Sigma.reshape(1,Sigma_1.shape[0],Sigma_1.shape[1])
                - self.gamma_vect * Sigma_1.reshape(1,Sigma_1.shape[0],Sigma_1.shape[1]) )
        self.S = (self.H + self.H.transpose(0,2,1)) / 2.

        mu_mutop = self.mu.reshape(-1,1) @ self.mu.reshape(1,-1)
        self.Sigmahat = self.Sigma - mu_mutop
        self.Sigmahat_half = sp.linalg.sqrtm(self.Sigmahat)
        Sigmahat_1 = Sigma_1 - mu_mutop
        self.Hhat = (self.Sigmahat.reshape(1,Sigmahat_1.shape[0],Sigmahat_1.shape[1])
                     - self.gamma_vect * Sigmahat_1.reshape(1,Sigmahat_1.shape[0],Sigmahat_1.shape[1]) )
        self.Shat = (self.Hhat + self.Hhat.transpose(0,2,1)) / 2.

        #diff_feature_one_step = ((np.eye(self.num_states) - self.gamma * self.markov_reward_proc.transition_matrix)
        #                                   @ self.parametrization.feature_matrix )
        #sq_norm_feature_one_step = np.sum(diff_feature_one_step** 2, axis = 1)

        #self.expected_hhtop = self.parametrization.feature_matrix.T @ np.diag(sq_norm_feature_one_step * self.markov_reward_proc.stat_dist) @ self.parametrization.feature_matrix


        #sqrt_S = sp.linalg.sqrtm(self.S)
        #inv_sqrt_S = np.linalg.inv(sqrt_S)
        #self.coef_bound_Sigma_by_S = np.linalg.norm(inv_sqrt_S @ self.Sigma @ inv_sqrt_S,ord=2)


    #def compute_threshold_alpha(self) -> float:
    #    #self.compute_matrices()
    #    sqrt_S = sp.linalg.sqrtm(self.S)
    #    inv_sqrt_S = np.linalg.inv(sqrt_S)
    #    M = inv_sqrt_S @ self.expected_hhtop @ inv_sqrt_S
    #    norm_M = np.linalg.norm(M, ord=2)
    #    return 0.5 / max(norm_M,2.)

    def compute_solution(self):
        #expected_reward should be of size (num_states,)
        #stat_dist should be of size (num_states,)
        #feature_matrix should be of size (num_states, num_features)
        right_hand_side = (self.markov_reward_proc.expected_reward * self.markov_reward_proc.stat_dist) @ self.parametrization.feature_matrix
        self.theta_star = np.zeros((self.gamma_vect.shape[0],1, self.parametrization.num_features))
        for i in range(self.gamma_vect.shape[0]):
            self.theta_star[i,0] = np.linalg.solve(self.H[i], right_hand_side)
        #theta_star should be of size (nb_parallel_copies, num_features)
        self.values_at_solution = (self.theta_star @ self.parametrization.feature_matrix.T)
        #values_at_solution should be of size (nb_parallel_copies, num_states)

    def generate_iid_sample(self, nb_samples) -> (np.ndarray, np.ndarray, np.ndarray):
        current_states = self.markov_reward_proc.generate_invariant_distribution(nb_samples)
        next_states = self.markov_reward_proc.sample_transition(current_states)
        rewards = self.markov_reward_proc.reward(current_states)
        return current_states, next_states, rewards
    
    def compute_temporal_difference(self, current_states: np.ndarray, next_states: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        #current_states admits a shape of (nb_copies_to_average)
        td = rewards + self.gamma_vect.reshape(-1,1) * self.parametrization.evaluate(next_states, with_average=False) - self.parametrization.evaluate(current_states, with_average=False)
        #td admits a shape of (nb_parallel_copies,nb_copies_to_average)
        return td
    
    def update_parameters(self, alpha_vect: np.ndarray):
        current_states, next_states, rewards = self.generate_iid_sample(self.parametrization.nb_copies_to_average)
        #current_states admits a shape of (nb_copies_to_average)
        td = self.compute_temporal_difference(current_states, next_states, rewards).reshape(self.parametrization.nb_parallel_copies,
                                                                                            self.parametrization.nb_copies_to_average,
                                                                                            1)
        #td admits a shape of (nb_parallel_copies,nb_copies_to_average,1)
        
        feature_vectors = self.parametrization.feature(current_states).reshape(1,current_states.shape[0],-1)
        #feature_vectors admits a shape of (1,nb_copies_to_average,num_features)
        self.parametrization.update_theta(alpha_vect.reshape(self.parametrization.nb_parallel_copies,1,-1)
                                           * td * feature_vectors)
        
    def update_parameters_PCTD0(self, alpha_vect: np.ndarray):
        current_states_1, next_states_1, rewards_1 = self.generate_iid_sample(self.parametrization.nb_copies_to_average)
        current_states_2, next_states_2, rewards_2 = self.generate_iid_sample(self.parametrization.nb_copies_to_average)
        #current_states admits a shape of (nb_copies_to_average)
        pctd = 0.5*(self.compute_temporal_difference(current_states_1, next_states_1, rewards_1)
                - self.compute_temporal_difference(current_states_2, next_states_2, rewards_2)).reshape(self.parametrization.nb_parallel_copies,
                                                                                            self.parametrization.nb_copies_to_average,
                                                                                            1)
        #td admits a shape of (nb_parallel_copies,nb_copies_to_average,1)
        
        feature_vectors = (self.parametrization.feature(current_states_1)
                           - self.parametrization.feature(current_states_2)).reshape(1,current_states_1.shape[0],-1)
        #feature_vectors admits a shape of (1,nb_copies_to_average,num_features)
        self.parametrization.update_theta(alpha_vect.reshape(self.parametrization.nb_parallel_copies,1,-1)
                                           * pctd * feature_vectors )

    def train(self, num_iterations: int, alpha_vect: np.ndarray, compute_mse: bool = False, compute_mse_averaged: bool = False, indeces_mse_computation=[]) -> (list, list):
        mse_list = []
        mse_list_averaged = []
        for iter in range(1,num_iterations+1):
            if self.do_pctd:
                self.update_parameters_PCTD0(alpha_vect)
            else:
                self.update_parameters(alpha_vect)
            if iter in indeces_mse_computation:
                if compute_mse:
                    #mse_list.append(self.compute_MSE(with_average=False))
                    mse_list.append(self.compute_errors(with_average=False))
                if compute_mse_averaged & self.parametrization.do_average:
                    #mse_list_averaged.append(self.compute_MSE(with_average=True))
                    mse_list_averaged.append(self.compute_errors(with_average=True))
        mse_list = np.array(mse_list)
        mse_list_averaged = np.array(mse_list_averaged)
        #any mse_list should be of size (len(indeces_mse_computation),3, nb_parallel_copies, nb_copies_to_average)
        return mse_list, mse_list_averaged

    def compute_MSE(self, with_average: bool = False) -> float:
        learned_values = self.parametrization.evaluate_all_states(with_average=with_average)
        mse = np.sum(self.markov_reward_proc.stat_dist.reshape(1,-1) * (learned_values - self.values_at_solution)**2,axis=1)
        return mse
    
    def compute_errors(self, with_average: bool = False) -> float:
        difference_values = self.parametrization.evaluate_all_states(with_average=with_average) - self.values_at_solution
        #difference_values should be of size (nb_parallel_copies,nb_copies_to_average,num_states)
        difference_next_values = self.markov_reward_proc.transition_operator_on_functions(difference_values)
        #difference_next_values should be of size (nb_parallel_copies, nb_copies_to_average, num_states)
        mse_values = np.sum(self.markov_reward_proc.stat_dist.reshape(1,1,-1) * difference_values**2,axis=-1)
        mse_S = np.sum(self.markov_reward_proc.stat_dist.reshape(1,1,-1) * difference_values*(difference_values 
                                                                                              - self.gamma_vect * difference_next_values),axis=-1)
        mse_advantages = (mse_S - (1. - self.gamma_vect.reshape(-1,1)) * mse_values) / self.gamma_vect.reshape(-1,1)
        #any mse should be of size (nb_parallel_copies, nb_copies_to_average)
        return mse_S, mse_values, mse_advantages
        th = (self.parametrization.theta_average if with_average else self.parametrization.theta)
        mse_S = ((th -  self.theta_star).reshape(1,-1) @ self.S @ (th - self.theta_star).reshape(-1,1)).reshape(-1)
        mse_values = (th -  self.theta_star) @ self.Sigma @ (th - self.theta_star)
        mse_advantages = mse_S - (1.-self.gamma) * mse_values
        return mse_S, mse_values, mse_advantages
        
    #def compute_squared_residual(self, with_average: bool = False) -> float:
    #    current_values = self.parametrization.evaluate_all_states(with_average=with_average)
    #    expected_next_values = self.markov_reward_proc.transition_operator_on_functions(current_values)
    #    expected_rewards = self.markov_reward_proc.expected_reward
    #    expected_td = expected_rewards + self.gamma * expected_next_values - current_values
    #    return np.mean(expected_td**2)

def generate_U_stochastic(d,dirichlet_param=0.1):
    U_mat = np.random.dirichlet(dirichlet_param * np.ones(d), size=d)
    return U_mat

def generate_U_bistochastic(d,m=None, dirichlet_param=0.1):
    if m is None:
        m = d*d
    if d == 2:
        raise ValueError('Bistochastic matrix of dimensions 2 are symmetric')
    symmetric = True
    while symmetric:
        weights = np.random.dirichlet(alpha=dirichlet_param*np.ones(m))
        U = np.zeros((d,d))
        for i in range(m):
            perm = np.random.permutation(d)
            P = np.zeros((d,d))
            for j in range(d):
                P[j, perm[j]] = 1.
            U += weights[i] * P
        symmetric = np.allclose(U, U.T, atol=1e-12, rtol=0)
    return U

def compute_mean_MSE_with_controlled_EV(d, gamma, alpha, std_dev_noise, initial_theta, dirichlet_param, diag_Sigma_half, nb_iter, nb_trials,indeces_mse_computation):
    np.random.seed(1)
    U = generate_U_bistochastic(d, dirichlet_param=dirichlet_param)
    #U = generate_U_stochastic(d, dirichlet_param=dirichlet_param)

    expected_reward = np.zeros(d)

    MRP = Markov_reward_process(transition_matrix=U, expected_reward=expected_reward, std_dev_noise=std_dev_noise)
    parametrization = Diagonal_Parametrization(eigenvalues=diag_Sigma_half, initial_theta=initial_theta,
                                                nb_parallel_copies=1, nb_copies_to_average=nb_trials)

    td0_agent = TD0_Linear_Functional_Approximation(markov_reward_proc=MRP, parametrization=parametrization,
                                                     gamma_vect=np.array([gamma]))



    #td0_agent.train(nb_iter_pretrain, alpha_pretrain)
    parametrization.do_average = True
    mse, mse_averaged = td0_agent.train(nb_iter, np.array([alpha]), compute_mse_averaged=True, indeces_mse_computation = indeces_mse_computation)
    #mse_averaged should be of size (len(indeces_mse_computation), 3, 1, nb_trials)
    #list_mse.append(mse)
    #list_mse_averaged.append(mse_averaged)

    #mse_mean = np.mean(list_mse, axis=-1).reshape(-1)
    mse_averaged_mean = np.mean(mse_averaged, axis=-1).reshape(-1,3)
    #mse_std = np.std(list_mse, axis=-1)
    #mse_averaged_std = np.std(list_mse_averaged, axis=-1)

    return mse_averaged_mean


def compute_mean_MSE_different_gamma(d, gamma_vect, alpha_vect, std_dev_noise, initial_theta, dirichlet_param,
                                      diag_Sigma_half, nb_iter, nb_copies_to_average, indeces_mse_computation):
    np.random.seed(1)
    #U = generate_U_bistochastic(d, dirichlet_param=dirichlet_param)
    U = generate_U_stochastic(d, dirichlet_param=dirichlet_param)

    expected_reward = np.zeros(d)

    MRP = Markov_reward_process(transition_matrix=U, expected_reward=expected_reward, std_dev_noise=std_dev_noise)
    parametrization = Diagonal_Parametrization(eigenvalues=diag_Sigma_half, initial_theta=initial_theta,
                                                nb_parallel_copies=gamma_vect.shape[0], 
                                                nb_copies_to_average=nb_copies_to_average)

    td0_agent = TD0_Linear_Functional_Approximation(markov_reward_proc=MRP, parametrization=parametrization,
                                                     gamma_vect=gamma_vect)

    parametrization.do_average = True
    _, mse_averaged = td0_agent.train(nb_iter, alpha_vect, compute_mse_averaged=True, indeces_mse_computation = indeces_mse_computation)
    #mse_averaged should be of size (len(indeces_mse_computation), 3, gamma_vect.shape[0], nb_copies_to_average)
    return np.mean(mse_averaged, axis=-1)
    #the output should be of size (len(indeces_mse_computation), 3, gamma_vect.shape[0])


def compute_mean_MSE_averaged(num_states, num_features, gamma_vect, alpha_vect, expected_reward, std_dev_noise, initial_theta,
                              dirichlet_param, nb_iter, nb_copies_to_average, indeces_mse_computation, nb_trials):
    list_mse_averaged = []
    for tr in range(nb_trials):
        np.random.seed(tr)
        U = generate_U_stochastic(num_states, dirichlet_param=dirichlet_param)

        MRP = Markov_reward_process(transition_matrix=U, expected_reward=expected_reward, std_dev_noise=std_dev_noise)
        parametrization = Linear_Parametrization(num_states=num_states, num_features=num_features,
                                                 initial_theta=initial_theta, nb_parallel_copies=gamma_vect.shape[0], 
                                                 nb_copies_to_average=nb_copies_to_average, do_average=True)

        parametrization.feature_matrix = np.random.dirichlet(dirichlet_param * np.ones(num_states), size=num_features).T
        print('the shape of the feature matrix is:' + str(parametrization.feature_matrix.shape))

        td0_agent = TD0_Linear_Functional_Approximation(markov_reward_proc=MRP, parametrization=parametrization, gamma_vect=gamma_vect)
        print(np.linalg.eig(td0_agent.Sigma)[0][-1])

        #nb_iter_pretrain = int(0)
        #alpha_pretrain = 0.01

        #td0_agent.train(nb_iter_pretrain, alpha_pretrain)
        parametrization.do_average = True
        mse, mse_averaged = td0_agent.train(nb_iter, alpha_vect, compute_mse_averaged=True, indeces_mse_computation=indeces_mse_computation)
        #list_mse.append(mse)
        list_mse_averaged.append(mse_averaged)

    mse_averaged_array = np.array(list_mse_averaged)
    print(f"Shape of mse_averaged_array: {mse_averaged_array.shape}")
    
    #mse_mean = np.mean(list_mse, axis=0)
    #mse_averaged_mean = np.mean(list_mse_averaged, axis=0)
    #mse_std = np.std(list_mse, axis=0)
    #mse_averaged_std = np.std(list_mse_averaged, axis=0)

    return mse_averaged_array

if __name__ == "__main__":
    t0 = time.perf_counter()
    num_states = int(2**(10))
    num_features = int(2**4)
    dirichlet_param = 0.1
    gamma = 0.9
    std_dev_noise = 0.1
    nb_iter = int(1e5)
    batch_size = 1
    #expected_reward = np.zeros(num_states)
    expected_reward = np.random.uniform(-1.,1., size=num_states)
    
    nb_trials = 1

    mean_MSE_averaged = compute_mean_MSE_averaged(num_states, num_features, gamma, expected_reward, std_dev_noise, dirichlet_param, nb_iter, nb_trials,batch_size=batch_size)
    plt.plot(mean_MSE_averaged)


    
    dt = time.perf_counter() - t0
    print(f"Total computation time: {dt:.2f} seconds")

    #plt.plot(mse_mean, label='MSE')
    #plt.plot(mse_averaged_mean, label='MSE with Averaging')
    #plt.fill_between(range(len(mse_averaged_mean)), mse_averaged_mean - mse_averaged_std, mse_averaged_mean + mse_averaged_std, alpha=0.2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    #plt.title('MSE of TD(0) on Markov Chain with Bistochastic Transition')
    plt.legend(loc='lower left')
    plt.grid()
    plt.savefig("td0_markov_chain_mse_plot.png")
    plt.show()


############not up to date with the rest of the code, just a draft for now, not used in the paper
class Haar_Parametrization(Linear_Parametrization):
    def __init__(self, num_states: int, num_features: int, do_average: bool = False):
        super().__init__(num_states=num_states, num_features=num_features, do_average=do_average)
        if np.log2(num_states) % 1 != 0:
            raise ValueError("Number of states must be a power of 2 for Haar parametrization.")
        if np.log2(num_features) % 1 != 0:
            raise ValueError("Number of features must be a power of 2 for Haar parametrization.")

        self.log2_num_states = int(np.log2(num_states))
        self.log2_num_features = int(np.log2(num_features))

        self.feature_matrix = self.compute_feature_matrix()

    def compute_feature_matrix(self) -> np.ndarray:
        vectors = [np.ones(self.num_states, dtype=float) / np.sqrt(self.num_features)]
        #vectors = [np.ones(self.num_states, dtype=float) / np.sqrt(self.log2_num_features + 1)]
        if self.log2_num_features == 0:
            output = np.array(vectors).T
        else:
            aux_vect = np.arange(self.num_states)[None,:]
            for i in range(self.log2_num_features):
                nb_vect = int(2**i)
                length_support = self.num_states // nb_vect
                half_length_support = length_support // 2

                start = (np.arange(nb_vect) * length_support)[:,None]
                positive_part = (aux_vect >= start) & (aux_vect < start + half_length_support)
                negative_part = (aux_vect >= start + half_length_support) & (aux_vect < start + length_support)

                normalization = 2**(0.5 * (self.log2_num_features - i))
                #normalization = np.sqrt(self.log2_num_features + 1)

                vectors.append((positive_part.astype(float) - negative_part.astype(float)) / normalization)
            
            output = np.vstack(vectors).T / np.sqrt(1. + self.log2_num_features)
        return output
 