import numpy as np
import time

from Markov_Chain import *


if __name__ == "__main__":
    t0 = time.perf_counter()
    seed = 2
    np.random.seed(seed)
    d = 3
    dirichlet_param = 1.
    gamma = 0.99
    alpha = (1.-gamma)/4.
    nb_iter = int(1e7)
    with_bias = False
    with_variance = True

    nb_trials = 1000

    save = True
    print(f"Starting computations with d={d}, gamma={gamma}, alpha={alpha}, with_bias={with_bias}, with_variance={with_variance}, nb_trials={nb_trials}")
    print(f"Number of iterations: {nb_iter}")
    print(f"The results will be saved: {save}")

    if with_bias:
        initial_theta = np.random.uniform(-1.,1.,size=d)
    else:
        initial_theta = np.zeros(d)
    if with_variance:
        std_dev_noise = 1.#np.sqrt(0.1)
    else:
        std_dev_noise = 0.
    
    mean_MSE_averaged_list = []

    start_plot = 10

    nb_mse_computations = 200
    ratio_mse_comput = (nb_iter/start_plot)**(1./nb_mse_computations)
    indeces_mse_comput = start_plot * ratio_mse_comput ** np.arange(nb_mse_computations+1)
    indeces_mse_comput = np.unique(np.floor(indeces_mse_comput).astype(int))

    where_k = [0.,1.,2.,3.,4.]
    omegas = []
    for k in where_k:
        other_diag_coef = 0.1**(0.5*k)
        diag_Sigma_half = np.array([1.,other_diag_coef,other_diag_coef])
        #diag_Sigma_half = other_diag_coef*np.ones(d)
        omega = 1./d/10**k
        omegas.append(omega)
        mean_MSE_averaged = compute_mean_MSE_with_controlled_EV(d=d, gamma=gamma, alpha=alpha, std_dev_noise=std_dev_noise,
                                                                initial_theta=initial_theta, dirichlet_param=dirichlet_param,
                                                                diag_Sigma_half=diag_Sigma_half, nb_iter=nb_iter,
                                                                nb_trials=nb_trials, indeces_mse_computation=indeces_mse_comput)
        #plt.plot(indeces_mse_comput,mean_MSE_averaged, label=rf'$\omega = {omega}$')

        mean_MSE_averaged_list.append(mean_MSE_averaged)
        print('End of computations for k='+str(k))
    #mean_MSE_averaged should be of size (len(where_k), len(indeces_mse_comput), 3)

    dt = time.perf_counter() - t0
    print(f"Total computation time: {dt:.2f} seconds")

    mean_MSE_averaged = np.array(mean_MSE_averaged_list)
    to_save = {
        'gamma': gamma,
        'alpha': alpha,
        'dimension': d,
        'nb_iter': nb_iter,
        'nb_trials': nb_trials,
        'with_bias': with_bias,
        'with_variance': with_variance,
        'std_dev_noise': std_dev_noise,
        'dirichlet_param': dirichlet_param,
        'indeces_mse_comput': indeces_mse_comput,
        'MSE_S': mean_MSE_averaged[:,:,0],
        'MSE_values': mean_MSE_averaged[:,:,1],
        'MSE_advantages': mean_MSE_averaged[:,:,2],
        'omegas': omegas,
        'start_plot': start_plot,
        'nb_mse_computations': nb_mse_computations,
        'seed': seed,
        'computation_time_seconds': dt,
    }

    if save:
        savename = 'experiments/data/CEV_TD0_d' + str(d) + '_gamma_' + str(gamma) + '_alpha_' + str(alpha) + ('_WB' if with_bias else '_WoutB') + ('_WV' if with_variance else '_WoutV') + '.npz'
        np.savez(savename, **to_save)
