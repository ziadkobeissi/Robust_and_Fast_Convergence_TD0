import numpy as np
import matplotlib.pyplot as plt

def apply_style(ax):
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Iterations', fontsize=14)
    ax.set_ylabel('MSE', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)  
    #ax.legend(loc='upper right', fontsize=11)
    ax.legend(loc='lower left', fontsize=11)
    ax.set_ylim(bottom=1e-6, top=1e0)
    #ax.set_ylim(bottom=1e-6, top=5e1)
    ax.set_xlim(left=1e1, right=1e7)
    #ax.grid(True, which="both")
    ax.grid(True, which="major")
    ax.grid(False, which="minor")

def sci_label(x, precision=1):
    exp = int(np.floor(np.log10(abs(x))))
    coeff = x / 10**exp
    return rf"{coeff:.{precision}f} \times 10^{{{exp}}}"

if __name__ == "__main__":
    d = 3
    gamma = 0.99
    alpha = (1.-gamma)/4.
    with_bias = True
    with_variance = False

    postfix ='CEV_TD0_d' + str(d) + '_gamma_' + str(gamma) + '_alpha_' + str(alpha) + ('_WB' if with_bias else '_WoutB') + ('_WV' if with_variance else '_WoutV')
    savename = 'experiments/data/' + postfix + '.npz'

    # Chargement
    data = np.load(savename)
    to_draw = data["MSE_values"]
    indeces_mse_comput = data["indeces_mse_comput"]
    omegas = data["omegas"]
    start_plot = data["start_plot"]
    #upper_bound_slope = data["upper_bound_slope"]


    #sanity check on the computational time 
    print(f"Total computation time: {data['computation_time_seconds']:.2f} seconds")


    #compute the constant for the upper bound
    mse_sup = np.max(to_draw,axis=0)
    upper_bound_constant = np.max(indeces_mse_comput * mse_sup,axis=0)
    print('The slope of the upper bound in log-log scale is ' + str(upper_bound_constant))

    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    for k in range(len(omegas)):
        ax.plot(indeces_mse_comput,to_draw[k,:], label=rf'$\omega = {sci_label(omegas[k])}$')
    #ax.set_title(titles[d], fontsize=14)
    upper_bound = upper_bound_constant / indeces_mse_comput
    plt.plot(indeces_mse_comput,upper_bound, '--',linewidth=4, label='y=C/x', color='pink')
    apply_style(ax)

    fig.savefig('experiments/figures/' + postfix + '.png', dpi=300, bbox_inches="tight")
    plt.close(fig)
