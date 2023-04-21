import argparse
import numpy as np
from psmc.model import PSMC
from psmc.utils import process_psmcfa

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run PSMC analysis on genome data')
    parser.add_argument('input_file', type=str, help='Name of input  psmcfa file')
    parser.add_argument('output_file', type=str, help='Name of output model params file')
    parser.add_argument('n_iter', type=int, help='Number of iterations to run PSMC')
    parser.add_argument('--t_max', type=int, default=15, help='Maximum time parameter')
    parser.add_argument('--n_steps', type=int, default=64, help='Number of steps in discretization')
    parser.add_argument('--theta0', type=float, default=None, help='Initial value of theta parameter')
    parser.add_argument('--rho0', type=float, default=None, help='Initial value of rho parameter')
    parser.add_argument('--pattern', type=str, default=None, help='Pattern on how to group hidden states, e.g. 1*4+25*2+1*4+1*6')
    parser.add_argument('--batch_size', type=int, default=None, help='If batching is needed what is the batch size')
    parser.add_argument('--subset', type=int, default=None, help='Run on a subset of the data, first N sequences')
    args = parser.parse_args()

    xs = process_psmcfa(args.input_file, batch_size=args.batch_size)

    if args.theta0 is None:
        theta0 = np.sum(xs[xs == 1]) / (xs.shape[0] * xs.shape[1])
    else:
        theta0 = args.theta0

    if args.rho0 is None:
        rho0 = theta0 / 5
    else:
        rho0 = args.rho0

    if args.subset is not None:
        xs = xs[:args.subset]

    print(f"Using theta0={np.round(theta0, 4)} and rho0={np.round(rho0, 4)}")

    psmc_model = PSMC(t_max=args.t_max, n_steps=args.n_steps, theta0=theta0, rho0=rho0, pattern=args.pattern, progress_bar='terminal')
    psmc_model.param_recalculate()

    initial_params = [theta0, rho0, args.t_max] + [1.] * (psmc_model.n_free_params - 3)
    bounds = [(1e-4, 1e-1), (1e-5, 1e-1), (12, 20)] + [(0.1, 10)] * (psmc_model.n_free_params - 3)

    loss_list, params_history = psmc_model.EM(initial_params, bounds, x=xs, n_iter=args.n_iter)

    # code to save output to output file

    psmc_model.save_params(args.output_file)
