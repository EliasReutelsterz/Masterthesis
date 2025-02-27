import argparse

from helpers import calculate_vector_field_eigenpairs
from helpers_diffusion import diffusion_sobol_run_samples_and_save, z_calculate_random_field_eigenpairs, z_cov

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Run Poisson rhs random samples for Sobol indices calculation.')

    # Add arguments
    parser.add_argument('--mc_samples', type=int, default=30, help='Number of Monte Carlo samples')
    parser.add_argument('--fem_res', type=int, default=8, help='FEM Mesh resolution')
    parser.add_argument('--kl_res', type=int, default=8, help='KL expansion resolution')
    parser.add_argument('--size_xi_v', type=int, default=4, help='Size of the random vector for random field V')
    parser.add_argument('--size_xi_z', type=int, default=4, help='Size of the random vector for random field Z')


    # Parse the arguments
    args = parser.parse_args()

    # Use the arguments
    randomFieldV, jacobianV = calculate_vector_field_eigenpairs(args.kl_res)
    randomFieldZ = z_calculate_random_field_eigenpairs(args.kl_res, z_cov)
    while True:
        diffusion_sobol_run_samples_and_save(mc_sample_size=args.mc_samples,
                                           fem_res=args.fem_res,
                                           kl_res=args.kl_res,
                                           size_xi_v=args.size_xi_v,
                                           size_xi_z=args.size_xi_z,
                                           randomFieldV=randomFieldV,
                                           jacobianV=jacobianV,
                                           randomFieldZ=randomFieldZ)

if __name__ == '__main__':
    main()