import argparse

from helpers import calculate_vector_field_eigenpairs, poisson_sobol_run_samples_and_save

def main():
    """Run samples for Sobol indices calculation for Poisson equation.
    The resulting matrices are saved and automatically appended at poisson_and_variants/poisson/sobol_data_storage/femres_{fem_res}_klres_{kl_res}_size_of_xi_{size_of_xi}"""
    # Create the parser
    parser = argparse.ArgumentParser(description='Run Poisson samples for Sobol indices calculation.')

    # Add arguments
    parser.add_argument('--mc_samples', type=int, default=30, help='Number of Monte Carlo samples')
    parser.add_argument('--fem_res', type=int, default=8, help='FEM Mesh resolution')
    parser.add_argument('--kl_res', type=int, default=8, help='KL expansion resolution')
    parser.add_argument('--size_of_xi', type=int, default=8, help='Size of the random vector')

    # Parse the arguments
    args = parser.parse_args()

    # Use the arguments
    randomFieldV, jacobianV = calculate_vector_field_eigenpairs(args.kl_res)
    while True:
        poisson_sobol_run_samples_and_save(mc_sample_size=args.mc_samples,
                                           fem_res=args.fem_res,
                                           kl_res=args.kl_res,
                                           size_of_xi=args.size_of_xi,
                                           randomFieldV=randomFieldV,
                                           jacobianV=jacobianV)

if __name__ == '__main__':
    main()