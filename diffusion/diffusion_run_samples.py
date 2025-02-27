import argparse
from helpers import calculate_vector_field_eigenpairs
from helpers_diffusion import z_cov, diffusion_calculate_samples_and_save_results, z_calculate_random_field_eigenpairs

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Run Poisson samples calculation.')

    # Add arguments
    parser.add_argument('--mc_samples', type=int, default=500, help='Number of Monte Carlo samples')
    parser.add_argument('--fem_res', type=int, default=10, help='FEM Mesh resolution')
    parser.add_argument('--kl_res', type=int, default=10, help='KL expansion resolution')

    # Parse the arguments
    args = parser.parse_args()

    # Use the arguments
    randomFieldV, jacobianV = calculate_vector_field_eigenpairs(args.kl_res)
    randomFieldZ = z_calculate_random_field_eigenpairs(args.kl_res, z_cov)
    while True:
        diffusion_calculate_samples_and_save_results(mc_samples=args.mc_samples, fem_res=args.fem_res, kl_res=args.kl_res, randomFieldV=randomFieldV, jacobianV=jacobianV, randomFieldZ=randomFieldZ)

if __name__ == '__main__':
    main()