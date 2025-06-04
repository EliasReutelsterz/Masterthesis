import argparse
from helpers_diffusion import z_cov, diffusion_calculate_samples_and_save_results, z_calculate_random_field_eigenpairs, calculate_vector_field_eigenpairs

def main():
    """Run the diffusion samples for the Monte Carlo Analysis calculation with random samples.
    The combining of the files saved in poisson_and_variants/diffusion/mc_data_storage should be done by hand.
    The combined sample are saved in the file: poisson_and_variants/diffusion/mc_data_storage/samples_femres_{fem_res}_klres_{kl_res}_combined.csv"""
    # Create the parser
    parser = argparse.ArgumentParser(description='Run Diffusion samples calculation.')

    # Add arguments
    parser.add_argument('--mc_samples', type=int, default=30, help='Number of Monte Carlo samples')
    parser.add_argument('--fem_res', type=int, default=10, help='FEM Mesh resolution')
    parser.add_argument('--kl_res', type=int, default=14, help='KL expansion resolution')

    # Parse the arguments
    args = parser.parse_args()

    # Use the arguments
    randomFieldV, jacobianV = calculate_vector_field_eigenpairs(args.kl_res)
    randomFieldZ = z_calculate_random_field_eigenpairs(args.kl_res, z_cov)
    while True:
        diffusion_calculate_samples_and_save_results(mc_samples=args.mc_samples, fem_res=args.fem_res, kl_res=args.kl_res, randomFieldV=randomFieldV, jacobianV=jacobianV, randomFieldZ=randomFieldZ)

if __name__ == '__main__':
    main()