import argparse
from helpers import calculate_vector_field_eigenpairs, calculate_samples_and_save_results

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Run Poisson samples calculation.')

    # Add arguments
    parser.add_argument('--mc_samples', type=int, default=1600, help='Number of Monte Carlo samples')
    parser.add_argument('--fem_res', type=int, default=8, help='FEM Mesh resolution')
    parser.add_argument('--kl_res', type=int, default=8, help='KL expansion resolution')

    # Parse the arguments
    args = parser.parse_args()

    # Use the arguments
    randomFieldV, jacobianV = calculate_vector_field_eigenpairs(args.kl_res)
    while True:
        calculate_samples_and_save_results(mc_samples=args.mc_samples, fem_res=args.fem_res, kl_res=args.kl_res, randomFieldV=randomFieldV, jacobianV=jacobianV)

if __name__ == '__main__':
    main()