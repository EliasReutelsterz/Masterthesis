import argparse
from model_sp_5_helpers import calculate_randomFieldE, sp_5_sobol_run_samples_and_save

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Run Steel Plate Problem sp5 samples calculation.')

    # Add arguments
    parser.add_argument('--mc_samples', type=int, default=10, help='Number of Monte Carlo samples')
    parser.add_argument('--fem_res', type=int, default=12, help='FEM Mesh resolution')
    parser.add_argument('--kl_res_e', type=int, default=12, help='KL expansion Random Field E resolution')

    # Parse the arguments
    args = parser.parse_args()

    # Use the arguments
    randomFieldE = calculate_randomFieldE(mesh_resolution=args.kl_res_e)
    while True:
        sp_5_sobol_run_samples_and_save(mc_sample_size=args.mc_samples,
                                fem_res=args.fem_res,
                                kl_res_e=args.kl_res_e,
                                size_of_xi_e=4,
                                randomFieldE=randomFieldE)

if __name__ == '__main__':
    main()