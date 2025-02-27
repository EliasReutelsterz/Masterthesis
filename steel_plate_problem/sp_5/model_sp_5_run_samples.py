import argparse
from model_sp_5_helpers import calculate_randomFieldE, run_and_save_mc

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Run Steel Plate Problem sp5 samples calculation.')

    # Add arguments
    parser.add_argument('--mc_samples', type=int, default=50, help='Number of Monte Carlo samples')
    parser.add_argument('--fem_res', type=int, default=18, help='FEM Mesh resolution')
    parser.add_argument('--kl_res_e', type=int, default=18, help='KL expansion Random Field E resolution')

    # Parse the arguments
    args = parser.parse_args()

    # Use the arguments
    randomFieldE = calculate_randomFieldE(mesh_resolution=args.kl_res_e)
    while True:
        run_and_save_mc(mesh_resolution_kl_e=args.kl_res_e,
                        mesh_resolution=args.fem_res,
                        sample_size=args.mc_samples,
                        randomFieldE=randomFieldE)

if __name__ == '__main__':
    main()