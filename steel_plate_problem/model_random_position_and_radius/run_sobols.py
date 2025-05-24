import argparse
from helpers import calculate_randomFieldE, sp_5_sobol_run_samples_and_save

def main():
    """Run Sobol samples for steel plate model with random position and radius.
    Saves the resulting matrices at steel_plate_problem/model_random_position_and_radius/sobol_data_storage/klres_e_{kl_res_e}_femres_{fem_res}_size_of_xi_e_{size_of_xi_e}"""
    
    # Create the parser
    parser = argparse.ArgumentParser(description='Run Sobol samples for steel plate model with random position and radius.')

    # Add arguments
    parser.add_argument('--mc_samples', type=int, default=10, help='Number of Monte Carlo samples')
    parser.add_argument('--fem_res', type=int, default=14, help='FEM Mesh resolution')
    parser.add_argument('--kl_res_e', type=int, default=14, help='KL expansion Random Field E resolution')

    # Parse the arguments
    args = parser.parse_args()

    size_of_xi_e = 4

    # Use the arguments
    randomFieldE = calculate_randomFieldE(mesh_resolution=args.kl_res_e)
    while True:
        sp_5_sobol_run_samples_and_save(mc_sample_size=args.mc_samples,
                                fem_res=args.fem_res,
                                kl_res_e=args.kl_res_e,
                                size_of_xi_e=size_of_xi_e,
                                randomFieldE=randomFieldE)

if __name__ == '__main__':
    main()