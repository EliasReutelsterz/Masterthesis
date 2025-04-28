import argparse
from sp_6_helper import calculate_randomFieldE, calculate_randomFieldVBar, sp_6_sobol_run_samples_and_save

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Run Steel Plate Problem sp6 samples calculation.')

    # Add arguments
    parser.add_argument('--mc_samples', type=int, default=10, help='Number of Monte Carlo samples')
    parser.add_argument('--fem_res', type=int, default=14, help='FEM Mesh resolution')
    parser.add_argument('--kl_res_e', type=int, default=6, help='KL expansion Random Field E resolution')
    parser.add_argument('--kl_res_v', type=int, default=6, help='KL expansion Random Field VBar resolution')
    parser.add_argument('--size_of_xi_e', type=int, default=4, help='Number of RV of KL of E to analyze')
    parser.add_argument('--size_of_xi_v', type=int, default=4, help='Number of RV of KL of V to analyze')



    # Parse the arguments
    args = parser.parse_args()

    # Use the arguments
    randomFieldE = calculate_randomFieldE(mesh_resolution=args.kl_res_e)
    randomFieldVBar = calculate_randomFieldVBar(mesh_resolution=args.kl_res_v)
    while True:
        sp_6_sobol_run_samples_and_save(mc_sample_size=args.mc_samples,
                                fem_res=args.fem_res,
                                kl_res_e=args.kl_res_e,
                                kl_res_v=args.kl_res_v,
                                size_of_xi_e=args.size_of_xi_e,
                                size_of_xi_v=args.size_of_xi_v,
                                randomFieldE=randomFieldE,
                                randomFieldVBar=randomFieldVBar)

if __name__ == '__main__':
    main()