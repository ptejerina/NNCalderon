# filename: codebase/generate_results.py
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set plotting style
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['savefig.facecolor'] = 'white'
mpl.rcParams['text.usetex'] = False
mpl.rcParams['font.size'] = 12


def calculate_l2_error(y_true, y_pred):
    """
    Calculates the relative L2 error between two arrays.

    Args:
        y_true (np.ndarray): The ground truth array.
        y_pred (np.ndarray): The predicted array.

    Returns:
        float: The relative L2 error.
    """
    l2_error = np.linalg.norm(y_pred - y_true) / np.linalg.norm(y_true)
    return l2_error


def calculate_psnr(y_true, y_pred, data_range):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR).

    Args:
        y_true (np.ndarray): The ground truth array.
        y_pred (np.ndarray): The predicted array.
        data_range (float): The maximum possible pixel value of the image
                            (e.g., max_gamma - min_gamma).

    Returns:
        float: The PSNR value in dB.
    """
    mse = np.mean((y_true - y_pred) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(data_range / np.sqrt(mse))
    return psnr


def plot_results_for_case(case_name, results_data):
    """
    Generates and saves a comprehensive plot for a given test case.
    The plot includes ground truth, reconstructions, and error maps for all noise levels.

    Args:
        case_name (str): The name of the test case (e.g., 'single_inclusion').
        results_data (dict): A dictionary containing the ground truth and
                             prediction data for the case.
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle('Reconstruction Results for ' + case_name.replace('_', ' ').title(), fontsize=20, y=0.96)

    gamma_true = results_data['gamma_true']
    vmin, vmax = gamma_true.min(), gamma_true.max()

    # Plot Ground Truth (top-left)
    im = axes[0, 0].imshow(gamma_true, origin='lower', extent=[0, 1, 0, 1], cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('Ground Truth')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    fig.colorbar(im, ax=axes[0, 0], orientation='vertical', fraction=0.046, pad=0.04)

    # Hide unused subplots
    axes[1, 0].axis('off')
    axes[2, 0].axis('off')

    noise_levels = ['0pct', '1pct', '5pct']
    for i, noise_str in enumerate(noise_levels):
        gamma_pred = results_data[noise_str]['prediction']
        error_map = np.abs(gamma_pred - gamma_true)

        # Plot Reconstruction
        ax_recon = axes[i, 1]
        im_recon = ax_recon.imshow(gamma_pred, origin='lower', extent=[0, 1, 0, 1], cmap='viridis', vmin=vmin, vmax=vmax)
        ax_recon.set_title('Reconstruction (' + noise_str.replace('pct', '%') + ' Noise)')
        ax_recon.set_xlabel('x')
        ax_recon.set_ylabel('y')
        fig.colorbar(im_recon, ax=ax_recon, orientation='vertical', fraction=0.046, pad=0.04)

        # Plot Error Map
        ax_error = axes[i, 2]
        im_error = ax_error.imshow(error_map, origin='lower', extent=[0, 1, 0, 1], cmap='inferno', vmin=0, vmax=error_map.max())
        ax_error.set_title('Absolute Error (' + noise_str.replace('pct', '%') + ' Noise)')
        ax_error.set_xlabel('x')
        ax_error.set_ylabel('y')
        fig.colorbar(im_error, ax=ax_error, orientation='vertical', fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    plot_filename = os.path.join('data', 'plot_results_summary_' + case_name + '_' + str(int(time.time())) + '.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig)
    
    print('\nResults plot saved to ' + plot_filename)
    print('Plot description: Side-by-side comparison of ground truth, PINN reconstructions, and error maps for the "' + case_name + '" case at 0%, 1%, and 5% noise levels.')


def main():
    """
    Main function to load results, compute metrics, and generate plots.
    """
    DATA_DIR = 'data'
    CASES = ['single_inclusion', 'multiple_inclusions', 'checkerboard']
    NOISE_LEVELS = ['0pct', '1pct', '5pct']
    
    all_results = []

    print('=' * 80)
    print('Generating Quantitative and Qualitative Results')
    print('=' * 80)

    for case in CASES:
        print('\n--- Processing Case: ' + case + ' ---')
        
        # Load ground truth gamma
        try:
            gt_filepath = os.path.join(DATA_DIR, 'dtn_data_' + case + '.npz')
            gamma_true = np.load(gt_filepath)['gamma_true']
            # The FDM solver grid was 128x128, but predictions are on 256x256.
            # We need to upscale the ground truth for fair comparison.
            if gamma_true.shape[0] < 256:
                gamma_true = gamma_true.repeat(2, axis=0).repeat(2, axis=1)

        except FileNotFoundError:
            print('Ground truth file not found for case: ' + case + '. Skipping.')
            continue
            
        case_results_for_plotting = {'gamma_true': gamma_true}

        for noise_str in NOISE_LEVELS:
            pred_filename = 'pinn_results_' + case + '_' + noise_str + '_gamma_pred.npy'
            pred_filepath = os.path.join(DATA_DIR, pred_filename)
            
            if not os.path.exists(pred_filepath):
                print('Prediction file not found: ' + pred_filepath + '. Skipping.')
                continue

            gamma_pred = np.load(pred_filepath)
            
            # Ensure shapes match
            if gamma_pred.shape != gamma_true.shape:
                 print('Shape mismatch for ' + case + ' (' + noise_str + '): GT=' + str(gamma_true.shape) + ', Pred=' + str(gamma_pred.shape) + '. Skipping.')
                 continue

            # Compute metrics
            l2_err = calculate_l2_error(gamma_true, gamma_pred)
            # Data range for gamma is [0.5, 2.5] -> 2.0
            psnr = calculate_psnr(gamma_true, gamma_pred, data_range=2.0)
            
            result_entry = {
                'case': case,
                'noise': noise_str,
                'l2_error': l2_err,
                'psnr': psnr
            }
            all_results.append(result_entry)
            
            case_results_for_plotting[noise_str] = {
                'prediction': gamma_pred
            }

        # Generate plot for the case
        plot_results_for_case(case, case_results_for_plotting)

    # Print summary table
    print('\n\n' + '=' * 80)
    print('PINN Reconstruction Performance Summary')
    print('=' * 80)
    header = "| " + "Test Case".ljust(25) + " | " + "Noise".ljust(10) + " | " + "Relative L2 Error".ljust(20) + " | " + "PSNR (dB)".ljust(15) + " |"
    print(header)
    print('-' * len(header))

    for res in all_results:
        row = "| " + res['case'].replace('_', ' ').title().ljust(25) + " | "
        row += res['noise'].replace('pct', '%').ljust(10) + " | "
        row += str(round(res['l2_error'], 4)).ljust(20) + " | "
        row += str(round(res['psnr'], 2)).ljust(15) + " |"
        print(row)
    print('-' * len(header))
    print('\n')


if __name__ == "__main__":
    main()