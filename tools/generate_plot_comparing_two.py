import os
import json
import matplotlib.pyplot as plt
import argparse
import math
import numpy as np

from matplotlib.ticker import FuncFormatter
from scipy.stats import norm

planner_dir_to_label = {
    "llm_ic": "LLM-as-planner",
    "llm_ic_pddl": "LLM + Planner"
}

def load_single_directory_results(swap_dir):
    results = {}

    # Directly check for the 'evaluation' directory
    evaluation_path = os.path.join(swap_dir, 'evaluation')
    if not os.path.isdir(evaluation_path):
        print(f"  'evaluation' directory not found in: {swap_dir}")
        return results

    print(f"  Processing evaluation directory: {evaluation_path}")

    for planner_dir in os.listdir(evaluation_path):
        planner_path = os.path.join(evaluation_path, planner_dir)
        if not os.path.isdir(planner_path):
            print(f"    Skipping non-directory: {planner_path}")
            continue

        print(f"    Processing planner directory: {planner_path}")

        domain_dir = os.listdir(planner_path)[0]  # Only one domain dir per planner
        domain_path = os.path.join(planner_path, domain_dir)
        print(f"      Processing domain directory: {domain_path}")

        json_file_path = os.path.join(domain_path, 'results_summary.json')
        if not os.path.isfile(json_file_path):
            print(f"        Skipping non-file: {json_file_path}")
            continue

        print(f"        Processing JSON file: {json_file_path}")
        with open(json_file_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"        Skipping invalid JSON file: {json_file_path}")
                continue

        if planner_dir not in results:
            results[planner_dir] = {}

        results[planner_dir] = {
            'total': data.get('total', 0),
            'valid': data.get('valid', 0),
            'successful': data.get('successful', 0),
            'safe': data.get('safe', 0)
        }
    
    return results

def wilson_score_interval(p, N, confidence=0.95):
    if N <= 0:
        raise ValueError("Sample size N must be greater than 0.")
    if not (0 <= p <= 1):
        raise ValueError("Probability p must be between 0 and 1.")
    
    # Calculate the Z-score based on the confidence level
    z = norm.ppf((1 + confidence) / 2)
    
    # Compute the components for the Wilson score interval
    denominator = 1 + z**2 / N
    centre_adjusted_probability = p + z**2 / (2 * N)
    adjusted_standard_deviation = math.sqrt((p * (1 - p) + z**2 / (4 * N)) / N)
    
    # Calculate the lower and upper bounds
    lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator
    
    # Ensure bounds are within [0, 1] range
    lower_bound = max(lower_bound, 0)
    upper_bound = min(upper_bound, 1)

    if p == 0:
        lower_bound = 0
    
    return lower_bound, upper_bound

def plot_results(results1, results2, dir1_label, dir2_label, base_dir, x_label1, x_label2):
    metrics = ['safe', 'successful']
    
    for metric in metrics:
        fig, ax1 = plt.subplots(figsize=(12, 8))

        ax2 = ax1.twinx()  # Create a duplicate y-axis on the right

        for planner in results1.keys():
            if planner not in results2:
                continue

            metric_values1 = (results1[planner][metric] / results1[planner]['total']) if results1[planner]['total'] > 0 else 0
            metric_values2 = (results2[planner][metric] / results2[planner]['total']) if results2[planner]['total'] > 0 else 0

            p1 = metric_values1
            N1 = results1[planner]['total']
            lb1, ub1 = wilson_score_interval(p1, N1)

            p2 = metric_values2
            N2 = results2[planner]['total']
            lb2, ub2 = wilson_score_interval(p2, N2)

            metric_percent1 = p1 * 100
            metric_percent2 = p2 * 100
            lb1_percent, ub1_percent = lb1 * 100, ub1 * 100
            lb2_percent, ub2_percent = lb2 * 100, ub2 * 100

            ax1.errorbar([0.4, 0.6], [metric_percent1, metric_percent2], 
                        yerr=[[metric_percent1 - lb1_percent, metric_percent2 - lb2_percent],
                            [ub1_percent - metric_percent1, ub2_percent - metric_percent2]],
                        label=planner_dir_to_label.get(planner, planner), fmt='-o', linestyle='--', markersize=8, capsize=5)

        def percent_formatter(x, _):
            return f'{x:.0f}%'

        ax1.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
        ax1.set_ylim([0, 110])
        ax1.set_xlim([0.3, 0.7])

        # Update fontsize for x-ticks
        ax1.set_xticks([0.4, 0.6])
        ax1.set_xticklabels([x_label1, x_label2], fontsize=16)

        # Update fontsize for y-ticks (left y-axis)
        ax1.tick_params(axis='y', labelsize=14)

        # Set up the right y-axis (duplicated)
        ax2.set_ylim(ax1.get_ylim())  # Match the limits of the left y-axis
        ax2.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
        ax2.tick_params(axis='y', labelsize=14)  # Update fontsize for right y-axis

        # Label for the right y-axis
        # ax2.set_ylabel(f'Percentage of {metric.capitalize()} Plans', fontsize=14, color='grey')

        # ax1.set_xlabel('Directory', fontsize=14)
        ax1.set_ylabel(f'Percentage of {metric.capitalize()} Plans', fontsize=18)
        
        plt.title(f'Percentage of {metric.capitalize()} Plans - Comparison', fontsize=16)
        ax1.legend(title='Planner', title_fontsize='20', fontsize='13', bbox_to_anchor=(0.95, 0.2))
        ax1.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        plot_file_path = os.path.join(base_dir, f'{metric}_comparison_plot.png')
        plt.savefig(plot_file_path)
        print(f"Saved {metric} comparison plot to {plot_file_path}")
        plt.close()

def main(dir1, dir2, base_dir, x_label1, x_label2):
    results1 = load_single_directory_results(dir1)
    results2 = load_single_directory_results(dir2)
    plot_results(results1, results2, os.path.basename(dir1.rstrip('/')), os.path.basename(dir2.rstrip('/')), base_dir, x_label1, x_label2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate comparison plots from two experiment directories.')
    parser.add_argument('dir1', type=str, help='First directory containing the experiment results.')
    parser.add_argument('dir2', type=str, help='Second directory containing the experiment results.')
    parser.add_argument('base_dir', type=str, help='Directory where to save the plots.')
    parser.add_argument('x_label1', type=str, help='Custom label for the first x-axis tick.')
    parser.add_argument('x_label2', type=str, help='Custom label for the second x-axis tick.')
    args = parser.parse_args()
    
    main(args.dir1, args.dir2, args.base_dir, args.x_label1, args.x_label2)
