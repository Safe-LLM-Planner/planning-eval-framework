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

def load_results(base_dir):
    results = {}

    # Traverse the directory structure
    for swap_dir in os.listdir(base_dir):
        swap_path = os.path.join(base_dir, swap_dir)
        if not os.path.isdir(swap_path):
            print(f"Skipping non-directory: {swap_path}")
            continue
        
        print(f"Processing swap directory: {swap_path}")

        try:
            swap_percentage = float(swap_dir.rstrip('/').split('_')[0])
        except ValueError:
            print(f"Skipping invalid swap directory: {swap_dir}")
            continue

        # Directly check for the 'evaluation' directory
        evaluation_path = os.path.join(swap_path, 'evaluation')
        if not os.path.isdir(evaluation_path):
            print(f"  'evaluation' directory not found in: {swap_path}")
            continue

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

            results[planner_dir][swap_percentage] = {
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

def plot_results(results, base_dir):
    metrics = ['safe', 'successful']
    # metrics = ['valid', 'successful', 'safe']
    
    for metric in metrics:
        plt.figure(figsize=(12, 8))

        # Dictionary to keep track of unique planner labels
        plotted_planners = {}

        for planner, data in results.items():
            # Sort data by percentage of words swapped
            swap_sorted = sorted(data.keys())
            metric_values = [
                (data[swap][metric] / data[swap]['total']) if data[swap]['total'] > 0 else 0
                for swap in swap_sorted
            ]

            # Calculate error bars
            lower_bounds = []
            upper_bounds = []
            for swap in swap_sorted:
                p = data[swap][metric] / data[swap]['total']
                N = data[swap]['total']
                lb, ub = wilson_score_interval(p, N)
                lower_bounds.append(lb)
                upper_bounds.append(ub)
            
            # Convert to percentages for plotting
            metric_sorted_percent = [v * 100 for v in metric_values]
            lower_bounds_percent = [lb * 100 for lb in lower_bounds]
            upper_bounds_percent = [ub * 100 for ub in upper_bounds]

            # Plot with dashed line and markers
            try:
                plt.errorbar(swap_sorted, metric_sorted_percent, 
                            yerr=[np.array(metric_sorted_percent) - np.array(lower_bounds_percent),
                                np.array(upper_bounds_percent) - np.array(metric_sorted_percent)],
                            label=planner_dir_to_label[planner], fmt='-o', linestyle='--', markersize=8, capsize=5)
            except:
                print(metric)
                print(planner)
                print(np.array(metric_sorted_percent))
                print(np.array(lower_bounds_percent))
                raise ValueError

            plotted_planners[planner] = True

        # Custom y-axis formatter to add percentage signs
        def percent_formatter(x, _):
            return f'{x:.0f}%'

        plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_formatter))
        plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*100:.0f}%'))

        # Adjust the size of tick labels
        plt.tick_params(axis='both', which='major', labelsize=14)

        plt.gca().set_ylim([0, 110])
        plt.gca().set_xlim([0, 1])

        plt.xlabel('Percentage of Words Perturbed', fontsize=18)
        plt.ylabel(f'Percentage of {metric.capitalize()} Plans', fontsize=18)
        plt.title(f'Percentage of {metric.capitalize()} Plans vs. Percentage of Words Perturbed', fontsize=16)
        plt.legend(title='Planner', title_fontsize='20', fontsize='13', bbox_to_anchor=(0.95,0.2))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save the plot in the base directory with the original filenames
        plot_file_path = os.path.join(base_dir, f'{metric}_plot.png')
        plt.savefig(plot_file_path)
        print(f"Saved {metric} plot to {plot_file_path}")
        plt.close()


def main(base_dir):
    results = load_results(base_dir)
    plot_results(results, base_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate histograms from experiment results.')
    parser.add_argument('base_dir', type=str, help='Base directory containing the experiment results.')
    args = parser.parse_args()
    
    main(args.base_dir)
