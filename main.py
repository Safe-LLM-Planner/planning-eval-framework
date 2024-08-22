import juliacall

import argparse
import os

from domains import available_domains
from experiment_runner import ExperimentRunner, print_all_prompts, available_textattack_perturbations
from planners import available_planners

import argparse

def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"Invalid value: {value}. It must be an integer greater than 0.")
    return ivalue

def create_common_args():
    common_args = argparse.ArgumentParser(add_help=False)
    common_group = common_args.add_argument_group('common arguments')
    common_group.add_argument('--domain', type=str, choices=available_domains.keys())
    common_group.add_argument('--time-limit', type=int, default=200)
    common_group.add_argument('--task', type=positive_int, )
    common_group.add_argument('--run', type=int, default=-1)
    common_group.add_argument('--print-prompts', action='store_true')
    common_group.add_argument('--method', type=str, choices=available_planners.keys(), nargs="+")
    return common_args

def create_parser():
    common_args = create_common_args()
    
    parser = argparse.ArgumentParser(description="LLM-Planner", parents=[common_args])
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest='command')

    # Create the robustness experiment subcommand
    robustness_parser = subparsers.add_parser('robustness-experiment',
                                              help='Run robustness experiment',
                                              parents=[common_args])

    # Add additional arguments specific to robustness experiment
    robustness_parser.add_argument('--perturbation-recipe', type=str, choices=available_textattack_perturbations.keys())
    robustness_parser.add_argument('--pct-words-to-swap', type=restricted_float, help='Percentage of words to transform')

    return parser

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x

def save_args_to_file(args, filename):
    # Convert args namespace to a dictionary
    args_dict = vars(args)
    
    # Write to file
    with open(filename, 'w') as f:
        for key, value in args_dict.items():
            f.write(f"{key}: {value}\n")

def find_next_missing_run(directory):
    # List all items in the directory
    items = os.listdir(directory)
    
    # Filter out directories that start with 'run' and extract the numbers
    run_numbers = []
    for item in items:
        if item.startswith('run') and item[3:].isdigit():
            run_numbers.append(int(item[3:]))
    
    # Find the next missing number
    if run_numbers:
        next_run = max(run_numbers) + 1
    else:
        next_run = 0  # If no 'run' directories exist, start with 0
    
    return next_run

if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()
    
    # if run number is not set, compute next one
    if args.run == -1:
        args.run = find_next_missing_run("./experiments")

    # log cli arguments
    args_filepath = f"./experiments/run{args.run}/cli_args"
    os.makedirs(os.path.dirname(args_filepath))
    save_args_to_file(args, args_filepath)

    # initialize problem domain
    domain = available_domains[args.domain]
    
    # initialize experiment runner
    exp_runner = ExperimentRunner(args, domain)

    # Produce perturbations if needed
    if args.command == "robustness-experiment":
        exp_runner.produce_perturbations()

    # execute the llm planner

    if args.print_prompts:
        print_all_prompts()
    else:
        for method in args.method:
            exp_runner.run_experiment(method)
