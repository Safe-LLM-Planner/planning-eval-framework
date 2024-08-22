import juliacall

import argparse
import os

from collections import namedtuple
from config import DEFAULT_PYD_GENERATORS, DEFAULT_PLAN_MATCHER
from domains import available_domains
from experiment_runner import ExperimentRunner, print_all_prompts, available_textattack_perturbations
from planners import available_planners
from plan_evaluator import available_plan_matchers
from pydantic_generator import available_pydantic_generators

PlannerPydModelTuple = namedtuple("PlannerPydModelTuple", ["planner", "pyd_gen"])

def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"Invalid value: {value}. It must be an integer greater than 0.")
    return ivalue

def validate_planner(name: str):
    if name not in available_planners.keys():
        raise argparse.ArgumentTypeError(f"Invalid value '{name}' for planner. Must be one of {available_planners.keys()}")

def validate_pydantic_generator(name: str):
    if name not in available_pydantic_generators.keys():
        raise argparse.ArgumentTypeError(f"Invalid value '{name}' for pydantic response model generator. Must be one of {available_pydantic_generators.keys()}")

def method_tuple(s):
    parts = s.split(',')

    # Handle the case where only one value is provided
    if len(parts) == 1:
        planner_name = parts[0].strip()
        method = PlannerPydModelTuple(planner_name, DEFAULT_PYD_GENERATORS[planner_name])
    elif len(parts) == 2:
        method = PlannerPydModelTuple(parts[0].strip(), parts[1].strip())
    else:
        raise argparse.ArgumentTypeError("Tuple must have one or two elements")
    
    # Validate each element
    validate_planner(method.planner)
    validate_pydantic_generator(method.pyd_gen)
    
    return method

method_tuple_help_text = (
            f"Provide one or more tuples in the format 'planner,pyd_gen'. Valid planners: {list(available_planners.keys())}. "
            f"Valid Pydantic generators: {list(available_pydantic_generators.keys())}. If only one value is provided, the default for the second value depends on the planner: '{DEFAULT_PYD_GENERATORS}'."
    )

def create_common_args():
    common_args = argparse.ArgumentParser(add_help=False)
    common_group = common_args.add_argument_group('common arguments')
    common_group.add_argument('--domain', type=str, choices=available_domains.keys())
    common_group.add_argument('--plan-matcher', type=str, choices=available_plan_matchers.keys(), default=DEFAULT_PLAN_MATCHER)
    common_group.add_argument('--time-limit', type=int, default=200)
    common_group.add_argument('--task', type=positive_int, )
    common_group.add_argument('--run', type=int, default=-1)
    common_group.add_argument('--print-prompts', action='store_true')
    common_group.add_argument('--method', type=method_tuple, nargs="+", help=method_tuple_help_text)
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
        for (planner_name, pyd_generator) in args.method:
            exp_runner.set_experiment(planner_name, pyd_generator, args.plan_matcher)
            exp_runner.run_experiment()
