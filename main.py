import juliacall

import argparse
import glob
import os
import time
import textattack

from planners import (
    BasePlanner,
    LlmIcPddlPlanner, 
    LlmPddlPlanner, 
    LlmPlanner, 
    LlmIcPlanner, 
    LlmSbSPlanner, 
    LlmTotPlanner
)
from domains import (
    DOMAINS,
    Barman,
    Floortile,
    Termes,
    Tyreworld,
    Grippers,
    Storage,
    Blocksworld,
    Manipulation
)

available_planners = {
    "llm_ic_pddl"   : LlmIcPddlPlanner(),
    "llm_pddl"      : LlmPddlPlanner(),
    "llm"           : LlmPlanner(),
    "llm_ic"        : LlmIcPlanner(),
    "llm_stepbystep": LlmSbSPlanner(),
    "llm_tot_ic"    : LlmTotPlanner()
    
}

def run_experiment(args, method, planner: BasePlanner, domain):

    context          = domain.get_context()
    domain_pddl      = domain.get_domain_pddl()
    domain_pddl_file = domain.get_domain_pddl_file()
    domain_nl        = domain.get_domain_nl()
    domain_nl_file   = domain.get_domain_nl_file()

    # create the tmp / result folders
    problem_folder = f"./experiments/run{args.run}/problems/{method}/{domain.name}"
    plan_folder    = f"./experiments/run{args.run}/plans/{method}/{domain.name}"

    os.makedirs(problem_folder, exist_ok=True)
    os.makedirs(plan_folder, exist_ok=True)

    task = args.task

    if(args.command == "robustness-experiment"):

        perturbations_folder = f"./experiments/run{args.run}/perturbed_descriptions/"

        task_suffix = domain.get_task_suffix(task)
        task_suffix = os.path.splitext(task_suffix)[0]

        for fn in glob.glob(f"{perturbations_folder}/{task_suffix}_*"):
            perturbed_task_suffix = f"{domain.name}/{os.path.splitext(os.path.basename(fn))[0]}.pddl"
            with open(fn, "r") as f:
                perturbed_task_nl = f.read()
                
                start_time = time.time()

                planner.set_context(context)
                plan, task_pddl = planner.run_planner(perturbed_task_nl, domain_nl, domain_pddl)

                end_time = time.time()

                if (task_pddl):
                    task_pddl_file_name = f"./experiments/run{args.run}/problems/{method}/{perturbed_task_suffix}"
                    with open(task_pddl_file_name, "w") as f:
                        f.write(task_pddl)

                plan_pddl_file_name = f"./experiments/run{args.run}/plans/{method}/{perturbed_task_suffix}"
                with open(plan_pddl_file_name, "w") as f:
                    f.write(plan)

                print(f"[info] task {task} takes {end_time - start_time} sec")

    else:
        task_suffix = domain.get_task_suffix(task)
        task_nl, _ = domain.get_task(task) 
        
        start_time = time.time()

        planner.set_context(context)
        plan, task_pddl = planner.run_planner(task_nl, domain_nl, domain_pddl)

        end_time = time.time()

        if (task_pddl):
            task_pddl_file_name = f"./experiments/run{args.run}/problems/{method}/{task_suffix}"
            with open(task_pddl_file_name, "w") as f:
                f.write(task_pddl)

        plan_pddl_file_name = f"./experiments/run{args.run}/plans/{method}/{task_suffix}"
        with open(plan_pddl_file_name, "w") as f:
            f.write(plan)

        print(f"[info] task {task} takes {end_time - start_time} sec")

def print_all_prompts():
    for domain_name in DOMAINS:
        domain = eval(domain_name.capitalize())()
        context = domain.get_context()
        domain_pddl = domain.get_domain_pddl()
        domain_pddl_file = domain.get_domain_pddl_file()
        domain_nl = domain.get_domain_nl()
        
        folders = [ f"./prompts/{method}/{domain.name}" for method in available_planners.keys() if method != "llm_tot_ic"]
        for folder_name in folders:
            os.makedirs(folder_name, exist_ok=True)

        for task in range(len(domain)):
            task_nl_file, task_pddl_file = domain.get_task_file(task) 
            task_nl, task_pddl = domain.get_task(task) 
            task_suffix = domain.get_task_suffix(task)

            for method in available_planners:
                if method != "llm_tot_ic":
                    planner = available_planners[method]
                    planner.set_context(context)
                    prompt = planner._create_prompt(task_nl, domain_nl)

                    with open(f"./prompts/{method}/{task_suffix}.prompt", "w") as f:
                        f.write(prompt)

def produce_perturbations(args, domain):

    task = args.task

    # produce perturbed instructions
    task_nl, _ = domain.get_task(task)
    task_suffix = domain.get_task_suffix(task)
    task_suffix = os.path.splitext(task_suffix)[0]

    augmenter_classes = {
        "wordnet": textattack.augmentation.recipes.WordNetAugmenter,
        "charswap": textattack.augmentation.recipes.CharSwapAugmenter,
        "back_trans": textattack.augmentation.recipes.BackTranslationAugmenter,
        "back_transcription": textattack.augmentation.recipes.BackTranscriptionAugmenter
    }

    augmenter = augmenter_classes[args.perturbation_recipe](
                                                    pct_words_to_swap=args.pct_words_to_swap, 
                                                    transformations_per_example=10)
    perturbed_task_nl_list = augmenter.augment(task_nl)

    # create the tmp / result folders
    perturbations_folder = f"./experiments/run{args.run}/perturbed_descriptions/"

    if not os.path.exists(perturbations_folder):
        os.makedirs(f"{perturbations_folder}/{domain.name}", exist_ok=True)
    
    for i in range(0, len(perturbed_task_nl_list)):

        with open(f"{perturbations_folder}/{task_suffix}_{i+1}.nl", "w") as f:
            f.write(perturbed_task_nl_list[i])


import argparse

def create_common_args():
    common_args = argparse.ArgumentParser(add_help=False)
    common_group = common_args.add_argument_group('common arguments')
    common_group.add_argument('--domain', type=str, choices=DOMAINS, default="barman")
    common_group.add_argument('--time-limit', type=int, default=200)
    common_group.add_argument('--task', type=int, default=0)
    common_group.add_argument('--run', type=int, default=-1)
    common_group.add_argument('--print-prompts', action='store_true')
    return common_args

def create_parser():
    common_args = create_common_args()
    
    parser = argparse.ArgumentParser(description="LLM-Planner", parents=[common_args])
    parser.add_argument('--method', type=str, choices=["llm_ic_pddl",
                                                        "llm_pddl",
                                                        "llm",
                                                        "llm_stepbystep",
                                                        "llm_ic",
                                                        "llm_tot_ic"],
                                                        default="llm_ic_pddl",
                                                        nargs="+"
                                                        )
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest='command')

    # Create the robustness experiment subcommand
    robustness_parser = subparsers.add_parser('robustness-experiment',
                                              help='Run robustness experiment',
                                              parents=[common_args])

    # Add additional arguments specific to robustness experiment
    robustness_parser.add_argument('--method', type=str, choices=["llm_ic_pddl",
                                                                    "llm_pddl",
                                                                    "llm",
                                                                    "llm_stepbystep",
                                                                    "llm_ic",
                                                                    "llm_tot_ic"
                                                                    ],
                                                                    default="llm_ic_pddl",
                                                                    nargs="+"
                                                                    )
    robustness_parser.add_argument('--perturbation-recipe', type=str, choices=[
                                                                                "wordnet",
                                                                                "charswap",
                                                                                "back_trans",
                                                                                "back_transcription"
                                                                                ]
                                                                                )
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
    domain = eval(args.domain.capitalize())()

    # Produce perturbations if needed
    if args.command == "robustness-experiment":
        produce_perturbations(args, domain)

    # execute the llm planner

    if args.print_prompts:
        print_all_prompts()
    else:
        for method in args.method:
            run_experiment(args, method, available_planners[method], domain)
