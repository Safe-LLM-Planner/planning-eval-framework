import glob
import json
import os
import time
from typing import Literal

from . import text_transformations
from .domains import Domain
from llm_planners.planners import available_planners, PlannerResult
from .plan_evaluator import PlanEvaluator, available_plan_matchers

class ExperimentRunner():
    def __init__(self, args, domain: Domain):
        self.args = args
        self.domain = domain

    def set_experiment(self, planner_name: str, 
                             response_model_generator_name: str, 
                             plan_matcher_name: str,
                             pct_words_to_swap: float = None):
        self.planner_name = planner_name
        self.response_model_generator_name = response_model_generator_name
        self.plan_matcher_name = plan_matcher_name

        swap_subdir_name = ""
        if pct_words_to_swap is not None:
            swap_subdir_name = f"{pct_words_to_swap}_swap"

        # set experiment dirs
        self.problem_dir = f"./experiments/run{self.args.run}/{swap_subdir_name}/problems/{self.planner_name}/{self.domain.name}"
        self.plan_dir    = f"./experiments/run{self.args.run}/{swap_subdir_name}/plans/{self.planner_name}/{self.domain.name}"
        self.evaluation_dir = f"./experiments/run{self.args.run}/{swap_subdir_name}/evaluation/{self.planner_name}/{self.domain.name}"
        self.perturbations_dir = f"./experiments/run{self.args.run}/{swap_subdir_name}/perturbed_descriptions/"

        os.makedirs(self.problem_dir, exist_ok=True)
        os.makedirs(self.plan_dir, exist_ok=True)
        os.makedirs(self.evaluation_dir, exist_ok=True)

    def run_experiment(self):
        task = self.args.task
        init_nl = self.domain.get_task_init_nl(task)
        goal_nl = self.domain.get_task_goal_nl(task)
        constraints_nl = self.domain.get_task_constraints_nl(task)
        task_nl, _ = self.domain.get_task(task) 
        task_name = self.domain.get_task_name(task)
        task_suffix = self.domain.get_task_suffix(task)

        if(self.args.command == "robustness-experiment"):
            for perturbed_task_name, perturbed_task in self._grab_perturbed_tasks(task_name).items():
                produced_plan: PlannerResult = self.run_planner(perturbed_task["init_nl"], perturbed_task["goal_nl"], perturbed_task["constraints_nl"], perturbed_task_name, task)
                self.run_evaluator(produced_plan, task, perturbed_task_name)
            self._summarize_results()
        else:
            planner_result: PlannerResult = self.run_planner(init_nl, goal_nl, constraints_nl, task_name, task)
            self.run_evaluator(planner_result, task, task_name)

    def _grab_perturbed_tasks(self, task_name):
        perturbed_tasks = {}
        for init_fn in glob.glob(f"{self.perturbations_dir}/{self.domain.name}/{task_name}_*.init.nl"):
            perturbed_task_name = os.path.basename(init_fn).rpartition('.init.nl')[0]
            goal_fn = init_fn.replace(".init.nl", ".goal.nl")
            constraints_fn = init_fn.replace(".init.nl", ".constraints.nl")

            if not os.path.exists(goal_fn):
                raise RuntimeError(f"Goal file not present for perturbed problem {perturbed_task_name} of domain {self.name}")
            elif not os.path.exists(constraints_fn):
                raise RuntimeError(f"Constraints file not present for perturbed problem {perturbed_task_name} of domain {self.name}")
            else:
                with open(init_fn, "r") as f:
                    init_nl = f.read()
                with open(goal_fn, "r") as f:
                    goal_nl = f.read()
                with open(constraints_fn, "r") as f:
                    constraints_nl = f.read()
                perturbed_tasks[perturbed_task_name] = {
                    "init_nl": init_nl,
                    "goal_nl": goal_nl,
                    "constraints_nl": constraints_nl
                }
        return perturbed_tasks

    def run_planner(self, init_nl, goal_nl, constraints_nl, task_name, task):

        # get domain, task and planner information
        context = self.domain.get_context()
        domain_pddl = self.domain.get_domain_pddl()
        domain_nl = self.domain.get_domain_nl()
        planner = available_planners[self.planner_name]

        start_time = time.time()

        planner.set_context(context, self.domain.name, task_name)
        planner.set_response_model_generator(self.response_model_generator_name)
        planner_result = planner.run_planner(init_nl, goal_nl, constraints_nl, domain_nl, domain_pddl)

        end_time = time.time()

        if (planner_result.plan_json is not None):
            plan_json_file_name = f"{self.plan_dir}/{task_name}.json"
            with open(plan_json_file_name, "w") as f:
                f.write(planner_result.plan_json)

        if (planner_result.task_pddl is not None):
            produced_task_pddl_file_name = f"{self.problem_dir}/{task_name}.pddl"
            with open(produced_task_pddl_file_name, "w") as f:
                f.write(planner_result.task_pddl)

        if (planner_result.plan_pddl is not None):
            plan_pddl_file_name = f"{self.plan_dir}/{task_name}.pddl"
            with open(plan_pddl_file_name, "w") as f:
                f.write(planner_result.plan_pddl)

        print(f"[info] task {task} takes {end_time - start_time} sec")
        return planner_result

    def run_evaluator(self, planner_result: PlannerResult, task, task_name):

        domain_pddl = self.domain.get_domain_pddl()
        _, ground_truth_task_pddl = self.domain.get_task(task)

        plan_matcher = available_plan_matchers[self.plan_matcher_name](domain_pddl, ground_truth_task_pddl)
        closest_plan = plan_matcher.plan_closest_match(planner_result)
        closest_plan_pddl_file_name = f"{self.evaluation_dir}/{task_name}.pddl.closest"
        with open(closest_plan_pddl_file_name, "w") as f:
            f.write(closest_plan)

        evaluator = PlanEvaluator(domain_pddl,ground_truth_task_pddl,closest_plan)
        evaluator.try_simulation()

        results = {}

        results["valid"] = evaluator.is_valid()
        if(results["valid"]):
            results["successful"] = evaluator.is_successful()
            results["safe"] = evaluator.is_safe()

        results_file_name = f"{self.evaluation_dir}/{task_name}.results.json"
        with open(results_file_name, 'w') as json_file:
            json.dump(results, json_file, indent=4)

    def produce_perturbations(self, perturbation_recipe: str, 
                                    pct_words_to_swap: float, 
                                    perturbations_number: int = 10,
                                    perturbation_targets: list[Literal["init", "goal", "constraints"]] = ["init", "goal", "constraints"],
                                    jailbreak_text: str = None
                                    ):

        self.perturbations_dir = f"./experiments/run{self.args.run}/{pct_words_to_swap}_swap/perturbed_descriptions/"
        os.makedirs(f"{self.perturbations_dir}/{self.domain.name}", exist_ok=True)

        task_number = self.args.task
        task_name = self.domain.get_task_name(task_number)

        perturbed_tasks = {}
        task_init_nl = self.domain.get_task_init_nl(task_number)
        if "init" in perturbation_targets:
            perturbed_tasks["init"] = text_transformations.produce_perturbations(task_init_nl, perturbation_recipe, pct_words_to_swap, perturbations_number, jailbreak_text)
        else:
            perturbed_tasks["init"] = [task_init_nl] * perturbations_number
        task_goal_nl = self.domain.get_task_goal_nl(task_number)
        if "goal" in perturbation_targets:
            perturbed_tasks["goal"] = text_transformations.produce_perturbations(task_goal_nl, perturbation_recipe, pct_words_to_swap, perturbations_number, jailbreak_text)
        else:
            perturbed_tasks["goal"] = [task_goal_nl] * perturbations_number
        task_constraints_nl = self.domain.get_task_constraints_nl(task_number)
        if "constraints" in perturbation_targets:
            perturbed_tasks["constraints"] = text_transformations.produce_perturbations(task_constraints_nl, perturbation_recipe, pct_words_to_swap, perturbations_number, jailbreak_text)
        else:
            perturbed_tasks["constraints"] = [task_constraints_nl] * perturbations_number

        for component_name in perturbed_tasks:
            for i in range(0, len(perturbed_tasks[component_name])):
                with open(f"{self.perturbations_dir}/{self.domain.name}/{task_name}_{i+1}.{component_name}.nl", "w") as f:
                    f.write(perturbed_tasks[component_name][i])

    def _summarize_results(self):
        # Initialize counters for each category
        total_count = 0
        valid_count = 0
        successful_count = 0
        safe_count = 0
        
        # Traverse through all files in the evaluation dir
        for filename in os.listdir(self.evaluation_dir):
            # Process only files with the .results.json extension
            if filename.endswith(".results.json"):
                file_path = os.path.join(self.evaluation_dir, filename)
                
                total_count += 1

                # Open and load the JSON content
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    
                    # Count the true values for each key
                    if data.get("valid"):
                        valid_count += 1
                    if data.get("successful"):
                        successful_count += 1
                    if data.get("safe"):
                        safe_count += 1
        
        # Prepare the result dictionary
        result = {
            "total": total_count,
            "valid": valid_count,
            "successful": successful_count,
            "safe": safe_count
        }

        # Write the result to a JSON file in the same directory
        output_file_path = os.path.join(self.evaluation_dir, "results_summary.json")
        with open(output_file_path, 'w') as output_file:
            json.dump(result, output_file, indent=4)
        
        print(f"[info] results summary written to {output_file_path}")