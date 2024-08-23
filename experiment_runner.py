import glob
import json
import os
import textattack
import time

from domains import Domain
from planners import available_planners, PlannerResult
from plan_evaluator import PlanEvaluator, available_plan_matchers

available_textattack_perturbations = {
    "wordnet": textattack.augmentation.recipes.WordNetAugmenter,
    "charswap": textattack.augmentation.recipes.CharSwapAugmenter,
    "back_trans": textattack.augmentation.recipes.BackTranslationAugmenter,
    "back_transcription": textattack.augmentation.recipes.BackTranscriptionAugmenter
}

class ExperimentRunner():
    def __init__(self, args, domain: Domain):
        self.args = args
        self.domain = domain

    def set_experiment(self, planner_name: str, 
                             response_model_generator_name: str, 
                             plan_matcher_name: str,
                             pct_words_to_swap: float):
        self.planner_name = planner_name
        self.response_model_generator_name = response_model_generator_name
        self.plan_matcher_name = plan_matcher_name

        # set experiment folders
        self.problem_folder = f"./experiments/run{self.args.run}/{pct_words_to_swap}_swap/problems/{self.planner_name}/{self.domain.name}"
        self.plan_folder    = f"./experiments/run{self.args.run}/{pct_words_to_swap}_swap/plans/{self.planner_name}/{self.domain.name}"
        self.evaluation_folder = f"./experiments/run{self.args.run}/{pct_words_to_swap}_swap/evaluation/{self.planner_name}/{self.domain.name}"
        self.perturbations_folder = f"./experiments/run{self.args.run}/{pct_words_to_swap}_swap/perturbed_descriptions/"

        os.makedirs(self.problem_folder, exist_ok=True)
        os.makedirs(self.plan_folder, exist_ok=True)
        os.makedirs(self.evaluation_folder, exist_ok=True)

    def run_experiment(self):
        task = self.args.task
        task_nl, _ = self.domain.get_task(task) 
        task_name = self.domain.get_task_name(task)
        task_suffix = self.domain.get_task_suffix(task)

        if(self.args.command == "robustness-experiment"):
            for fn in glob.glob(f"{self.perturbations_folder}/{self.domain.name}/{task_name}_*"):
                perturbed_task_name = os.path.splitext(os.path.basename(fn))[0]
                with open(fn, "r") as f:
                    perturbed_task_nl = f.read()
                    produced_plan = self.run_planner(perturbed_task_nl, perturbed_task_name, task)
                    self.run_evaluator(produced_plan, task, perturbed_task_name)
            self._summarize_results()
        else:
            planner_result: PlannerResult = self.run_planner(task_nl, task_name, task)
            self.run_evaluator(planner_result, task, task_name)

    def run_planner(self, task_nl, task_name, task):

        # get domain, task and planner information
        context = self.domain.get_context()
        domain_pddl = self.domain.get_domain_pddl()
        domain_nl = self.domain.get_domain_nl()
        planner = available_planners[self.planner_name]

        start_time = time.time()

        planner.set_context(context)
        planner.set_response_model_generator(self.response_model_generator_name)
        planner_result = planner.run_planner(task_nl, domain_nl, domain_pddl)

        end_time = time.time()

        if (planner_result.plan_json):
            plan_json_file_name = f"{self.plan_folder}/{task_name}.json"
            with open(plan_json_file_name, "w") as f:
                f.write(planner_result.plan_json)


        if (planner_result.task_pddl):
            produced_task_pddl_file_name = f"{self.problem_folder}/{task_name}.pddl"
            with open(produced_task_pddl_file_name, "w") as f:
                f.write(planner_result.task_pddl)

        if (planner_result.plan_pddl):
            plan_pddl_file_name = f"{self.plan_folder}/{task_name}.pddl"
            with open(plan_pddl_file_name, "w") as f:
                f.write(planner_result.plan_pddl)

        print(f"[info] task {task} takes {end_time - start_time} sec")
        return planner_result

    def run_evaluator(self, planner_result: PlannerResult, task, task_name):

        domain_pddl = self.domain.get_domain_pddl()
        _, ground_truth_task_pddl = self.domain.get_task(task)

        plan_matcher = available_plan_matchers[self.plan_matcher_name](domain_pddl, ground_truth_task_pddl)
        closest_plan = plan_matcher.plan_closest_match(planner_result)
        closest_plan_pddl_file_name = f"{self.evaluation_folder}/{task_name}.pddl.closest"
        with open(closest_plan_pddl_file_name, "w") as f:
            f.write(closest_plan)

        evaluator = PlanEvaluator(domain_pddl,ground_truth_task_pddl,closest_plan)
        evaluator.try_simulation()

        results = {}

        results["valid"] = evaluator.is_valid()
        if(results["valid"]):
            results["successful"] = evaluator.is_successful()
            results["safe"] = evaluator.is_safe()

        results_file_name = f"{self.evaluation_folder}/{task_name}.results.json"
        with open(results_file_name, 'w') as json_file:
            json.dump(results, json_file, indent=4)

    def produce_perturbations(self, perturbation_recipe: str, pct_words_to_swap: float, transformations_per_example: int = 10):

        self.perturbations_folder = f"./experiments/run{self.args.run}/{pct_words_to_swap}_swap/perturbed_descriptions/"
        os.makedirs(f"{self.perturbations_folder}/{self.domain.name}", exist_ok=True)

        task = self.args.task
        task_nl, _ = self.domain.get_task(task)
        task_name = self.domain.get_task_name(task)

        augmenter = available_textattack_perturbations[perturbation_recipe](
                                                        pct_words_to_swap=pct_words_to_swap, 
                                                        transformations_per_example=transformations_per_example)
        perturbed_task_nl_list = augmenter.augment(task_nl)

        
        for i in range(0, len(perturbed_task_nl_list)):
            with open(f"{self.perturbations_folder}/{self.domain.name}/{task_name}_{i+1}.nl", "w") as f:
                f.write(perturbed_task_nl_list[i])

    def _summarize_results(self):
        # Initialize counters for each category
        valid_count = 0
        successful_count = 0
        safe_count = 0
        
        # Traverse through all files in the evaluation folder
        for filename in os.listdir(self.evaluation_folder):
            # Process only files with the .results.json extension
            if filename.endswith(".results.json"):
                file_path = os.path.join(self.evaluation_folder, filename)
                
                # Open and load the JSON content
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    
                    # Count the true values for each key
                    if data.get("valid") is True:
                        valid_count += 1
                    if data.get("successful") is True:
                        successful_count += 1
                    if data.get("safe") is True:
                        safe_count += 1
        
        # Prepare the result dictionary
        result = {
            "valid": valid_count,
            "successful": successful_count,
            "safe": safe_count
        }

        # Write the result to a JSON file in the same directory
        output_file_path = os.path.join(self.evaluation_folder, "results_summary.json")
        with open(output_file_path, 'w') as output_file:
            json.dump(result, output_file, indent=4)
        
        print(f"[info] results summary written to {output_file_path}")