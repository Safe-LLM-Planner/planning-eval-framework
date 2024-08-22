import glob
import json
import os
import textattack
import time

from domains import Domain
from planners import available_planners, PlannerResult
from plan_evaluator import PlanEvaluator, PlanGreedyActionMatcher, PlanIndividualObjectMatcher

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

    def run_experiment(self, method: str):
        
        task = self.args.task
        task_nl, _ = self.domain.get_task(task) 
        task_name = self.domain.get_task_name(task)
        task_suffix = self.domain.get_task_suffix(task)

        if(self.args.command == "robustness-experiment"):
            perturbations_folder = f"./experiments/run{self.args.run}/perturbed_descriptions/"
            for fn in glob.glob(f"{perturbations_folder}/{self.domain.name}/{task_name}_*"):
                perturbed_task_name = os.path.splitext(os.path.basename(fn))[0]
                with open(fn, "r") as f:
                    perturbed_task_nl = f.read()
                    produced_plan = self.run_planner(method, perturbed_task_nl, perturbed_task_name, task)
                    self.run_evaluator(produced_plan, task, perturbed_task_name, method)
        else:
            planner_result: PlannerResult = self.run_planner(method, task_nl, task_name, task)
            self.run_evaluator(planner_result, task, task_name, method)

    def run_planner(self, method, task_nl, task_name, task):

        # create result folders
        problem_folder = f"./experiments/run{self.args.run}/problems/{method}/{self.domain.name}"
        plan_folder    = f"./experiments/run{self.args.run}/plans/{method}/{self.domain.name}"
        os.makedirs(problem_folder, exist_ok=True)
        os.makedirs(plan_folder, exist_ok=True)

        # get domain, task and planner information
        context = self.domain.get_context()
        domain_pddl = self.domain.get_domain_pddl()
        domain_nl = self.domain.get_domain_nl()
        planner = available_planners[method]

        start_time = time.time()

        planner.set_context(context)
        planner_result = planner.run_planner(task_nl, domain_nl, domain_pddl)

        end_time = time.time()

        if (planner_result.plan_json):
            plan_json_file_name = f"{plan_folder}/{task_name}.json"
            with open(plan_json_file_name, "w") as f:
                f.write(planner_result.plan_json)


        if (planner_result.task_pddl):
            produced_task_pddl_file_name = f"{problem_folder}/{task_name}.pddl"
            with open(produced_task_pddl_file_name, "w") as f:
                f.write(planner_result.task_pddl)

        if (planner_result.plan_pddl):
            plan_pddl_file_name = f"{plan_folder}/{task_name}.pddl"
            with open(plan_pddl_file_name, "w") as f:
                f.write(planner_result.plan_pddl)

        print(f"[info] task {task} takes {end_time - start_time} sec")
        return planner_result

    def run_evaluator(self, planner_result: PlannerResult, task, task_name, method):

        domain_pddl = self.domain.get_domain_pddl()
        _, ground_truth_task_pddl = self.domain.get_task(task)

        evaluation_folder = f"./experiments/run{self.args.run}/evaluation/{method}/{self.domain.name}"
        os.makedirs(evaluation_folder, exist_ok=True)

        plan_matcher = PlanGreedyActionMatcher(domain_pddl, ground_truth_task_pddl)
        closest_plan = plan_matcher.plan_closest_match(planner_result)
        closest_plan_pddl_file_name = f"{evaluation_folder}/{task_name}.pddl.closest"
        with open(closest_plan_pddl_file_name, "w") as f:
            f.write(closest_plan)

        evaluator = PlanEvaluator(domain_pddl,ground_truth_task_pddl,closest_plan)
        evaluator.try_simulation()

        results = {}

        results["valid"] = evaluator.is_valid()
        if(results["valid"]):
            results["successful"] = evaluator.is_successful()
            results["safe"] = evaluator.is_safe()

        results_file_name = f"{evaluation_folder}/{task_name}.results.json"
        with open(results_file_name, 'w') as json_file:
            json.dump(results, json_file, indent=4)

    def produce_perturbations(self):

        task = self.args.task
        task_nl, _ = self.domain.get_task(task)
        task_name = self.domain.get_task_name(task)

        augmenter = available_textattack_perturbations[self.args.perturbation_recipe](
                                                        pct_words_to_swap=self.args.pct_words_to_swap, 
                                                        transformations_per_example=10)
        perturbed_task_nl_list = augmenter.augment(task_nl)

        # create the tmp / result folders
        perturbations_folder = f"./experiments/run{self.args.run}/perturbed_descriptions/"
        os.makedirs(f"{perturbations_folder}/{self.domain.name}", exist_ok=True)
        
        for i in range(0, len(perturbed_task_nl_list)):
            with open(f"{perturbations_folder}/{self.domain.name}/{task_name}_{i+1}.nl", "w") as f:
                f.write(perturbed_task_nl_list[i])

def print_all_prompts():
    for domain_name in DOMAINS:
        domain = eval(domain_name.capitalize())()
        context = domain.get_context()
        domain_pddl = domain.get_domain_pddl()
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