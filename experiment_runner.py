import glob
import os
import textattack
import time

from planners import available_planners
from plan_evaluator import PlanEvaluator, PlanGreedyActionMatcher, PlanIndividualObjectMatcher

available_textattack_perturbations = {
    "wordnet": textattack.augmentation.recipes.WordNetAugmenter,
    "charswap": textattack.augmentation.recipes.CharSwapAugmenter,
    "back_trans": textattack.augmentation.recipes.BackTranslationAugmenter,
    "back_transcription": textattack.augmentation.recipes.BackTranscriptionAugmenter
}

class ExperimentRunner():
    def __init__(self, args, domain):
        self.args = args
        self.domain = domain

    def run_experiment(self, method):
        
        task = self.args.task
        task_nl, _ = self.domain.get_task(task) 
        task_suffix = self.domain.get_task_suffix(task)

        # create result folders
        problem_folder = f"./experiments/run{self.args.run}/problems/{method}/{self.domain.name}"
        plan_folder    = f"./experiments/run{self.args.run}/results/{method}/{self.domain.name}"
        os.makedirs(problem_folder, exist_ok=True)
        os.makedirs(plan_folder, exist_ok=True)

        if(self.args.command == "robustness-experiment"):
            perturbations_folder = f"./experiments/run{self.args.run}/perturbed_descriptions/"
            task_suffix = os.path.splitext(task_suffix)[0]
            for fn in glob.glob(f"{perturbations_folder}/{task_suffix}_*"):
                perturbed_task_suffix = f"{self.domain.name}/{os.path.splitext(os.path.basename(fn))[0]}.pddl"
                with open(fn, "r") as f:
                    perturbed_task_nl = f.read()
                    self.run_planner(method, perturbed_task_nl, perturbed_task_suffix, task)
        else:
            self.run_planner(method, task_nl, task_suffix, task)

    def run_planner(self, method, task_nl, task_suffix, task):

        context = self.domain.get_context()
        domain_pddl = self.domain.get_domain_pddl()
        domain_nl = self.domain.get_domain_nl()
        _, ground_truth_task_pddl = self.domain.get_task(task) 
        planner = available_planners[method]

        start_time = time.time()

        planner.set_context(context)
        produced_plan, task_pddl = planner.run_planner(task_nl, domain_nl, domain_pddl)

        end_time = time.time()

        if (task_pddl):
            task_pddl_file_name = f"./experiments/run{self.args.run}/problems/{method}/{task_suffix}"
            with open(task_pddl_file_name, "w") as f:
                f.write(task_pddl)

        plan_pddl_file_name = f"./experiments/run{self.args.run}/results/{method}/{task_suffix}"
        with open(plan_pddl_file_name, "w") as f:
            f.write(produced_plan)

        print(f"[info] task {task} takes {end_time - start_time} sec")

        plan_matcher = PlanGreedyActionMatcher(domain_pddl, ground_truth_task_pddl)
        closest_plan = plan_matcher.plan_closest_match(produced_plan)
        closest_plan_plan_pddl_file_name = f"./experiments/run{self.args.run}/results/{method}/{task_suffix}.closest"
        with open(closest_plan_plan_pddl_file_name, "w") as f:
            f.write(closest_plan)
        evaluator = PlanEvaluator(domain_pddl,ground_truth_task_pddl,closest_plan)
        evaluator.try_simulation()
        is_valid = evaluator.is_valid()
        is_successful = None
        is_safe = None
        if(is_valid):
            is_successful = evaluator.is_successful()
            is_safe = evaluator.is_safe()

        print(f"valid: {is_valid}, successful: {is_successful}, safe: {is_safe}")

    def produce_perturbations(self):

        task = self.args.task
        task_nl, _ = self.domain.get_task(task)
        task_suffix = self.domain.get_task_suffix(task)
        task_suffix = os.path.splitext(task_suffix)[0]

        augmenter = available_textattack_perturbations[self.args.perturbation_recipe](
                                                        pct_words_to_swap=self.args.pct_words_to_swap, 
                                                        transformations_per_example=10)
        perturbed_task_nl_list = augmenter.augment(task_nl)

        # create the tmp / result folders
        perturbations_folder = f"./experiments/run{self.args.run}/perturbed_descriptions/"

        if not os.path.exists(perturbations_folder):
            os.makedirs(f"{perturbations_folder}/{self.domain.name}", exist_ok=True)
        
        for i in range(0, len(perturbed_task_nl_list)):

            with open(f"{perturbations_folder}/{task_suffix}_{i+1}.nl", "w") as f:
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