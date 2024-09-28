import glob
import os
from utils import postprocess

###############################################################################
#
# Define different problem domains
#
###############################################################################

class Task:
    def __init__(self, name: str):
        self.name = name

    def get_init_filename(self):
        return f"{self.name}.init.nl"

    def get_goal_filename(self):
        return f"{self.name}.goal.nl"

    def get_constraints_filename(self):
        return f"{self.name}.constraints.nl"

    def get_ground_truth_pddl_filename(self):
        return f"{self.name}.pddl"

class Context(Task):
    def get_ground_truth_plan_nl_file(self):
        return f"{self.name}.sol"

    def get_ground_truth_pddl_components_f(self):
        return f"{self.name}.init.pddl", f"{self.name}.goal.pddl", f"{self.name}.constraints.pddl"

class Domain:
    def __init__(self):
        # every domain should contain the context as in "in-context learning" (ICL)
        # which are the example problem in natural language.
        # For instance, in our case, context is:
        # - p_example.init.nl  (a language description of the problem's initial state)
        # - p_example.goal.nl  (a language description of the problem's goal)
        # - p_example.constraints.nl  (a language description of the problem's constraints)
        # - p_example.pddl (the ground-truth problem pddl for the problem)
        # - p_example.sol  (the ground-truth solution in natural language to the problem)
        self.context = Context("p_example")
        self.tasks: list[Task] = []
        self.domain_dir = f"./domains/{self.name}/"

        self.grab_tasks()

    def grab_tasks(self):
        path = f"./domains/{self.name}"
        problem_name_list = []
        for fn in glob.glob(f"{path}/*.init.nl"):
            file_base_name = os.path.basename(fn)
            problem_name = file_base_name.rpartition('.init.nl')[0]
            if "domain" not in problem_name and "p_example" not in problem_name:
                if not os.path.exists(fn.replace(".init.nl", ".goal.nl")):
                    raise RuntimeError(f"Goal file not present for problem {problem_name} of domain {self.name}")
                elif not os.path.exists(fn.replace(".init.nl", ".constraints.nl")):
                    raise RuntimeError(f"Constraints file not present for problem {problem_name} of domain {self.name}")
                elif not os.path.exists(fn.replace(".init.nl", ".pddl")):
                    raise RuntimeError(f"Ground truth PDDL file not present for problem {problem_name} of domain {self.name}")
                else:
                    problem_name_list.append(problem_name)
        problem_name_list = sorted(problem_name_list)
        self.tasks = [Task(p_name) for p_name in problem_name_list]

    def __len__(self):
        return len(self.tasks)
    
    # task number i is a 1-based index

    def get_task_name(self, i) -> str:
        return self.tasks[i-1].name

    def get_task_suffix(self, i) -> str:
        pddl = self.tasks[i-1].get_ground_truth_pddl_filename()
        return f"{self.name}/{pddl}"

    def get_task_init_nl(self, i):
        init_nl_f = self.tasks[i-1].get_init_filename()
        with open(os.path.join(self.domain_dir, init_nl_f), 'r') as f:
            init_nl = f.read()
        
        return postprocess(init_nl)

    def get_task_goal_nl(self, i):
        goal_nl_f = self.tasks[i-1].get_goal_filename()
        with open(os.path.join(self.domain_dir, goal_nl_f), 'r') as f:
            goal_nl = f.read()
        
        return postprocess(goal_nl)

    def get_task_constraints_nl(self, i):
        constraints_nl_f = self.tasks[i-1].get_constraints_filename()
        with open(os.path.join(self.domain_dir, constraints_nl_f), 'r') as f:
            constraints_nl = f.read()
        
        return postprocess(constraints_nl)

    def get_task_pddl(self, i):
        pddl_f = self.tasks[i-1].get_ground_truth_pddl_filename()
        with open(os.path.join(self.domain_dir, pddl_f), 'r') as f:
            pddl = f.read()
        
        return postprocess(pddl)

    def get_task(self, i):
        init_nl = self.get_task_init_nl(i)
        goal_nl = self.get_task_goal_nl(i)
        constraints_nl = self.get_task_constraints_nl(i)
        pddl = self.get_task_pddl(i)
        nl = "\n".join([init_nl, goal_nl, constraints_nl])

        return nl, pddl

    def get_context(self):
        init_nl_f = self.context.get_init_filename()
        goal_nl_f = self.context.get_goal_filename()
        constraints_nl_f = self.context.get_constraints_filename()
        pddl_f = self.context.get_ground_truth_pddl_filename()
        init_pddl_f, goal_pddl_f, constraints_pddl_f = self.context.get_ground_truth_pddl_components_f()
        sol_f = self.context.get_ground_truth_plan_nl_file()

        with open(os.path.join(self.domain_dir, init_nl_f), 'r') as f:
            init_nl = f.read()
        with open(os.path.join(self.domain_dir, goal_nl_f), 'r') as f:
            goal_nl = f.read()
        with open(os.path.join(self.domain_dir, constraints_nl_f), 'r') as f:
            constraints_nl = f.read()
        with open(os.path.join(self.domain_dir, init_pddl_f), 'r') as f:
            init_pddl = f.read()
        with open(os.path.join(self.domain_dir, goal_pddl_f), 'r') as f:
            goal_pddl = f.read()
        with open(os.path.join(self.domain_dir, constraints_pddl_f), 'r') as f:
            constraints_pddl = f.read()
        with open(os.path.join(self.domain_dir, sol_f), 'r') as f:
            sol = f.read()
        res = {
            "init_nl": postprocess(init_nl),
            "goal_nl": postprocess(goal_nl),
            "constraints_nl": postprocess(constraints_nl),
            "init_pddl": postprocess(init_pddl),
            "goal_pddl": postprocess(goal_pddl),
            "constraints_pddl": postprocess(constraints_pddl),
            "sol": postprocess(sol)
        }
        return res

    def get_domain_pddl(self):
        domain_pddl_f = self.get_domain_pddl_file()
        with open(domain_pddl_f, 'r') as f:
            domain_pddl = f.read()
        return postprocess(domain_pddl)

    def get_domain_pddl_file(self):
        domain_pddl_f = f"{self.domain_dir}/domain.pddl"
        return domain_pddl_f

    def get_domain_nl(self):
        domain_nl_f = self.get_domain_nl_file()
        try:
            with open(domain_nl_f, 'r') as f:
                domain_nl = f.read()
        except:
            domain_nl = "Nothing"
        return postprocess(domain_nl)

    def get_domain_nl_file(self):
        domain_nl_f = f"{self.domain_dir}/domain.nl"
        return domain_nl_f


class Barman(Domain):
    name = "barman" # this should match the directory name

class Floortile(Domain):
    name = "floortile" # this should match the directory name

class Termes(Domain):
    name = "termes" # this should match the directory name

class Tyreworld(Domain):
    name = "tyreworld" # this should match the directory name

class Grippers(Domain):
    name = "grippers" # this should match the directory name

class Storage(Domain):
    name = "storage" # this should match the directory name

class Blocksworld(Domain):
    name = "blocksworld" # this should match the directory name

class Manipulation(Domain):
    name = "manipulation" # this should match the directory name

available_domains = {
    # "barman": Barman(),
    # "blocksworld": Blocksworld(),
    # "floortile": Floortile(),
    # "grippers": Grippers(),
    # "storage": Storage(),
    # "termes": Termes(),
    # "tyreworld": Tyreworld(),
    "manipulation": Manipulation()
}