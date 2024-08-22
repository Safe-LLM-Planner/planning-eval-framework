import glob
import os
from utils import postprocess

###############################################################################
#
# Define different problem domains
#
###############################################################################

class Domain:
    def __init__(self):
        # every domain should contain the context as in "in-context learning" (ICL)
        # which are the example problem in natural language.
        # For instance, in our case, context is:
        # 1. p_example.nl  (a language description of the problem)
        # 2. p_example.pddl (the ground-truth problem pddl for the problem)
        # 3. p_example.sol  (the ground-truth solution in natural language to the problem)
        self.context = ("p_example.nl", "p_example.pddl", "p_example.sol")
        self.tasks: list[tuple[str,str]] = [] # should be list of tuples like (descritpion file, ground_truth_pddl file)

        self.grab_tasks()

    def grab_tasks(self):
        path = f"./domains/{self.name}"
        nls = []
        for fn in glob.glob(f"{path}/*.nl"):
            fn_ = fn.split("/")[-1]
            if "domain" not in fn_ and "p_example" not in fn_:
                if os.path.exists(fn.replace("nl", "pddl")):
                    nls.append(fn_)
        sorted_nls = sorted(nls)
        self.tasks = [(nl, nl.replace("nl", "pddl")) for nl in sorted_nls]

    def __len__(self):
        return len(self.tasks)
    
    # task number i is a 1-based index

    def get_task_name(self, i) -> str:
        nl, _ = self.tasks[i-1]
        return os.path.splitext(nl)[0]

    def get_task_suffix(self, i) -> str:
        nl, pddl = self.tasks[i-1]
        return f"{self.name}/{pddl}"

    def get_task_file(self, i) -> str:
        nl, pddl = self.tasks[i-1]
        return f"./domains/{self.name}/{nl}", f"./domains/{self.name}/{pddl}"

    def get_task(self, i):
        nl_f, pddl_f = self.get_task_file(i)
        with open(nl_f, 'r') as f:
            nl = f.read()
        with open(pddl_f, 'r') as f:
            pddl = f.read()
        return postprocess(nl), postprocess(pddl)

    def get_context(self):
        nl_f   = f"./domains/{self.name}/{self.context[0]}"
        pddl_f = f"./domains/{self.name}/{self.context[1]}"
        sol_f  = f"./domains/{self.name}/{self.context[2]}"
        with open(nl_f, 'r') as f:
            nl   = f.read()
        with open(pddl_f, 'r') as f:
            pddl = f.read()
        with open(sol_f, 'r') as f:
            sol  = f.read()
        return postprocess(nl), postprocess(pddl), postprocess(sol)

    def get_domain_pddl(self):
        domain_pddl_f = self.get_domain_pddl_file()
        with open(domain_pddl_f, 'r') as f:
            domain_pddl = f.read()
        return postprocess(domain_pddl)

    def get_domain_pddl_file(self):
        domain_pddl_f = f"./domains/{self.name}/domain.pddl"
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
        domain_nl_f = f"./domains/{self.name}/domain.nl"
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
    "barman": Barman(),
    "blocksworld": Blocksworld(),
    "floortile": Floortile(),
    "grippers": Grippers(),
    "storage": Storage(),
    "termes": Termes(),
    "tyreworld": Tyreworld(),
    "manipulation": Manipulation()
}