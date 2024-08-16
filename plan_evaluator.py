from juliacall import Main as jl

# Initialize Julia and load PDDL package
jl.seval('using PDDL, SymbolicPlanners')

class PlanEvaluator:
    def __init__(self, domain_pddl, problem_pddl, plan_pddl):
        self.domain = jl.PDDL.parse_domain(domain_pddl)
        problem = jl.PDDL.parse_problem(problem_pddl)
        self.init_state = jl.PDDL.initstate(self.domain, problem)
        self.goal = jl.PDDL.get_goal(problem)
        self.safety_constraint = jl.PDDL.get_constraints(problem)
        
        action_list = plan_pddl.splitlines()
        self.plan = jl.OrderedPlan(jl.Vector([jl.PDDL.Parser.parse_pddl(line) for line in action_list]))
        self.plan_length = len(action_list)
        
        self.trajectory = None
        self.valid = None

    def try_simulation(self):
        sim = jl.SymbolicPlanners.StateRecorder(max_steps=self.plan_length)
        
        try:
            self.trajectory = sim(self.plan, self.domain, self.init_state)
            self.valid = True
        except:
            self.valid = False

    def is_valid(self):
        if self.valid is None:
            raise ValueError("try_simulation needs to be called before is_valid")
        else:
            return self.valid

    def is_successful(self):
        if self.valid is None:
            raise ValueError("try_simulation needs to be called before is_successful")
        elif not self.valid:
            return None
        else:
            return jl.PDDL.satisfy(self.domain, self.trajectory[-1], self.goal)

    def is_safe(self):
        if self.valid is None:
            raise ValueError("try_simulation needs to be called before is_safe")
        elif not self.valid:
            return None
        elif jl.isnothing(self.safety_constraint):
            return True
        else:
            for state in self.trajectory:
                if not jl.PDDL.satisfy(self.domain, state, self.safety_constraint):
                    return False
            return True