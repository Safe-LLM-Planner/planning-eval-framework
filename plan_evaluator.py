import json

from juliacall import Main as jl
from planners import PlannerResult
from sentence_transformers import SentenceTransformer
from utils import openai_client

word_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

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

class PlanMatcher:
    def __init__(self, domain_pddl, problem_pddl):
        self.domain = jl.PDDL.parse_domain(domain_pddl)
        self.problem = jl.PDDL.parse_problem(problem_pddl)

    def plan_closest_match(self, planner_result: PlannerResult):
        raise NotImplementedError

    @staticmethod
    def _build_action(name, args):
        return jl.Compound(jl.Symbol(name), [jl.Const(jl.Symbol(a)) for a in args])

    @staticmethod
    def _compute_similarity(text1, text2):
        embedding1 = word_embedding_model.encode(text1)
        embedding2 = word_embedding_model.encode(text2)
        similarity = word_embedding_model.similarity(embedding1, embedding2).squeeze()
        # print(f"{text1}, {text2}: {similarity}")
        return similarity

class PlanGreedyActionMatcher(PlanMatcher):
    def plan_closest_match(self, planner_result: PlannerResult):
        if(planner_result.plan_pddl):
            return self._plan_closest_match_pddl(planner_result.plan_pddl)
        elif(planner_result.plan_json):
            return self._plan_closest_match_json(planner_result.plan_json)
        else:
            raise ValueError("No plan was given as input")
    
    def _plan_closest_match_json(self, plan_json):
        plan_dict = json.loads(plan_json)
        actions_texts = [ " ".join(step.values()) for step in plan_dict['steps']]
        
        current_state = jl.PDDL.initstate(self.domain, self.problem)
        acts_closest_match = []
        for act_text in actions_texts:
            available_actions = jl.PDDL.available(self.domain, current_state)
            closest_match = self._action_closest_match(act_text, available_actions)
            if(closest_match):
                current_state = jl.PDDL.execute(self.domain, current_state, closest_match)
                acts_closest_match.append(closest_match)
            else:
                break
            
        res_pddl_text = "\n".join([jl.PDDL.write_pddl(act) for act in acts_closest_match])
        
        return res_pddl_text

    def _plan_closest_match_pddl(self, plan_pddl):
        actions = [jl.PDDL.Parser.parse_pddl(line) 
                    for line in plan_pddl.splitlines()
                    if line.strip()[0] != ";"]
        
        current_state = jl.PDDL.initstate(self.domain, self.problem)
        acts_closest_match = []
        for act in actions:
            if jl.PDDL.available(self.domain, current_state, act):
                closest_match = act
            else:
                available_actions = jl.PDDL.available(self.domain, current_state)
                closest_match = self._action_closest_match(self._action_text(act), available_actions)
            if(closest_match):
                current_state = jl.PDDL.execute(self.domain, current_state, closest_match)
                acts_closest_match.append(closest_match)
            else:
                break
            
        res_pddl_text = "\n".join([jl.PDDL.write_pddl(act) for act in acts_closest_match])
        
        return res_pddl_text

    def _action_closest_match(self, action_text, available_actions):
        # action_text = self._action_text(action)
        current_similarity = -1
        res = None
        for act in available_actions:
            act_text = self._action_text(act)
            similarity = self._compute_similarity(action_text, act_text)
            if similarity > current_similarity:
                res = act
                current_similarity = similarity
        return res

    def _action_text(self, action):
        return " ".join([str(action.name)] + [str(a) for a in action.args])

class PlanIndividualObjectMatcher(PlanMatcher):
    def __init__(self, domain_pddl, problem_pddl):
        super().__init__(domain_pddl, problem_pddl)
        self.objects = jl.PDDL.get_objtypes(self.problem)
        
    def plan_closest_match(self, planner_result: PlannerResult):

        if not planner_result.plan_pddl:
            raise ValueError("This plan matcher requires the planner result to be in pddl format")

        actions = [jl.PDDL.Parser.parse_pddl(line) 
                    for line in planner_result.plan_pddl.splitlines()
                    if line.strip()[0] != ";"]
        acts_closest_match = [self._action_closest_match(act) for act in actions]
        res_pddl_text = "\n".join([jl.PDDL.write_pddl(act) for act in acts_closest_match])
        
        return res_pddl_text

    def _action_closest_match(self, action):
        new_args = [self._object_closest_match(a) for a in action.args]
        return self._build_action(action.name, new_args)

    def _object_closest_match(self, object):
        current_similarity = -1
        res = None
        for obj in self.objects:
            similarity = self._compute_similarity(str(object), str(obj))
            if similarity > current_similarity:
                res = obj
                current_similarity = similarity
        return res