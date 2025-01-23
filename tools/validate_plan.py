import argparse
from juliacall import Main as jl
from plan_evaluator import PlanEvaluator

# Initialize Julia and load PDDL package
jl.seval('using PDDL, SymbolicPlanners')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("domain_file", type=str, help="Path to the domain PDDL file.")
    parser.add_argument("problem_file", type=str, help="Path to the problem PDDL file.")
    parser.add_argument("plan_file", type=str, help="Path to the plan PDDL file.")
    
    args = parser.parse_args()

    # Read the PDDL files
    with open(args.domain_file, 'r') as domain_file:
        domain_pddl_text = domain_file.read()
    
    with open(args.problem_file, 'r') as problem_file:
        problem_pddl_text = problem_file.read()
    
    with open(args.plan_file, 'r') as plan_file:
        plan_pddl_text = plan_file.read()

    # Run the symbolic planner
    planner_evaluator = PlanEvaluator(domain_pddl_text, problem_pddl_text, plan_pddl_text)
    planner_evaluator.try_simulation()
    results = {}
    results["valid"] = planner_evaluator.is_valid()
    if(results["valid"]):
        results["successful"] = planner_evaluator.is_successful()
        results["safe"] = planner_evaluator.is_safe()

    # Print the solution
    print(results)

if __name__ == "__main__":
    main()
