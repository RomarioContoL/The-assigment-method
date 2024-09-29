import pyomo
import highspy
from pyomo.environ import *
from pyomo.opt import *
import itertools
import numpy as np



def create_design(levels1=[-1, 1],  # First factor levels
                  levels2=[-1, 0, 1],  # Second factor levels
                  n_factors=2,  # Number of factors
                  beta=5, 
                  solver='appsi_highs'):
    """
    Creates a design optimization model using Pyomo to minimize level changes
    in a factorial design while respecting constraints on the factors and
    distances between runs.

    Parameters:
    levels1 (list): Levels for the first factor.
    levels2 (list): Levels for the second factor.
    n_factors (int): Number of factors in the design.
    beta (int): Maximum allowable number of level changes.
    solver (str): Optimization solver to use

    Returns:
    dict: A dictionary with the solution status, order, biases, and total distance.
    """

    # Define the factors as a set of indices
    factors = {i for i in range(n_factors)}

    # Generate run orders as Cartesian product of the factor levels
    runs = itertools.product(levels1, levels2)
    runs = [i for i in runs]  # Convert iterator to list
    runs = {j + 1: runs[j] for j in range(len(runs))}  # Dictionary of run orders

    # Define positions corresponding to the run orders
    positions = {i + 1 for i in range(len(runs))}

    # Create the response dictionary and calculate distances between runs
    response = {i: i for i in positions}
    distances = {}
    for k1, v1 in runs.items():
        for k2, v2 in runs.items():
            distances[(k1, k2)] = [a == b for a, b in zip(v1, v2)].count(False)  # Count level changes
    arcos = set(distances.keys())  # Set of run pairs

    # Create the optimization model
    model = ConcreteModel(name="AssignmentPriority")

    # Define the sets used in the model
    model.POSITIONS = Set(ordered=False, initialize=positions)
    model.RUNS = Set(ordered=False, initialize=positions)
    model.FACTORS = Set(ordered=False, initialize=factors)
    model.ARCOS = Set(ordered=False, initialize=arcos)

    # Define parameters for the model
    model.runs = Param(model.RUNS, initialize=runs, within=Any)
    model.response = Param(model.POSITIONS, initialize=response, within=Any)
    model.distance = Param(model.ARCOS, initialize=distances, within=Any)
    model.beta = Param(initialize=beta, within=Any)

    # Define decision variables
    model.x = Var(model.POSITIONS, model.RUNS, domain=Binary)  # Binary assignment of runs to positions
    model.s = Var(model.FACTORS, domain=NonNegativeReals)  # Absolute values for each factor
    model.s_max = Var(domain=NonNegativeReals)  # Max absolute value
    model.p = Var(model.POSITIONS, model.RUNS, model.RUNS, domain=Binary)  # Binary distance variables

    # Objective function to minimize the maximum value of `s`
    def obj_rule(model):
        return model.s_max
    model.objective = Objective(sense=minimize, rule=obj_rule)

    # Constraints to linearize maximum absolute values
    def s_max_c(model, k):
        return model.s[k] <= model.s_max
    model.s_max_c = Constraint(model.FACTORS, rule=s_max_c)

    # Linearizing absolute value constraints
    def linear1(model, k):
        return sum(model.response[i] * model.runs[j][k] * model.x[i, j] for i in model.POSITIONS for j in model.RUNS) <= model.s[k]
    model.linear1 = Constraint(model.FACTORS, rule=linear1)

    def linear2(model, k):
        return -sum(model.response[i] * model.runs[j][k] * model.x[i, j] for i in model.POSITIONS for j in model.RUNS) <= model.s[k]
    model.linear2 = Constraint(model.FACTORS, rule=linear2)

    # Constraints to ensure one run per position
    def run_1(model, i):
        return sum(model.x[i, j] for j in model.RUNS) == 1
    model.run_1 = Constraint(model.POSITIONS, rule=run_1)

    # Constraints to ensure one position per run
    def position_1(model, j):
        return sum(model.x[i, j] for i in model.POSITIONS) == 1
    model.position_1 = Constraint(model.RUNS, rule=position_1)

    # Constraint to limit the total distance (level changes)
    def distance_c(model):
        return sum(model.distance[(k, j)] * model.p[i, k, j] for i in model.POSITIONS for k in model.RUNS for j in model.RUNS) <= model.beta
    model.distance_c = Constraint(rule=distance_c)

    # Linearizing product terms for distance constraints
    def product1(model, i, k, j):
        return model.p[i, k, j] <= model.x[i, k]
    model.product1 = Constraint(model.POSITIONS, model.RUNS, model.RUNS, rule=product1)

    def product2(model, i, k, j):
        if i < max(positions):
            return model.p[i, k, j] <= model.x[i + 1, j]
        else:
            return Constraint.Skip
    model.product2 = Constraint(model.POSITIONS, model.RUNS, model.RUNS, rule=product2)

    def product3(model, i, k, j):
        if i < max(positions):
            return model.p[i, k, j] >= model.x[i, k] + model.x[i + 1, j] - 1
        else:
            return Constraint.Skip
    model.product3 = Constraint(model.POSITIONS, model.RUNS, model.RUNS, rule=product3)

    
    # Solve the model using appsi_highs solver 
    # Solve the model using Gurobi solver (requires license)
    solver = SolverFactory(solver)
    results = solver.solve(model, tee=True)
    term_cond = results.solver.termination_condition
    print("Termination condition = {}".format(term_cond))
    
    results_dict = {}
    # Process results
    if term_cond == TerminationCondition.optimal:
        results_dict['status']: 'optimal'

        # Get the solution order
        res = model.x.get_values()
        order = [key for key, value in res.items() if value is not None and value > 0]
        results_dict['solution'] = order

        # Extended solution with run order
        run_order = [runs[val[1]] for val in order]
        results_dict['extended_solution'] = run_order

        # Calculate biases for each factor
        biases = [0 for _ in factors]
        for factor in range(len(factors)):
            for i in range(len(run_order)):
                biases[factor] += (i + 1) * run_order[i][factor]
        results_dict['biases'] = biases
        results_dict['max_bias'] = max(list(map(abs, biases)))

        # Calculate total level changes between runs
        distance_total = 0
        for i in range(len(order) - 1):
            distance_total += distances[(order[i][1], order[i + 1][1])]
        results_dict['distance'] = distance_total
    else:
        results_dict['status'] = 'unknown'

    return results_dict




    


