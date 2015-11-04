__author__ = 'nick'

"""
Multiple traveling salesmen (mTSP) problem solver.
"""

import math
import random
import numpy as np
from gurobipy import *

DEBUG = True


def mtsp_solver(cost, salesmen, min_cities=None, max_cities=None):
    """
    Multiple traveling salesmen MILP solver using the Gurobi MILP optimiser.
    :rtype : Returns tuple with the routes for each salesman, the objective value, and a model object.
    :param cost: Cost matrix for travelling from point to point.
    :param salesmen: Number of salesmen taking part in the solution.
    :param min_cities: Optional parameter of minimum cities to be visited by each salesman.
    :param max_cities: Optional parameter of maximum cities to be visited by each salesman.
    """

    n = cost.shape[0]

    if min_cities is None:
        K = 2
    else:
        K = min_cities

    if max_cities is None:
        L = n
    else:
        L = max_cities

    m = Model()

    # Create variables

    vars = {}
    for i in range(n):
        for j in range(n):
            vars[i, j] = m.addVar(obj=cost[i, j], vtype=GRB.BINARY, name='e'+str(i)+'_'+str(j))
    m.update()

    for i in range(n):
        vars[i, i].ub = 0

    uVars = {}
    for i in range(n):
        uVars[i] = m.addVar(lb=0, ub=L, vtype=GRB.INTEGER, name='u'+str(i))
    m.update()

    # Add degree-2 constraint, and forbid loops

    m.addConstr(quicksum(vars[0, i] for i in range(1, n)) == salesmen)

    m.addConstr(quicksum(vars[i, 0] for i in range(1, n)) == salesmen)

    for i in range(1, n):
        m.addConstr(quicksum(vars[i, j] for j in range(n) if i != j) == 1)

    for i in range(1, n):
        m.addConstr(quicksum(vars[j, i] for j in range(n) if i != j) == 1)

    for i in range(1, n):
        m.addConstr(uVars[i] + (L - 2)*vars[0, i] - vars[i, 0] <= L - 1)

    for i in range(1, n):
        m.addConstr(uVars[i] + vars[0, i] + (2 - K)*vars[i, 0] >= 2)

    for i in range(1, n):
        m.addConstr(vars[0, i] + vars[i, 0] <= 1)

    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                m.addConstr(uVars[i] - uVars[j] + L*vars[i, j] + (L - 2)*vars[j, i] <= L - 1)
    m.update()

    m._vars = vars
    m._uvars = uVars
    m.optimize()

    try:
        solution = m.getAttr('X', vars)
        u = m.getAttr('X', uVars)
        selected = [(i, j) for i in range(n) for j in range(n) if solution[i, j] > 0.5]

        if DEBUG:
            mat = np.zeros((n, n))

            for k,v in solution.iteritems():
                mat[k[0], k[1]] = v

            print(mat)
            print(selected)
            print(u)

        routes = []

        for i in range(salesmen):
            routes.append([])
            next_city = 0
            finished = False
            while not finished:
                for j in range(len(selected)):
                    if selected[j][0] == next_city:
                        routes[i].append(next_city)
                        next_city = selected[j][1]
                        selected.pop(j)
                        break
                if next_city == 0:
                    finished = True

        return routes, m.objVal, m
    except GurobiError:
        return 0, 0, 0


def main():
    import time
    RANDOM = False
    if RANDOM:
        # generate random problem
        n = 4
        points = np.random.randint(-50, 50, (n, 2))
        cities = ['c_{}'.format(k) for k in xrange(n)]
    else:
        n = 5
        points = np.zeros((n,2))
        points[1, :] = [1, 1]
        points[2, :] = [1, 2]
        points[3, :] = [-1, 1]
        points[4, :] = [-1, 2]
        print(points)

    # standard cost
    distances = np.zeros((n, n))

    for k in xrange(n):
        for p in xrange(n):
            distances[k, p] = np.linalg.norm(points[k, :] - points[p, :])

    print(distances)

    # solve using the Gurobi solver
    st = time.time()
    tsp_route, total_cost, model = mtsp_solver(distances, 2)
    dt = time.time() - st

    print('Gurobi Solver')
    print('Time to Solve: %.2f secs' % dt)
    print('Cost: %.3f' % total_cost)
    print('TSP Route: %s\n' % tsp_route)


if __name__ == '__main__':
    main()
