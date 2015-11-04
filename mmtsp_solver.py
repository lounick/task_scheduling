__author__ = 'nick'

"""
Multi-depot multiple traveling salesmen problem (MmTSP) solver.
"""

import math
import random
import numpy as np
from gurobipy import *

DEBUG = True

def mmtsp_solver(cost, num_depots, min_cities=None, max_cities=None):
    """
    Multi-depot multiple traveling salesmen MILP solver for multi-robot task scheduling using the Gurobi MILP optimiser.
    Points (in the cost matrix) should be ordered in a specific order. The first point is the extraction point for the
    robots. The next points are the robot positions (depots) and the final points are the tasks to be visited.
    :rtype : Returns tuple with the routes for each salesman from each depot, the objective value, and a model object.
    :param cost: Cost matrix for travelling from point to point.
    :param num_depots: Number of depots in the solution.
    :param min_cities: Optional parameter of minimum cities to be visited by each salesman.
    :param max_cities: Optional parameter of maximum cities to be visited by each salesman.
    """

    n = cost.shape[0]

    if min_cities is None:
        K = 0
    else:
        K = min_cities

    if max_cities is None:
        L = n
    else:
        L = max_cities

    m = Model()

    vars = {}
    for i in range(n):
        for j in range(n):
            vars[i, j] = m.addVar(obj=cost[i, j], vtype=GRB.BINARY,name='e_'+str(i)+'_'+str(j))
    m.update()

    uVars = {}
    for i in range(n):
        uVars[i] = m.addVar(vtype=GRB.INTEGER, name='u_'+str(i))
    m.update()

    for i in range(n):
        vars[i, i].ub = 0
    m.update()

    # From each depot to other nodes. Notice that in the final depot no-one exits.
    for i in range(num_depots):
        if i == 0:
            m.addConstr(quicksum(vars[i, j] for j in range(num_depots, n)) == 0)
        else:
            m.addConstr(quicksum(vars[i, j] for j in range(num_depots, n)) == 1) #only one salesman allowed per depot (one robot per position)
    m.update()

    # From each node to the final depot. No-one returns to his original positions. They are forced to go to extraction.
    for j in range(num_depots):
        if j == 0:
            m.addConstr(quicksum(vars[i, j] for i in range(num_depots, n)) == num_depots-1)
        else:
            m.addConstr(quicksum(vars[i, j] for i in range(num_depots, n)) == 0)
    m.update()

    # For the task points someone enters
    for j in range(num_depots, n):
        m.addConstr(quicksum(vars[i, j] for i in range(n)) == 1)
    m.update()

    # For the task points someone exits
    for i in range(num_depots, n):
        m.addConstr(quicksum(vars[i, j] for j in range(n)) == 1)
    m.update()

    for i in range(num_depots, n):
        m.addConstr(
            uVars[i] + (L-2)*quicksum(vars[k, i] for k in range(num_depots)) -
            quicksum(vars[i, k] for k in range(num_depots)) <= (L-1)
        )
    m.update()

    for i in range(num_depots, n):
        m.addConstr(
            uVars[i] + quicksum(vars[k, i] for k in range(num_depots)) +
            (2-K)*quicksum(vars[i, k] for k in range(num_depots)) >= 2
        )
    m.update()

    for k in range(num_depots):
        for i in range(num_depots, n):
            m.addConstr(vars[k, i] + vars[i, k] <= 1)
    m.update()

    for i in range(num_depots, n):
        for j in range(num_depots, n):
            if i != j:
                m.addConstr(uVars[i] - uVars[j] + L*vars[i, j] + (L - 2)*vars[j, i] <= L - 1)
    m.update()

    m._vars = vars
    m._uvars = uVars
    # m.params.LazyConstraints = 1
    # m.optimize(subtourelim)
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

        for i in range(num_depots-1):
            routes.append([])
            next_city = selected[i][0]
            finished = False
            while not finished:
                for j in range(len(selected)):
                    if selected[j][0] == next_city:
                        routes[i].append(next_city)
                        next_city = selected[j][1]
                        # selected.pop(j)
                        break
                if next_city == 0:
                    routes[i].append(next_city)
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
        n = 7
        points = np.zeros((n,2))
        points[1, :] = [1, 1]
        points[3, :] = [1, 2]
        points[5, :] = [3, 3]
        points[2, :] = [-1, 1]
        points[4, :] = [-1, 2]
        points[6, :] = [-3, 3]
        print(points)

    # standard cost
    distances = np.zeros((n, n))

    for k in xrange(n):
        for p in xrange(n):
            distances[k, p] = np.linalg.norm(points[k, :] - points[p, :])

    print(distances)

    # solve using the Gurobi solver
    st = time.time()
    tsp_route, total_cost, model = mmtsp_solver(distances, 3)
    dt = time.time() - st

    print('Gurobi Solver')
    print('Time to Solve: %.2f secs' % dt)
    print('Cost: %.3f' % total_cost)
    print('TSP Route: %s\n' % tsp_route)

if __name__ == '__main__':
    main()
