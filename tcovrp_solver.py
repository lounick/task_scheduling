__author__ = 'nick'

"""
Implementation of an integer linear formulation for maximizing the targets visited by a vehicle in a specific amount
of time. The vehicle has to start and finish at predefined places and it is allowed to skip targets.
"""

import numpy as np
from gurobipy import *

HAS_GUROBI = True
DEBUG = True


def subtourelim(model, where):
    if where == GRB.callback.MIPSOL:
        selected = []
        cities = []
        n = model._n
        for i in range(n):
            sol = model.cbGetSolution([model._vars[i,j] for j in range(n)])
            selected += [(i,j) for j in range(n) if sol[j] > 0.5]

        for i in range(len(selected)):
            if selected[i][0] not in cities:
                cities.append(selected[i][0])
            if selected[i][1] not in cities:
                cities.append(selected[i][1])

        if len(selected) < len(cities) - 1:
            expr = 0
            for edge in selected:
                expr += model._vars[edge[0], edge[1]]
            model.cbLazy(expr == len(cities) - 1)


def tcovrp(cost, max_time, start = None, finish = None):

    """
    Open vehicle routing problem solver for a single vehicle using the Gurobi MILP optimiser.
    :param cost: Cost matrix for traveling from point to point. Here is time (seconds) needed to go from points a to b.
    :param max_time: Maximum running time of the mission in seconds
    :param start: Optional starting point for the tour. If none is provided the first point of the array is chosen.
    :param finish: Optional ending point of the tour. If none is provided the last point of the array is chosen.
    :return: Returns the route the cost and the model.
    """

    # Number of points
    n = cost.shape[0]

    # Check for default values
    if start is None:
        start = 0
    if finish is None:
        finish = n - 1

    m = Model()

    # Create model variables
    vars = {}
    for i in range(n):
        for j in range(n):
            vars[i, j] = m.addVar(vtype=GRB.BINARY, name='e'+str(i)+'_'+str(j))

    m.update()

    for i in range(n):
        vars[i, i].ub = 0
    m.update()

    m.setObjective(quicksum(vars[i, j] for i in range(n) for j in range(n)), GRB.MAXIMIZE)

    m.addConstr(quicksum(vars[i, j]*cost[i, j] for i in range(n) for j in range(n)) <= max_time)
    # m.update()

    # None exits the finish point
    m.addConstr(quicksum(vars[finish, j] for j in range(n)) == 0)
    # m.update()

    # Always one enters the finish point
    m.addConstr(quicksum(vars[j, finish] for j in range(n)) == 1)
    # m.update()

    # None enters the starting point
    m.addConstr(quicksum(vars[j, start] for j in range(n)) == 0)
    # m.update()

    # Always one must exit the starting point
    m.addConstr(quicksum(vars[start, j] for j in range(n)) == 1)
    # m.update()

    # For all other points one may or may not enter or exit
    for i in range(n):
        if i != finish and i != start:
            m.addConstr(quicksum(vars[i, j] for j in range(n)) <= 1)
    # m.update()

    for i in range(n):
        if i != start:
            m.addConstr(quicksum(vars[j, i] for j in range(n)) <= 1)
    m.update()

    # Will probably need some S.E.C.
    # The S.E.C. would involve getting all the cities involved in a solution. And then setting the sum of x to be equal
    # to the number of cities involved - 1. It will be implemented by a callback.

    # Pass variables to model callbacks
    m._n = n
    m._vars = vars
    m.params.LazyConstraints = 1
    m.optimize(subtourelim)

    try:
        solution = m.getAttr('X', vars)
        selected = [(i, j) for i in range(n) for j in range(n) if solution[i, j] > 0.5]

        if DEBUG:
            mat = np.zeros((n, n))

            for k,v in solution.iteritems():
                mat[k[0], k[1]] = v

            print(mat)
            print(selected)
            # print(u)

        route = []# np.zeros(n, dtype=np.int)
        next_city = start
        while len(selected) > 0:
            for i in range(len(selected)):
                if selected[i][0] == next_city:
                    route.append(next_city)
                    next_city = selected[i][1]
                    selected.pop(i)
                    break
        route.append(next_city)
        # for k, v in u.iteritems():
        #     route[v] = int(k)

        return route, m.objVal, m
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
        n = 4
        points = np.zeros((n,2))
        points[1, :] = [1, 1]
        points[2, :] = [1, 10]
        points[3, :] = [0, 2]
        print(points)

    # standard cost
    distances = np.zeros((n, n))

    for k in xrange(n):
        for p in xrange(n):
            distances[k, p] = np.linalg.norm(points[k, :] - points[p, :])

    distances = distances / 0.8

    print(distances)

    if HAS_GUROBI:
        # solve using the Gurobi solver
        st = time.time()
        tsp_route, total_cost, model = tcovrp(distances, 10)
        dt = time.time() - st

        print('Gurobi Solver')
        print('Time to Solve: %.2f secs' % dt)
        print('Cost: %.3f' % total_cost)
        print('TSP Route: %s\n' % tsp_route)


if __name__ == '__main__':
    main()