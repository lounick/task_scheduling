#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import copy
import numpy as np
np.set_printoptions(precision=3, suppress=True)

HAS_PULP = False
HAS_GUROBI = False

try:
    import pulp
    HAS_PULP = True
except ImportError:
    pass

try:
    import gurobipy
    HAS_GUROBI = True
except ImportError:
    pass


def solve_problem(cities, distances):
    if HAS_GUROBI:
        return gurobi_solve(cities, distances)
    elif HAS_PULP:
        return pulp_solve(cities, distances)
    else:
        return naive_solve(cities, distances)

def naive_solve(cities, distances, **kwargs):
    """Naive solution, no optimization done, useful only when no other optimization are available."""
    tsp_route = copy.copy(cities)
    total_cost = sum(distances[0])

    return tsp_route, total_cost, None


def gurobi_solve(cities, distances, **kwargs):
    """Uses Gurobi solver with sub-tour optimization to generate the best tour given the distances (recommended)"""
    model = gurobipy.Model()
    n = len(cities)

    # Create variables
    eVars = {}
    uVars = {}

    for i in range(n):
        for j in range(i + 1):
            name = 'e{}_{}'.format(i, j)
            eVars[i, j] = model.addVar(obj=distances[i][j], vtype=gurobipy.GRB.BINARY, name=name)
            eVars[j, i] = eVars[i, j]

    for i in range(n):
        name = 'u{}'.format(i)
        uVars[i] = model.addVar(vtype=gurobipy.GRB.INTEGER, name=name)

    model.update()

    # Add degree-2 constraint, and forbid loops
    for i in range(n):
        model.addConstr(gurobipy.quicksum(eVars[i, j] for j in range(n)) == 2)
        eVars[i, i].ub = 0

    # pass variables to callbacks
    model._n = n
    model._eVars = eVars
    model._uVars = uVars

    model.update()

    # (optionally) write problem
    #model.write("tsp.lp")

    # set parameters
    model.params.OutputFlag = kwargs.get('output_flag', 0)
    model.params.LazyConstraints = 1

    # optimize model
    model.optimize(subtour_callback)
    n = model._n

    solution = model.getAttr('x', model._eVars)
    selected = [(i,j) for i in range(n) for j in range(n) if solution[i,j] > 0.5]

    route = subtour_calculate(n, selected)
    assert len(route) == n

    tsp_route = [cities[n] for n in route]
    total_cost = model.objVal

    return tsp_route, total_cost, model


def subtour_callback(model, where):
    """Callback that use lazy constraints to eliminate sub-tours"""
    if where == gurobipy.GRB.callback.MIPSOL:
        n = model._n
        selected = []

        # make a list of edges selected in the solution
        for i in range(n):
            sol = model.cbGetSolution([model._eVars[i, j] for j in range(n)])
            selected += [(i, j) for j in range(n) if sol[j] > 0.5]

        # find the shortest cycle in the selected edge list
        tour = subtour_calculate(n, selected)

        # add a subtour elimination constraint
        if len(tour) < n:
            expr = 0

            for i in range(len(tour)):
                for j in range(i +1, len(tour)):
                    expr += model._eVars[tour[i], tour[j]]

            model.cbLazy(expr <= len(tour) - 1)

def subtour_calculate(n, edges):
    """Given a list of edges, finds the shortest subtour"""
    visited = [False] * n
    cycles = []
    lengths = []
    selected = [[] for i in range(n)]

    for x, y in edges:
        selected[x].append(y)

    while True:
        current = visited.index(False)
        this_cycle = [current]

        while True:
            visited[current] = True
            neighbors = [x for x in selected[current] if not visited[x]]

            if len(neighbors) == 0:
                break

            current = neighbors[0]
            this_cycle.append(current)

        cycles.append(this_cycle)
        lengths.append(len(this_cycle))

        if sum(lengths) == n:
            break

    return cycles[lengths.index(min(lengths))]


def pulp_solve(cities, distances, **kwargs):
    """Uses PuLP solver to generate a path if no other solver is available (slower)"""
    # calculating costs and routes
    routes = []
    costs = pulp.makeDict([cities, cities], distances, 0)

    for i in cities:
        for j in cities:
            if(i != j):
                routes.append((i,j))

    # creating problem
    prob = pulp.LpProblem('Travelling Salesman Problem', pulp.LpMinimize)
    x = pulp.LpVariable.dicts('route', (cities, cities), lowBound=0, upBound=1, cat=pulp.LpInteger)
    u = pulp.LpVariable.dicts('u', cities[1:], lowBound=2, upBound=len(cities), cat=pulp.LpInteger)

    # objective function is added to prob first
    prob += sum([x[w][b]*costs[w][b] for (w, b) in routes]), 'Sum_of_Tour_Costs'

    # in constraints
    for i in cities:
        tmp = []
        for j in cities:
            if(i != j):
                tmp.append(x[i][j])

        prob += sum(tmp) == 1, 'route_out_%s' % i

    # out constraints
    for i in cities:
        tmp = []
        for j in cities:
            if(i != j):
                tmp.append(x[j][i])
        prob += sum(tmp) == 1, 'route_in_%s' % i

    # dummy constraints
    for i in cities:
        for j in cities:
            if((i != j) and ((i != cities[0]) and (j != cities[0]))):
                prob += u[i] - u[j] + 1 <= (len(cities) - 1) * (1 - x[i][j])

    # (optional) write problem
    #prob.writeLP('tsp.lp')

    # solve problem
    prob.solve()

    # generate results
    tsp_route = [cities[0] for _ in range(len(cities))]
    total_cost = prob.objective.value()

    # extract tsp_route
    for v in prob.variables():
        if v.name.startswith('u_'):
            idx = int(v.varValue) - 1
            city = v.name.split('u_')[1]

            tsp_route[idx] = city

    #print('Status: %s' % pulp.LpStatus[prob.status])
    return tsp_route, total_cost, prob



def main():
    import time
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.style.use('bmh')
    mpl.rcParams['figure.figsize'] = (12, 6)
    np.random.seed(47)

    def __plot_problem(ips, tsp_route, total_cost):
        idx = [int(city.split('c_')[1]) for city in tsp_route]
        ips_route = ips[idx, :]

        fig, ax = plt.subplots()
        ax.plot(ips[:, 1], ips[:, 0], 'o', label='inspection points')
        ax.plot(ips_route[:, 1], ips_route[:, 0], 'r-', alpha=0.3)

        for k, n in enumerate(idx):
            x, y = ips[n, 1], ips[n, 0]
            xt, yt = x + 0.05 * np.abs(x), y + 0.05 * np.abs(y)

            ax.annotate(str(k), xy=(x, y), xycoords='data', xytext=(xt, yt))

        ax.set_xlabel('East (m)')
        ax.set_ylabel('North (m)')
        ax.set_title('TSP Problem')

        return fig, ax

    # generate random problem
    n = 20
    ips = np.random.randint(-50, 50, (n, 2))
    cities = ['c_{}'.format(k) for k in xrange(n)]

    # standard cost
    distances = np.zeros((n, n))

    for k in xrange(n):
        for p in xrange(n):
            distances[k, p] = np.linalg.norm(ips[k, :] - ips[p, :])

    # optional cost
    distances_return = np.zeros((n, n))

    for k in xrange(n):
        for p in xrange(n):
            distances_return[k, p] = np.linalg.norm(ips[k, :] - ips[p, :]) + np.linalg.norm(ips[p, :] - ips[0, :])

    # # solve using the Naive solver
    # st = time.time()
    # tsp_route, total_cost, _ = naive_solve(cities, distances)
    # dt = time.time() - st
    #
    # print('Naive Solver')
    # print('Time to Solve: %.2f secs' % dt)
    # print('Cost: %.3f' % total_cost)
    # print('TSP Route: %s\n' % tsp_route)
    #
    # fig, ax = __plot_problem(ips, tsp_route, total_cost)
    # plt.show()

    if HAS_PULP:
        # solve using the PuLP solver
        st = time.time()
        tsp_route, total_cost, prob = pulp_solve(cities, distances)
        dt = time.time() - st

        print('Problem Status: %s' % pulp.LpStatus[prob.status])
        print('Time to Solve: %.2f secs' % dt)
        print('Cost: %.3f' % total_cost)
        print('TSP Route: %s\n' % tsp_route)

        fig, ax = __plot_problem(ips, tsp_route, total_cost)
        plt.show()

    if HAS_GUROBI:
        # solve using the Gurobi solver
        st = time.time()
        tsp_route, total_cost, model = gurobi_solve(cities, distances, output_flag=0)
        dt = time.time() - st

        print('Gurobi Solver')
        print('Time to Solve: %.2f secs' % dt)
        print('Cost: %.3f' % total_cost)
        print('TSP Route: %s\n' % tsp_route)

        fig, ax = __plot_problem(ips, tsp_route, total_cost)
        plt.show()

if __name__ == '__main__':
    main()