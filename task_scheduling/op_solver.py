#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Software License Agreement (BSD License)
#
#  Copyright (c) 2014, Ocean Systems Laboratory, Heriot-Watt University, UK.
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#   * Neither the name of the Heriot-Watt University nor the names of
#     its contributors may be used to endorse or promote products
#     derived from this software without specific prior written
#     permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  Original authors:
#   Nikolaos Tsiogkas, Valerio De Carolis

"""
Implementation of an integer linear formulation for maximizing the targets visited by a vehicle under cost constraint.
The vehicle has to start and finish at the first point and it is allowed to skip targets.
Described in:
Laporte, Gilbert, and Silvano Martello. "The selective travelling salesman problem."
Discrete applied mathematics 26.2 (1990): 193-207.
"""

from __future__ import division

import numpy as np
from gurobipy import *


def _callback(model, where):
    if where == GRB.callback.MIPSOL:
        n = model._n

        solmat = np.zeros((n, n))
        selected = []

        for i in range(n):
            sol = model.cbGetSolution([model._eVars[i,j] for j in range(n)])
            selected += [(i, j) for j in range(n) if sol[j] > 0.5]

            solmat[i, :] = sol

        if len(selected) <= 1:
            return

        for k in range(1, len(selected)):
            el = selected[k]
            entry = el[0]

            expr1 = quicksum(model._eVars[i, entry] for i in range(n))
            expr2 = quicksum(model._eVars[entry, j] for j in range(n))

            model.cbLazy(expr1, GRB.EQUAL, expr2)


def op_problem(cost, profit, cost_max, idx_start=None, idx_finish=None, **kwargs):
    """
    Cost constrained traveling salesman problem solver for a single vehicle using the Gurobi MILP optimiser.

    :param cost: Cost matrix for traveling from point to point. Here is time (seconds) needed to go from points a to b.
    :param profit: Profit vector for profit of visiting each point
    :param cost_max: Maximum running time of the mission in seconds
    :param idx_start: Optional starting point for the tour. If none is provided the first point of the array is chosen.
    :param idx_finish: Optional ending point of the tour. If none is provided the last point of the array is chosen.
    :return: Returns the route the cost and the model.
    """

    # Number of points
    n = cost.shape[0]

    if idx_start is None:
        idx_start = 0

    if idx_finish is None:
        idx_finish = n - 1

    m = Model()

    # Create model variables
    e_vars = {}
    for i in range(n):
        for j in range(n):
            e_vars[i, j] = m.addVar(vtype=GRB.BINARY, name='e_' + str(i) + '_' + str(j))
    m.update()

    for i in range(n):
        e_vars[i, i].ub = 0
    m.update()

    u_vars = {}
    for i in range(n):
        u_vars[i] = m.addVar(vtype=GRB.INTEGER, name='u_' + str(i))
    m.update()

    # Set objective function (0)
    expr = 0
    for i in range(1, n-1):
        for j in range(1, n):
            expr += profit[i] * e_vars[i, j]
    m.setObjective(expr, GRB.MAXIMIZE)
    m.update()

    # Constraints
    # None enters the starting point
    m.addConstr(quicksum(e_vars[j, idx_start] for j in range(n)) == 0)
    m.update()

    # None exits the finish point
    m.addConstr(quicksum(e_vars[idx_finish, j] for j in range(n)) == 0)
    m.update()

    # From all other points someone exits
    for i in range(n):
        if i != idx_finish:
            m.addConstr(quicksum(e_vars[i, j] for j in range(n) if i != j) <= 1)
    m.update()

    # To all other points someone enters
    for i in range(n):
        if i != idx_start:
            m.addConstr(quicksum(e_vars[j, i] for j in range(n) if i != j) <= 1)
    m.update()

    # Add constraints for the initial and final node (1)
    m.addConstr(quicksum(e_vars[idx_start, i] for i in range(1, n)) == 1)
    m.addConstr(quicksum(e_vars[i, idx_finish] for i in range(n - 1)) == 1)
    m.update()

    # # Add constraints for other nodes (2)
    # for k in range(1, n - 1):
    #     m.addConstr(quicksum(e_vars[i, k] for i in range(n - 1)) <= 1)
    #     m.addConstr(quicksum(e_vars[k, j] for j in range(1, n)) <= 1)

    # Add cost constraints (3)
    expr = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                expr += cost[i, j] * e_vars[i, j]
    m.addConstr(expr <= cost_max)
    m.update()

    # Constraint (4)
    u_vars[idx_start].lb = 0
    u_vars[idx_start].ub = 0

    for i in range(1, n):
        u_vars[i].lb = 1
        u_vars[i].ub = n
    m.update()

    # Add subtour constraint (5)
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                m.addConstr(u_vars[i] - u_vars[j] + 1, GRB.LESS_EQUAL, (n - 1)*(1 - e_vars[i, j]))
    m.update()

    m._n = n
    m._eVars = e_vars
    m._uVars = u_vars
    m.update()

    m.params.OutputFlag = 0
    m.params.LazyConstraints = 1
    m.optimize(_callback)

    solution = m.getAttr('X', e_vars)
    u = m.getAttr('X', u_vars)
    selected = [(i, j) for i in range(n) for j in range(n) if solution[i, j] > 0.5]

    solmat = np.zeros((n, n))
    for k, v in solution.iteritems():
        solmat[k[0], k[1]] = v

    # print("\n")
    # print(solmat)
    # print(u)
    # print(selected)
    # print(sum(cost[s[0], s[1]] for s in selected))

    route = []
    next_city = idx_start

    while len(selected) > 0:
        for i in range(len(selected)):
            if selected[i][0] == next_city:
                route.append(next_city)
                next_city = selected[i][1]
                selected.pop(i)
                break

    route.append(next_city)

    return route, m.objVal, m


def main():
    import time
    RANDOM = True
    if RANDOM:
        # generate random problem
        n = 10
        np.random.seed(42)
        points = np.random.randint(-50, 50, (n, 2))
        profits = np.ones(n)
    else:
        n = 5
        points = np.zeros((n, 2))
        points[1, :] = [1, 1]
        points[2, :] = [0, 10]
        points[3, :] = [0, 2]
        points[4, :] = [1, 2]

        profits = np.ones(n)

        #print(points)

    # standard cost
    distances = np.zeros((n, n))

    for k in xrange(n):
        for p in xrange(n):
            distances[k, p] = np.linalg.norm(points[k, :] - points[p, :])

    # Divide distances by maximum speed. To get time approximation.
    # distances = distances / 0.8

    #print(distances)

    # solve using the Gurobi solver
    st = time.time()

    max_cost = distances[0, -1]
    max_cost = 340
    print('Max Cost: %s' % max_cost)

    tsp_route, total_cost, model = op_problem(distances, profits, max_cost)


    dt = time.time() - st



    print('Gurobi Solver')
    print('Time to Solve: %.2f secs' % dt)
    print('Cost: %.3f' % total_cost)
    print('TSP Route: %s\n' % tsp_route)

if __name__ == '__main__':
    main()