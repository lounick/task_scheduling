#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2015, lounick and decabyte
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of task_scheduling nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Implementation of an integer linear formulation for maximizing the targets visited by a group of vehicles under cost
constraints. The vehicles have to start and finish at predefined places and it is allowed to skip targets.
"""

from __future__ import division

import numpy as np
from gurobipy import *


def subtourelim(model, where):
    if where == GRB.callback.MIPSOL:
        selected = []
        cities = []
        n = model._n
        num_depots = model._numDepots
        constraints = model._constraints
        cost = model._cost

        """
        1. Find out the tours for each vehicle.
        2. Ensure that each tour respects the costs.
        3. Ensure that there are no sub-tours
        """

        for i in range(n):
            sol = model.cbGetSolution([model._vars[i,j] for j in range(n)])
            selected += [(i, j) for j in range(n) if sol[j] > 0.5]

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
                        break
                if next_city == 0 or next_city in routes[i]:
                    routes[i].append(next_city)
                    finished = True

        for i in range(num_depots-1):
            num_cities = len(routes[i])
            expr = 0
            tmp_cost = 0
            for j in range(num_cities-1):
                tmp_cost += cost[routes[i][j], routes[i][j+1]]
                expr += model._vars[routes[i][j], routes[i][j+1]]*cost[routes[i][j], routes[i][j+1]]
            if tmp_cost > constraints[i]:
                model.cbLazy(expr <= constraints[i])


        # Subtour elimination
        for i in range(len(selected)):
            if selected[i][0] not in cities:
                cities.append(selected[i][0])
            if selected[i][1] not in cities:
                cities.append(selected[i][1])

        if len(selected) != len(cities) - 2:
            expr = 0
            for edge in selected:
                expr += model._vars[edge[0], edge[1]]
            model.cbLazy(expr == len(cities) - 2)


def ccmmtsp_problem(cost, num_depots, constraints, **kwargs):

    n = cost.shape[0]

    m = Model()

    e_vars = {}
    for i in range(n):
        for j in range(n):
            e_vars[i, j] = m.addVar(vtype=GRB.BINARY, name='e_'+str(i)+'_'+str(j))
    m.update()

    for i in range(n):
        e_vars[i, i].ub = 0
    m.update()

    m.setObjective(quicksum(e_vars[i, j] for i in range(n) for j in range(n)), GRB.MAXIMIZE)

    m.addConstr(quicksum(e_vars[0, j] for j in range(1, n)) == 0)

    for i in range(1, num_depots):
        pass

    for i in range(num_depots, n):
        pass

    #  # From each depot to other nodes. Notice that in the final depot no-one exits.
    # for i in range(num_depots):
    #     if i == 0:
    #         m.addConstr(quicksum(e_vars[i, j] for j in range(1, n)) == 0)
    #     else:
    #         # Only one salesman allowed per depot (one robot per position)
    #         m.addConstr(quicksum(e_vars[i, j] for j in range(num_depots, n)) == 1)
    #         m.addConstr(quicksum(e_vars[i, j] for j in range(num_depots)) == 0)
    # m.update()
    #
    # # From each node to the final depot. No-one returns to his original positions. They are forced to go to extraction.
    # for j in range(num_depots):
    #     if j == 0:
    #         m.addConstr(quicksum(e_vars[i, j] for i in range(num_depots, n)) == num_depots-1)
    #     else:
    #         m.addConstr(quicksum(e_vars[i, j] for i in range(num_depots, n)) == 0)
    # m.update()
    #
    # # For the task points someone enters
    # for j in range(num_depots, n):
    #     m.addConstr(quicksum(e_vars[i, j] for i in range(1, n)) <= 1)
    # m.update()
    #
    # # For the task points someone exits
    # for i in range(num_depots, n):
    #     m.addConstr(quicksum(e_vars[i, j] for j in range(num_depots, n)) + e_vars[i, 0] <= 1)
    # m.update()

    m._vars = e_vars
    m._n = n
    m._numDepots = num_depots
    m._constraints = constraints
    m._cost = cost
    m.params.OutputFlag = 1
    m.params.LazyConstraints = 1
    m.optimize(subtourelim)

    solution = m.getAttr('X', e_vars)
    selected = [(i, j) for i in range(n) for j in range(n) if solution[i, j] > 0.5]

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
                    break
            if next_city == 0:
                routes[i].append(next_city)
                finished = True

    return routes, m.objVal, m


def main():
    import time
    RANDOM = False
    if RANDOM:
        # generate random problem
        n = 4
        points = np.random.randint(-50, 50, (n, 2))
    else:
        n = 7
        points = np.zeros((n, 2))
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
    tsp_route, total_cost, model = ccmmtsp_problem(distances, 3, [6,6])
    dt = time.time() - st

    print('Gurobi Solver')
    print('Time to Solve: %.2f secs' % dt)
    print('Cost: %.3f' % total_cost)
    print('TSP Route: %s\n' % tsp_route)

if __name__ == '__main__':
    main()
