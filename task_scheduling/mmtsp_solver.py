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
#   Nikolaos Tsiogkas

"""
Multi-depot multiple traveling salesmen problem (MmTSP) solver.
"""

from __future__ import division

import numpy as np
from gurobipy import *


def mmtsp_problem(cost, num_depots, min_cities=None, max_cities=None):
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

    e_vars = {}
    for i in range(n):
        for j in range(n):
            e_vars[i, j] = m.addVar(obj=cost[i, j], vtype=GRB.BINARY, name='e_'+str(i)+'_'+str(j))
    m.update()

    u_vars = {}
    for i in range(n):
        u_vars[i] = m.addVar(vtype=GRB.INTEGER, name='u_'+str(i))
    m.update()

    for i in range(n):
        e_vars[i, i].ub = 0
    m.update()

    # From each depot to other nodes. Notice that in the final depot no-one exits.
    for i in range(num_depots):
        if i == 0:
            m.addConstr(quicksum(e_vars[i, j] for j in range(num_depots, n)) == 0)
        else:
            # Only one salesman allowed per depot (one robot per position)
            m.addConstr(quicksum(e_vars[i, j] for j in range(num_depots, n)) == 1)
    m.update()

    # From each node to the final depot. No-one returns to his original positions. They are forced to go to extraction.
    for j in range(num_depots):
        if j == 0:
            m.addConstr(quicksum(e_vars[i, j] for i in range(num_depots, n)) == num_depots-1)
        else:
            m.addConstr(quicksum(e_vars[i, j] for i in range(num_depots, n)) == 0)
    m.update()

    # For the task points someone enters
    for j in range(num_depots, n):
        m.addConstr(quicksum(e_vars[i, j] for i in range(n)) == 1)
    m.update()

    # For the task points someone exits
    for i in range(num_depots, n):
        m.addConstr(quicksum(e_vars[i, j] for j in range(n)) == 1)
    m.update()

    for i in range(num_depots, n):
        m.addConstr(
            u_vars[i] + (L-2)*quicksum(e_vars[k, i] for k in range(num_depots)) -
            quicksum(e_vars[i, k] for k in range(num_depots)) <= (L-1)
        )
    m.update()

    for i in range(num_depots, n):
        m.addConstr(
            u_vars[i] + quicksum(e_vars[k, i] for k in range(num_depots)) +
            (2-K)*quicksum(e_vars[i, k] for k in range(num_depots)) >= 2
        )
    m.update()

    for k in range(num_depots):
        for i in range(num_depots, n):
            m.addConstr(e_vars[k, i] + e_vars[i, k] <= 1)
    m.update()

    for i in range(num_depots, n):
        for j in range(num_depots, n):
            if i != j:
                m.addConstr(u_vars[i] - u_vars[j] + L*e_vars[i, j] + (L - 2)*e_vars[j, i] <= L - 1)
    m.update()

    m._vars = e_vars
    m._uvars = u_vars
    m.params.OutputFlag = 0
    m.optimize()

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
    tsp_route, total_cost, model = mmtsp_problem(distances, 3)
    dt = time.time() - st

    print('Gurobi Solver')
    print('Time to Solve: %.2f secs' % dt)
    print('Cost: %.3f' % total_cost)
    print('TSP Route: %s\n' % tsp_route)

if __name__ == '__main__':
    main()
