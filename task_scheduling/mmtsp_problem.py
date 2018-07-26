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
Multi-depot multiple traveling salesmen problem (MmTSP) solver.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
import gurobipy


def mmtsp_solver(cost, salesmen=1, min_cities=None, max_cities=None, **kwargs):
    """
    Multi-depot multiple traveling salesmen MILP solver for multi-robot task scheduling using the Gurobi MILP optimiser.
    Points (in the cost matrix) should be ordered in a specific order. The first point is the extraction point for the
    robots. The next points are the robot positions (depots) and the final points are the tasks to be visited.

    :rtype : Returns tuple with the routes for each salesman from each depot, the objective value, and a model object.
    :param cost: Cost matrix for travelling from point to point.
    :param salesmen: Number of salesmen taking part in the solution.
    :param min_cities: Optional parameter of minimum cities to be visited by each salesman.
    :param max_cities: Optional parameter of maximum cities to be visited by each salesman.
    """
    n = cost.shape[0]
    depots = salesmen + 1

    if min_cities is None:
        K = 0
    else:
        K = min_cities

    if max_cities is None:
        L = n
    else:
        L = max_cities

    m = gurobipy.Model()

    e_vars = {}
    for i in range(n):
        for j in range(n):
            e_vars[i, j] = m.addVar(obj=cost[i, j], vtype=gurobipy.GRB.BINARY, name='e_' + str(i) + '_' + str(j))
    m.update()

    u_vars = {}
    for i in range(n):
        u_vars[i] = m.addVar(vtype=gurobipy.GRB.INTEGER, name='u_' + str(i))
    m.update()

    for i in range(n):
        e_vars[i, i].ub = 0
    m.update()

    # From each depot to other nodes. Notice that in the final depot no-one exits.
    for i in range(depots):
        if i == 0:
            m.addConstr(gurobipy.quicksum(e_vars[i, j] for j in range(depots, n)) == 0)
        else:
            # Only one salesman allowed per depot (one robot per position)
            m.addConstr(gurobipy.quicksum(e_vars[i, j] for j in range(depots, n)) == 1)
    m.update()

    # From each node to the final depot. No-one returns to his original positions. They are forced to go to extraction.
    for j in range(depots):
        if j == 0:
            m.addConstr(gurobipy.quicksum(e_vars[i, j] for i in range(depots, n)) == depots - 1)
        else:
            m.addConstr(gurobipy.quicksum(e_vars[i, j] for i in range(depots, n)) == 0)
    m.update()

    # For the task points someone enters
    for j in range(depots, n):
        m.addConstr(gurobipy.quicksum(e_vars[i, j] for i in range(n)) == 1)
    m.update()

    # For the task points someone exits
    for i in range(depots, n):
        m.addConstr(gurobipy.quicksum(e_vars[i, j] for j in range(n)) == 1)
    m.update()

    for i in range(depots, n):
        m.addConstr(
            u_vars[i] + (L - 2) * gurobipy.quicksum(e_vars[k, i] for k in range(depots)) -
            gurobipy.quicksum(e_vars[i, k] for k in range(depots)) <= (L - 1)
        )
    m.update()

    for i in range(depots, n):
        m.addConstr(
            u_vars[i] + gurobipy.quicksum(e_vars[k, i] for k in range(depots)) +
            (2 - K) * gurobipy.quicksum(e_vars[i, k] for k in range(depots)) >= 2
        )
    m.update()

    for k in range(depots):
        for i in range(depots, n):
            m.addConstr(e_vars[k, i] + e_vars[i, k] <= 1)
    m.update()

    for i in range(depots, n):
        for j in range(depots, n):
            if i != j:
                m.addConstr(u_vars[i] - u_vars[j] + L * e_vars[i, j] + (L - 2) * e_vars[j, i] <= L - 1)
    m.update()

    m._vars = e_vars
    m._uvars = u_vars
    m.params.OutputFlag = int(kwargs.get('output_flag', 0))
    m.params.TimeLimit = float(kwargs.get('time_limit', 60.0))
    m.optimize()

    solution = m.getAttr('X', e_vars)
    selected = [(i, j) for i in range(n) for j in range(n) if solution[i, j] > 0.5]
    routes = []

    for i in range(salesmen):
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
    import matplotlib.pyplot as plt
    import task_scheduling.utils as tsu

    nodes = tsu.generate_nodes()
    cost = tsu.calculate_distances(nodes)
    salesmen = np.random.randint(2, 4)
    salesmen = 2
    solution, objective, _ = tsu.solve_problem(mmtsp_solver, cost, salesmen=salesmen,min_cities=int(cost.shape[0]/salesmen)-1,time_limit=3600, output_flag=1)

    fig, ax = tsu.plot_problem(nodes, solution, objective)
    plt.show()

    solution, objective, _ = tsu.solve_problem(mmtsp_solver, cost, salesmen=salesmen, min_cities=5)
    fig, ax = tsu.plot_problem(nodes, solution, objective)
    plt.show()
if __name__ == '__main__':
    main()
