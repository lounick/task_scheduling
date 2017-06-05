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
Simple path optimiser that accepts an entry and exit point in literature it is defined as Open Vehicle Routing Problem
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
import gurobipy


def movrp_solver(cost, start=None, finish=None, num_agents=None, min_cities=None, max_cities=None, **kwargs):
    """
    Open vehicle routing problem solver for a single vehicle using the Gurobi MILP optimiser.

    :param cost: Cost matrix for traveling from point to point.
    :param start: Optional starting point for the tour. If none is provided the first point of the array is chosen
    :param finish: Optional ending point of the tour. If none is provided the last point of the array is chosen
    :return: Returns the route the cost and the model.
    """

    # Number of points
    n = cost.shape[0]

    # Check for default values
    if start is None:
        start = 0
    if finish is None:
        finish = n - 1

    if num_agents is None:
        num_agents = 1

    if min_cities is None:
        K = 0
    else:
        K = min_cities

    if max_cities is None:
        L = n
    else:
        L = max_cities

    m = gurobipy.Model()

    # Create model variables
    e_vars = {}
    for i in range(n):
        for j in range(n):
            e_vars[i, j] = m.addVar(obj=cost[i, j], vtype=gurobipy.GRB.BINARY, name='e' + str(i) + '_' + str(j))

    m.update()

    for i in range(n):
        e_vars[i, i].ub = 0
    m.update()

    u_vars = {}
    for i in range(n):
        u_vars[i] = m.addVar(vtype=gurobipy.GRB.INTEGER, name='u' + str(i))
    m.update()

    # None exits the finish point
    m.addConstr(gurobipy.quicksum(e_vars[finish, j] for j in range(n)) == 0)
    m.update()

    m.addConstr(gurobipy.quicksum(e_vars[j, finish] for j in range(n)) == num_agents)
    m.update()

    # From all other points someone exits
    for i in range(n):
        if i != finish and i != start:
            m.addConstr(gurobipy.quicksum(e_vars[i, j] for j in range(n)) == 1)
    m.update()

    # None enters the starting point
    m.addConstr(gurobipy.quicksum(e_vars[j, start] for j in range(n)) == 0)
    m.update()

    m.addConstr(gurobipy.quicksum(e_vars[start, j] for j in range(n)) == num_agents)
    m.update()

    # To all other points someone enters
    for i in range(n):
        if i != start and i != finish:
            m.addConstr(gurobipy.quicksum(e_vars[j, i] for j in range(n)) == 1)
    m.update()

    # Sub-tour elimination constraint

    for i in range(n):
        for j in range(n):
            if i != j:
                m.addConstr(u_vars[i] - u_vars[j] + n * e_vars[i, j] <= n - 1)
    m.update()

    # for i in range(n):
    #     if i != start and i != finish:
    #         m.addConstr(
    #             u_vars[i] + (L - 2) * gurobipy.quicksum(e_vars[k, i] for k in (start,finish)) -
    #             gurobipy.quicksum(e_vars[i, k] for k in (start,finish)) <= (L - 1)
    #         )
    # m.update()
    #
    # for i in range(n):
    #     if i != start and i != finish:
    #         m.addConstr(
    #             u_vars[i] + gurobipy.quicksum(e_vars[k, i] for k in (start,finish)) +
    #             (2 - K) * gurobipy.quicksum(e_vars[i, k] for k in (start,finish)) >= 2
    #         )
    # m.update()
    #
    # for k in (start,finish):
    #     for i in range(n):
    #         if i != start and i != finish:
    #             m.addConstr(e_vars[k, i] + e_vars[i, k] <= 1)
    # m.update()
    #
    # for i in range(n):
    #     if i != start and i != finish:
    #         for j in range(n):
    #             if i != j and j != start and j != finish:
    #                 m.addConstr(u_vars[i] - u_vars[j] + L * e_vars[i, j] + (L - 2) * e_vars[j, i] <= L - 1)
    # m.update()

    m._vars = e_vars
    m._uVars = u_vars
    m.params.OutputFlag = int(kwargs.get('output_flag', 0))
    m.params.TimeLimit = float(kwargs.get('time_limit', 60.0))
    m.params.MIPGap = float(kwargs.get('mip_gap', 0.0))
    m.optimize()

    solution = m.getAttr('X', e_vars)
    selected = [(i, j) for i in range(n) for j in range(n) if solution[i, j] > 0.5]
    routes = []

    for i in range(num_agents):
        routes.append([])
        next_city = selected[i][0]
        finished = False
        routes[i].append(next_city)
        next_city = selected[i][1]
        while not finished:
            for j in range(len(selected)):
                if selected[j][0] == next_city:
                    routes[i].append(next_city)
                    next_city = selected[j][1]
                    break

            if next_city == finish:
                routes[i].append(next_city)
                finished = True

    return routes, m.objVal, m


def main():
    import matplotlib.pyplot as plt
    import task_scheduling.utils as tsu

    nodes = tsu.generate_nodes()
    cost = tsu.calculate_distances(nodes)

    solution, cost_total, _ = tsu.solve_problem(movrp_solver, cost, num_agents=3)

    fig, ax = tsu.plot_problem(nodes, solution, cost_total)
    plt.show()


if __name__ == '__main__':
    main()
