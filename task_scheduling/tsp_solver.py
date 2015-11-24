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
Simple TSP solver
"""

from __future__ import division

import numpy as np
import gurobipy
np.set_printoptions(precision=3, suppress=True)


def tsp_problem(cost, **kwargs):
    """Uses Gurobi solver with sub-tour optimization to generate the best tour given the distances (recommended)

    :param cost:
    :return:
    """
    m = gurobipy.Model()
    n = cost.shape[0]

    # Create variables
    e_vars = {}
    u_vars = {}

    for i in range(n):
        for j in range(i + 1):
            name = 'e{}_{}'.format(i, j)
            e_vars[i, j] = m.addVar(obj=cost[i, j], vtype=gurobipy.GRB.BINARY, name=name)
            e_vars[j, i] = e_vars[i, j]

    for i in range(n):
        name = 'u{}'.format(i)
        u_vars[i] = m.addVar(vtype=gurobipy.GRB.INTEGER, name=name)

    m.update()

    # Add degree-2 constraint, and forbid loops
    for i in range(n):
        m.addConstr(gurobipy.quicksum(e_vars[i, j] for j in range(n)) == 2)
        e_vars[i, i].ub = 0

    # pass variables to callbacks
    m._n = n
    m._eVars = e_vars
    m._uVars = u_vars
    m.update()

    # (optionally) write problem
    #model.write("tsp.lp")

    # set parameters
    m.params.OutputFlag = kwargs.get('output_flag', 0)
    m.params.LazyConstraints = 1

    # optimize model
    m.optimize(subtour_callback)
    n = m._n

    sol_u = m.getAttr('X', m._uVars)
    sol_e = m.getAttr('x', m._eVars)
    selected = [(i, j) for i in range(n) for j in range(n) if sol_e[i, j] > 0.5]

    sol_e = subtour_calculate(n, selected)
    assert len(sol_e) == n

    solution = [n for n in sol_e]
    cost_total = m.objVal

    return solution, cost_total, m


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


def main():
    import matplotlib.pyplot as plt
    import utils

    nodes = utils.generate_nodes()
    distances = utils.calculate_distances(nodes)

    solution, cost_total, _ = utils.solve_problem(tsp_problem, distances)

    fig, ax = utils.plot_problem(nodes, solution, cost_total)
    plt.show()

if __name__ == '__main__':
    main()