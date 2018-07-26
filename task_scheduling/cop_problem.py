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

"""Correlated Orienteering problem solver

Implementation of an integer linear formulation for maximizing the targets visited by a vehicle under cost constraint.
The vehicle has to start and finish at the first point and it is allowed to skip targets.
Described in:
Yu, Jingjin, Mac Schwager, and Daniela Rus. "Correlated orienteering problem and its application to informative path
planning for persistent monitoring tasks." In Intelligent Robots and Systems (IROS 2014), 2014 IEEE/RSJ International
Conference on, pp. 342-349. IEEE, 2014.
"""

from __future__ import division

import numpy as np
import gurobipy as gu


def _callback(model, where):
    """Callback function for the solver

    Callback function that adds lazy constraints for the optimisation process. Here it dynamically imposes cardinality
    constraints for the vertices in the solution, ensuring that if a path enters a vertex there must be a path exiting.

    Parameters
    ----------
    model : object
        The gurobi model instance
    where : int
        Gurobi specific callback variable

    Returns
    -------

    """
    if where == gu.GRB.callback.MIPSOL:
        V = set(range(model._n))
        idx_start = model._idxStart
        # idx_finish = model._idxFinish

        selected = []

        for i in V:
            sol = model.cbGetSolution([model._eVars[i, j] for j in V])
            selected += [(i, j) for j in V if sol[j] > 0.5]
            # solmat[i, :] = sol

        if len(selected) <= 1:
            return

        for k in range(len(selected)):
            el = selected[k]
            entry = el[0]
            if idx_start != entry:
                # if np.sum(solmat[:,entry]) != np.sum(solmat[entry,:]):
                expr1 = gu.quicksum(model._eVars[i, entry] for i in V)
                expr2 = gu.quicksum(model._eVars[entry, j] for j in V)
                model.cbLazy(expr1, gu.GRB.EQUAL, expr2)


def cop_solver(cost, profit=None, dist=None, cost_max=None, idx_start=None, idx_finish=None, **kwargs):
    """Orienteering problem solver instance

    Cost constrained traveling salesman problem solver for a single vehicle using the Gurobi MILP optimiser.

    Parameters
    ----------
    cost : ndarray (n, dims)
        Cost matrix for traveling from point to point. Here is time (seconds) needed to go from points a to b.
    profit : Optional[vector]
        Profit vector for profit of visiting each point.
    cost_max : Optional[double]
        Maximum running time of the mission in seconds.
    idx_start : Optional[int]
        Optional starting point for the tour. If none is provided the first point of the array is chosen.
    idx_finish : Optional[int]
        Optional ending point of the tour. If none is provided the last point of the array is chosen.
    kwargs : Optional[list]
        Optional extra arguments/
    Returns
    -------
    route : list
        The calculated route.
    profit : double
        The profit of the route.
    m : object
        A gurobi model object.
    """

    # problem size
    n = cost.shape[0]

    # exponential decay
    max_range = float(kwargs.get('max_range', 2.0))
    alpha = -np.log(0.01) / max_range

    # sensing distances
    if dist is None or dist.shape != cost.shape:
        dist = np.copy(cost)

    # sensing energy
    node_energy = float(kwargs.get('node_energy', 1.0))

    # Check for default values
    if idx_start is None:
        idx_start = 0

    if idx_finish is None:
        idx_finish = n - 1

    if profit is None:
        profit = np.ones(n)

    if cost_max is None:
        cost_max = cost[idx_start, idx_finish]

    # Create the vertices set
    V = set(range(n))

    m = gu.Model()

    # Create model variables
    e_vars = {}
    for i in V:
        for j in V:
            e_vars[i, j] = m.addVar(vtype=gu.GRB.BINARY, name='e_' + str(i) + '_' + str(j))
    m.update()

    for i in V:
        e_vars[i, i].ub = 0
    m.update()

    ei_vars = {}
    for i in V:
        ei_vars[i] = m.addVar(vtype=gu.GRB.BINARY, name='ei_' + str(i))
    m.update()

    u_vars = {}
    for i in V:
        u_vars[i] = m.addVar(vtype=gu.GRB.INTEGER, name='u_' + str(i))
    m.update()

    # Set objective function (0)
    expr = 0
    for i in V:
        if i != idx_start and i != idx_finish:
            expr += profit[i] * ei_vars[i] + gu.quicksum(
                profit[j] * np.exp(-alpha * dist[i, j]) * ei_vars[i] * (ei_vars[i] - ei_vars[j])
                for j in V if j != i and j != idx_start and j != idx_finish and dist[i,j] < max_range
            )
    m.setObjective(expr, gu.GRB.MAXIMIZE)
    m.update()

    # Constraints

    # Add constraints for the initial and final node (1)
    # None enters the starting point
    m.addConstr(gu.quicksum(e_vars[j, idx_start] for j in V.difference([idx_start])) == 0, "s_entry")
    m.update()

    # None exits the finish point
    m.addConstr(gu.quicksum(e_vars[idx_finish, j] for j in V.difference([idx_finish])) == 0, "f_exit")
    m.update()

    # Always exit the starting point
    m.addConstr(gu.quicksum(e_vars[idx_start, i] for i in V.difference([idx_start])) == 1, "s_exit")
    m.addConstr(gu.quicksum(e_vars[idx_start, i] for i in V.difference([idx_start])) == ei_vars[idx_start], "si_exit")
    m.update()

    # Always enter the finish point
    m.addConstr(gu.quicksum(e_vars[i, idx_finish] for i in V.difference([idx_finish])) == 1, "f_entry")
    m.addConstr(gu.quicksum(e_vars[i, idx_finish] for i in V.difference([idx_finish])) == ei_vars[idx_finish], "fi_entry")
    m.update()

    # From all other points someone may exit
    for i in V.difference([idx_start, idx_finish]):
        m.addConstr(gu.quicksum(e_vars[i, j] for j in V if i != j) <= 1, "v_" + str(i) + "_exit")
        m.addConstr(gu.quicksum(e_vars[i, j] for j in V if i != j) == ei_vars[i], "vi_" + str(i) + "_exit")
    m.update()

    # To all other points someone may enter
    for i in V.difference([idx_start, idx_finish]):
        m.addConstr(gu.quicksum(e_vars[j, i] for j in V if i != j) <= 1, "v_" + str(i) + "_entry")
        m.addConstr(gu.quicksum(e_vars[j, i] for j in V if i != j) == ei_vars[i], "vi_" + str(i) + "_entry")
    m.update()

    # for i in V.difference([idx_start, idx_finish]):
    #     m.addConstr(gu.quicksum(e_vars[j, i] for j in V if i != j) == gu.quicksum(e_vars[i, j] for j in V if i != j), "v_" + str(i) + "_cardinality")
    # m.update()

    # Add cost constraints (3)
    expr = 0
    for i in V:
        if i != idx_start and i != idx_finish:
            expr += node_energy * ei_vars[i]

        for j in V:
            expr += cost[i, j] * e_vars[i, j]
    m.addConstr(expr <= cost_max, "max_energy")
    m.update()

    # Constraint (4)
    for i in V:
        u_vars[i].lb = 0
        u_vars[i].ub = n
    m.update()

    # Add subtour constraint (5)
    for i in V:
        for j in V:
            m.addConstr(
                u_vars[i] - u_vars[j] + 1,
                gu.GRB.LESS_EQUAL, (n - 1)*(1 - e_vars[i, j]),
                "sec_" + str(i) + "_" + str(j)
            )
    m.update()

    m._n = n
    m._eVars = e_vars
    m._uVars = u_vars
    m._eiVars = ei_vars
    m._idxStart = idx_start
    m._idxFinish = idx_finish
    m.update()

    m.params.OutputFlag = int(kwargs.get('output_flag', 0))
    m.params.TimeLimit = float(kwargs.get('time_limit', 60.0))
    m.params.MIPGap = float(kwargs.get('mip_gap', 0.0))
    m.params.LazyConstraints = 1
    m.optimize(_callback)
    # m.optimize()

    solution = m.getAttr('X', e_vars)
    u = m.getAttr('X', u_vars)
    selected = [(i, j) for i in V for j in V if solution[i, j] > 0.5]

    # solmat = np.zeros((n, n))
    # for k, v in solution.iteritems():
    #     solmat[k[0], k[1]] = v
    #
    # print("\n")
    # print(solmat)
    # print(u)
    # print(selected)
    # print(sum(cost[s[0], s[1]] for s in selected))
    # print(m.getAttr('X',ei_vars))

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


def calculate_utility(solution, distance, profit=None, max_range=2.0, **kwargs):
    """Calculates COP utility assuming starting and ending points are solution[0] and solution[-1]."""
    n = distance.shape[0]
    k = len(solution)

    # exponential decay
    alpha = -np.log(0.01) / max_range

    # output
    utility = 0

    if profit is None:
        profit = np.ones(n)
        profit[0] = 0
        profit[-1] = 0

    for i in solution:
        if i != 0 and i != solution[k - 1]:
            for j in xrange(n):
                if j != i and j not in solution and j != 0 and j != solution[k - 1] and distance[i, j] < max_range:
                    utility += np.exp(-alpha * distance[i, j])

            utility += profit[i]

    return utility


def main():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import task_scheduling.utils as tsu
    import random

    mpl.rcParams['figure.facecolor'] = 'white'

    nodes = tsu.generate_nodes(n=40)
    cost = tsu.calculate_distances(nodes)

    nodes = []
    random.seed(42)
    nodes.append([0,0])

    for i in range(1,6):
        for j in range(-2,3):
            ni = i
            nj = j
            # ni = random.uniform(-0.5,0.5) + i
            # nj = random.uniform(-0.5,0.5) + j
            nodes.append([ni,nj])

    nodes.append([6,0])
    nodes = np.array(nodes)

    cost = tsu.calculate_distances(nodes)
    max_cost = [51*2/4]

    # for mc in max_cost:
    #     solution, objective, _ = tsu.solve_problem(cop_solver, cost, cost_max=mc ,output_flag=1, time_limit=36000, mip_gap=0.01)
    #     utility = calculate_utility(solution, cost)
    #
    #     print("Utility: {0}".format(utility))

    # fig, ax = tsu.plot_problem(nodes, solution, objective)
    # solution = [0,3,4,9,10,15,20,25,19,13,12,7,1,6,11,16,21,22,23,26]
    # fig, ax = tsu.plot_problem_correlation_gradient(nodes, solution, objective)
    # ax.axis('equal')
    # plt.show()
    # plt.savefig('miqp-grad.png', dpi=300)
    solution = []
    objective = 0
    fig, ax = tsu.plot_problem_correlation_circles(nodes, solution, objective)
    ax.axis('equal')
    fig.savefig("/tmp/fig.png", dpi=300)
    plt.show()
    # plt.savefig('miqp-circ.png', dpi=300)
    #
    # solution = [0,3,2,1,6,11,16,17,18,13,8,9,5,10,15,20,25,24,23,26]
    # fig, ax = tsu.plot_problem_correlation_gradient(nodes, solution, objective)
    # ax.axis('equal')
    # plt.savefig('ga-grad.png', dpi=300)
    # fig, ax = tsu.plot_problem_correlation_circles(nodes, solution, objective)
    # ax.axis('equal')
    # plt.savefig('ga-circ.png', dpi=300)


if __name__ == '__main__':
    main()
