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
from gurobipy import *



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
    if where == GRB.callback.MIPSOL:
        V = set(range(model._n))
        idx_start = model._idxStart
        # idx_finish = model._idxFinish
        num_robots = model._num_robots

        # solmat = np.zeros((model._n, model._n))
        selected = []

        for k in range(num_robots):
            for i in V:
                sol = model.cbGetSolution([model._eVars[i, j, k] for j in V])
                selected += [(i, j) for j in V if sol[j] > 0.5]

            # solmat[i, :] = sol

        if len(selected) <= 1:
            return

        for k in range(len(selected)):
            el = selected[k]
            entry = el[0]
            if idx_start != entry:
                # if np.sum(solmat[:,entry]) != np.sum(solmat[entry,:]):
                for r in range(num_robots):
                    expr1 = quicksum(model._eVars[i, entry, r] for i in V)
                    expr2 = quicksum(model._eVars[entry, j, r] for j in V)
                    model.cbLazy(expr1, GRB.EQUAL, expr2)


def ctop_solver(cost, num_robots=None, profit=None, cost_max=None, idx_start=None, idx_finish=None, **kwargs):
    """Orienteering problem solver instance

    Cost constrained traveling salesman problem solver for a single vehicle using the Gurobi MILP optimiser.

    Parameters
    ----------
    cost : ndarray (n, dims)
        Cost matrix for traveling from point to point. Here is time (seconds) needed to go from points a to b.
    num_robots : Optional[integer]
        Number of robots in the team.
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

    # exponential decay
    max_range = float(kwargs.get('max_range', 2.0))
    alpha = -np.log(0.01) / max_range

    # Number of points
    n = cost.shape[0]

    # Check for default values
    if idx_start is None:
        idx_start = 0

    if idx_finish is None:
        idx_finish = n - 1

    if profit is None:
        profit = np.ones(n)

    if num_robots is None:
        num_robots = 1

    if cost_max is None:
        cost_max = [cost[idx_start, idx_finish]]*num_robots

    # Create the vertices set
    V = set(range(n))

    m = Model()

    # Create model variables
    e_vars = {}
    for k in range(num_robots):
        for i in V:
            for j in V:
                e_vars[i, j, k] = m.addVar(vtype=GRB.BINARY, name='e_' + str(i) + '_' + str(j) + '_' + str(k))
    m.update()

    for k in range(num_robots):
        for i in V:
            e_vars[i, i, k].ub = 0
    m.update()

    ei_vars = {}
    for i in V:
        ei_vars[i] = m.addVar(vtype=GRB.INTEGER, name='ei_' + str(i))
    m.update()

    for i in V.difference([idx_start, idx_finish]):
        ei_vars[i].ub = 1
        ei_vars[i].lb = 0
    m.update()

    u_vars = {}
    for k in range(num_robots):
        for i in V:
            u_vars[i, k] = m.addVar(vtype=GRB.INTEGER, name='u_' + str(i))
    m.update()

    # Set objective function (0)
    expr = 0
    for i in V:
        if i != idx_start and i != idx_finish:
            expr += profit[i] * ei_vars[i] + quicksum(
                profit[j] * np.exp(-alpha * cost[i, j]) * ei_vars[i] * (ei_vars[i] - ei_vars[j])
                for j in V if j != i and j != idx_start and j != idx_finish and cost[i,j] < 2
            )
    m.setObjective(expr, GRB.MAXIMIZE)
    m.update()

    # Constraints

    # Add constraints for the initial and final node (1)
    for k in range(num_robots):
        # None enters the starting point
        m.addConstr(quicksum(e_vars[j, idx_start, k] for j in V.difference([idx_start])) == 0, 's_entry_' + str(k))

        # None exits the finish point
        m.addConstr(quicksum(e_vars[idx_finish, j, k] for j in V.difference([idx_finish])) == 0, 'f_exit_' + str(k))

        # Always exit the starting point
        m.addConstr(quicksum(e_vars[idx_start, i, k] for i in V.difference([idx_start])) == 1, 's_exit_' + str(k))
        m.addConstr(quicksum(e_vars[idx_start, i, k] for i in V.difference([idx_start])) == ei_vars[idx_start], 'si_exit_' + str(k))

        # Always enter the finish point
        m.addConstr(quicksum(e_vars[i, idx_finish, k] for i in V.difference([idx_finish])) == 1, 'f_entry_' + str(k))
        m.addConstr(quicksum(e_vars[i, idx_finish, k] for i in V.difference([idx_finish])) == ei_vars[idx_finish], 'fi_entry_' + str(k))
    m.update()

    m.addConstr(quicksum(e_vars[idx_start, i, k] for i in V.difference([idx_start]) for k in range(num_robots)) == num_robots, 'si_exit_' + str(k))
    m.addConstr(quicksum(e_vars[i, idx_finish, k] for i in V.difference([idx_finish]) for k in range(num_robots)) == num_robots, 'fi_entry_' + str(k))

    # From all other points someone may exit
    for k in range(num_robots):
        for i in V.difference([idx_start, idx_finish]):
            m.addConstr(quicksum(e_vars[i, j, k] for j in V if i != j) <= 1, 'v_' + str(i) + ' ' + str(k) + '_exit')
            # m.addConstr(quicksum(e_vars[i, j, k] for j in V if i != j) == ei_vars[i], 'vi_' + str(i) + '_' + str(k) + '_exit')
    m.update()

    for i in V.difference([idx_start, idx_finish]):
        m.addConstr(quicksum(e_vars[i, j, k] for k in range(num_robots) for j in V if i != j) == ei_vars[i], 'vi_' + str(i) + ' ' + str(k) + '_exit')
        m.addConstr(quicksum(e_vars[j, i, k] for k in range(num_robots) for j in V if i != j) == ei_vars[i], 'vi_' + str(i) + ' ' + str(k) + '_exit')

    # To all other points someone may enter
    for k in range(num_robots):
        for i in V.difference([idx_start, idx_finish]):
            m.addConstr(quicksum(e_vars[j, i, k] for j in V if i != j) <= 1, 'v_' + str(i) + ' ' + str(k) + '_entry')
            # m.addConstr(quicksum(e_vars[j, i, k] for j in V if i != j) == ei_vars[i], 'vi_' + str(i) + '_' + str(k) + '_entry')
    m.update()

    # for i in V.difference([idx_start, idx_finish]):
    #     m.addConstr(quicksum(e_vars[j, i] for j in V if i != j) == quicksum(e_vars[i, j] for j in V if i != j), "v_" + str(i) + "_cardinality")
    # m.update()

    # Add cost constraints (3)
    for k in range(num_robots):
        expr = 0
        for i in V:
            for j in V:
                if j != idx_finish:
                    expr += 1*e_vars[i, j, k]
                expr += cost[i, j] * e_vars[i, j, k]
        m.addConstr(expr <= cost_max[k], 'max_energy_' + str(k))
    m.update()

    # Constraint (4)
    for k in range(num_robots):
        for i in V:
            u_vars[i, k].lb = 0
            u_vars[i, k].ub = n
    m.update()

    # Add subtour constraint (5)
    for k in range(num_robots):
        for i in V:
            for j in V:
                m.addConstr(u_vars[i, k] - u_vars[j, k] + 1, GRB.LESS_EQUAL, (n - 1)*(1 - e_vars[i, j, k]),
                            'sec_' + str(i) + '_' + str(j) + '_' + str(k))
    m.update()

    m._n = n
    m._eVars = e_vars
    m._uVars = u_vars
    m._eiVars = ei_vars
    m._idxStart = idx_start
    m._idxFinish = idx_finish
    m._num_robots = num_robots
    m.update()

    m.params.OutputFlag = int(kwargs.get('output_flag', 0))
    m.params.TimeLimit = float(kwargs.get('time_limit', 60.0))
    m.params.MIPGap = float(kwargs.get('mip_gap', 0.0))
    m.params.LazyConstraints = 1
    m.optimize(_callback)
    # m.optimize()

    solution = m.getAttr('X', e_vars)
    u = m.getAttr('X', u_vars)
    # selected = []
    # for k in range(num_robots):
    #     selected.append([(i, j, k) for i in V for j in V if solution[i, j, k] > 0.5])
    selected = [(i, j) for i in V for j in V for k in range(num_robots) if solution[i, j, k] > 0.5]
    # print(u)

    # print(selected)

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


    # route = []
    # next_city = idx_start
    #
    # while len(selected) > 0:
    #     for i in range(len(selected)):
    #         if selected[i][0] == next_city:
    #             route.append(next_city)
    #             next_city = selected[i][1]
    #             selected.pop(i)
    #             break
    #
    # route.append(next_city)

    routes = []

    for i in range(num_robots):
        routes.append([])
        next_city = 0
        finished = False

        while not finished:
            for j in range(len(selected)):
                if selected[j][0] == next_city:
                    routes[i].append(next_city)
                    next_city = selected[j][1]

                    selected.pop(j)
                    break

            if next_city == idx_finish:
                routes[i].append(next_city)
                finished = True

    return routes, m.objVal, m


def main():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import task_scheduling.utils as tsu
    import random

    mpl.rcParams['figure.facecolor']='white'

    nodes = tsu.generate_nodes(n=40)
    cost = tsu.calculate_distances(nodes)

    nodes = []
    random.seed(42)
    nodes.append([0,0])
    for i in range(1,10):
        for j in range(-4,5):
            ni = i
            nj = j
            # ni = random.uniform(-0.5,0.5) + i
            # nj = random.uniform(-0.5,0.5) + j
            nodes.append([ni,nj])
    nodes.append([10,0])
    nodes = np.array(nodes)
    cost = tsu.calculate_distances(nodes)
    max_cost = [25.5]
    # mc = [13.5, 13.5]
    num_robots = 6
    # mc = [(81+82)/float(num_robots), (81+82)/float(num_robots), (81+82)/float(num_robots), (81+82)/float(num_robots), (81+82)/float(num_robots), (81+82)/float(num_robots)]
    mc = [4.0*((81+82)/float(num_robots))/4.0]*num_robots
    print(mc)

     # for mc in max_cost:
    solution, objective, _ = tsu.solve_problem(ctop_solver, cost, num_robots=num_robots, cost_max=mc, output_flag=1, time_limit=360000, mip_gap=0.05)

        # util = 0
        # print(solution)
        # for i in solution:
        #     extras = 0
        #     if i != 0 and i != solution[len(solution)-1]:
        #         for j in range(cost.shape[0]):
        #             if j != i and j not in solution and j != 0 and j != solution[len(solution)-1] and cost[i,j] < 2:
        #                 extras += np.e**(-2*cost[i,j])
        #         util += 1 + extras
        #
        # print("Utility: {0}".format(util))
    # solution = [ [0, 4, 13, 3, 12, 21, 20, 11, 2, 1, 10, 19, 28, 29, 37, 38, 48, 49, 58, 59, 68, 77, 82],  [0, 14, 15, 24, 23, 32, 33, 43, 42, 51, 60, 61, 62, 63, 72, 81, 80, 71, 70, 79, 69, 78, 82],  [0, 22, 31, 30, 39, 40, 41, 50, 57, 56, 47, 46, 55, 64, 73, 65, 74, 75, 66, 67, 76, 82],  [0, 5, 6, 7, 8, 9, 18, 17, 16, 25, 26, 27, 36, 45, 54, 53, 44, 35, 34, 52, 82]]
    # solution = [[0, 4, 13, 22, 31, 30, 21, 20, 29, 38, 39, 48, 49, 40, 41, 50, 59, 58, 67, 76, 82],
    #  [0, 5, 14, 23, 24, 15, 16, 25, 34, 33, 42, 43, 44, 53, 52, 51, 60, 61, 70, 78, 82],
    #  [0, 6, 7, 8, 9, 18, 27, 26, 35, 36, 45, 54, 63, 72, 71, 81, 80, 79, 69, 77, 82],
    #  [0, 3, 12, 2, 1, 10, 19, 28, 37, 46, 47, 56, 55, 64, 73, 74, 65, 66, 75, 82]]
    # nodes = tsu.generate_grid(9,9,[10,10])
    # nodes = np.array(nodes)
    # print(nodes)
    # BEST GA 6
    # solution = [ [0, 4, 12, 21, 20, 29, 38, 48, 47, 56, 65, 75, 76, 82],  [0, 15, 24, 25, 33, 42, 43, 52, 61, 60, 69, 78, 82],  [0, 5, 14, 23, 22, 31, 32, 41, 50, 49, 58, 59, 68, 77, 82],  [0, 7, 16, 17, 9, 18, 27, 36, 45, 54, 63, 82],  [0, 34, 35, 44, 53, 62, 71, 72, 81, 80, 79, 82],  [0, 3, 2, 1, 19, 28, 37, 46, 55, 64, 74, 82]]
    # solution = []
    # solution = [ [0, 15, 24, 34, 43, 82],  [0, 13, 57, 66, 75, 82],  [0, 25, 44, 53, 62, 82],  [0, 5, 14, 23, 22, 31, 82],  [0, 7, 17, 26, 82],  [0, 29, 37, 47, 82],  [0, 42, 51, 61, 70, 82],  [0, 21, 30, 48, 76, 82],  [0, 32, 41, 50, 59, 68, 77, 82],  [0, 3, 2, 11, 20, 39, 49, 82]]
    # objective = 0
    # solution = [ [0, 5, 14, 13, 22, 31, 30, 39, 40, 41, 42, 51, 52, 44, 53, 62, 71, 70, 69, 60, 59, 50, 49, 58, 67, 68, 77, 82],  [0, 6, 7, 16, 15, 24, 23, 32, 33, 34, 35, 26, 17, 8, 9, 18, 27, 36, 45, 54, 63, 72, 81, 80, 79, 78, 82],  [0, 4, 3, 12, 11, 2, 1, 10, 19, 20, 29, 28, 37, 47, 46, 55, 56, 48, 57, 66, 65, 64, 73, 74, 75, 76, 82]]
    # objective = 0
    fig, ax = tsu.plot_problem(nodes, solution, objective)
    plt.show()

    nodes = []
    nodes.append([0,0])
    nodes += tsu.generate_grid(9, 9,[15,-18])
    nodes += tsu.generate_grid(6, 6, [33, -6.5])
    nodes += tsu.generate_grid(7, 7, [54, -9])
    nodes += tsu.generate_grid(6, 6, [12, 24.5])
    nodes += tsu.generate_grid(9, 9, [42, 9])
    nodes += tsu.generate_grid(7, 7, [60, 15])
    nodes.append([69,0])
    print(nodes)
    print(len(nodes))
    nodes = np.array(nodes)
    fig, ax = tsu.plot_problem(nodes, [], objective)
    plt.show()

if __name__ == '__main__':
    main()
