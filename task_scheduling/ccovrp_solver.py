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
The vehicle has to start and finish at predefined places and it is allowed to skip targets.
"""

from __future__ import division

import numpy as np
from gurobipy import *


def subtourelim(model, where):
    if where == GRB.callback.MIPSOL:
        selected = []
        cities = []
        n = model._n
        start = model._start
        finish = model._finish
        for i in range(n):
            sol = model.cbGetSolution([model._vars[i,j] for j in range(n)])
            selected += [(i, j) for j in range(n) if sol[j] > 0.5]

        route = []

        next_city = start
        finished = False
        while not finished:
            for j in range(len(selected)):
                if selected[j][0] == next_city:
                    route.append(next_city)
                    next_city = selected[j][1]
                    break
            if ((next_city == finish or next_city in route) or (j == len(selected)-1 and next_city not in (selected[k][0] for k in range(len(selected))))):
                route.append(next_city)
                finished = True


        for i in range(len(selected)):
            if selected[i][0] not in cities:
                cities.append(selected[i][0])
            if selected[i][1] not in cities:
                cities.append(selected[i][1])

        if len(route) != len(cities):
            expr = 0
            for edge in selected:
                expr += model._vars[edge[0], edge[1]]
            model.cbLazy(expr == len(cities))


def ccovrp_problem(cost, max_cost, start=None, finish=None, **kwargs):
    """
    Cost constrained open vehicle routing problem solver for a single vehicle using the Gurobi MILP optimiser.

    :param cost: Cost matrix for traveling from point to point. Here is time (seconds) needed to go from points a to b.
    :param max_cost: Maximum running time of the mission in seconds
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
    e_vars = {}
    for i in range(n):
        for j in range(n):
            e_vars[i, j] = m.addVar(vtype=GRB.BINARY, name='e'+str(i)+'_'+str(j))

    m.update()

    for i in range(n):
        e_vars[i, i].ub = 0
    m.update()

    u_vars = {}
    for i in range(n):
        u_vars[i] = m.addVar(vtype=GRB.INTEGER, name='u'+str(i))
    m.update()

    m.setObjective(quicksum(e_vars[i, j] for i in range(n) for j in range(n)), GRB.MAXIMIZE)
    m.update()

    m.addConstr(quicksum(e_vars[i, j]*cost[i, j] for i in range(n) for j in range(n)) <= max_cost, "max_energy")
    m.update()

    # None exits the finish point
    m.addConstr(quicksum(e_vars[finish, j] for j in range(n)) == 0, "finish_out")
    m.update()

    # Always one enters the finish point
    m.addConstr(quicksum(e_vars[j, finish] for j in range(n)) == 1, "finish_in")
    m.update()

    # None enters the starting point
    m.addConstr(quicksum(e_vars[j, start] for j in range(n)) == 0, "start_in")
    m.update()

    # Always one must exit the starting point
    m.addConstr(quicksum(e_vars[start, j] for j in range(n)) == 1, "start_out")
    m.update()

    # For all other points one may or may not enter or exit
    for i in range(n):
        if i != finish and i != start:
            m.addConstr(quicksum(e_vars[i, j] for j in range(n)) <= 1, "target_"+str(i)+"_out")
    m.update()

    for i in range(n):
        if i != start:
            m.addConstr(quicksum(e_vars[j, i] for j in range(n)) <= 1, "target_"+str(i)+"_in")
    m.update()

    # Sub-tour elimination constraint
    for i in range(n):
        for j in range(n):
            if i != j:
                m.addConstr(u_vars[i] - u_vars[j] + n * e_vars[i, j] <= n-1)
    m.update()

    # Pass variables to model callbacks
    m._n = n
    m._vars = e_vars
    m._start = start
    m._finish = finish
    m.params.OutputFlag = 1
    m.params.LazyConstraints = 1
    m.optimize()
    # status = m.status
    # if status == GRB.status.UNBOUNDED:
    #     print('The model cannot be solved because it is unbounded')
    #     exit(0)
    # if status == GRB.status.OPTIMAL:
    #     print('The optimal objective is %g' % m.objVal)
    #     exit(0)
    # if status != GRB.status.INF_OR_UNBD and status != GRB.status.INFEASIBLE:
    #     print('Optimization was stopped with status %d' % status)
    #     exit(0)
    #
    # # do IIS
    # print('The model is infeasible; computing IIS')
    # m.computeIIS()
    # print('\nThe following constraint(s) cannot be satisfied:')
    # for c in m.getConstrs():
    #     if c.IISConstr:
    #         print('%s' % c.constrName)
    # status = m.status
    # if status == GRB.UNBOUNDED:
    #     print('The model cannot be solved because it is unbounded')
    #     exit(0)
    # if status == GRB.OPTIMAL:
    #     print('The optimal objective is %g' % m.objVal)
    #     exit(0)
    # if status != GRB.INF_OR_UNBD and status != GRB.INFEASIBLE:
    #     print('Optimization was stopped with status %d' % status)
    #     exit(0)
    #
    # # Relax the constraints to make the model feasible
    # print('The model is infeasible; relaxing the constraints')
    # orignumvars = m.NumVars
    # m.feasRelaxS(0, False, False, True)
    # m.optimize()
    # status = m.status
    # if status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
    #     print('The relaxed model cannot be solved \
    #            because it is infeasible or unbounded')
    #     exit(1)
    #
    # if status != GRB.OPTIMAL:
    #     print('Optimization was stopped with status %d' % status)
    #     exit(1)
    #
    # print('\nSlack values:')
    # slacks = m.getVars()[orignumvars:]
    # for sv in slacks:
    #     if sv.X > 1e-6:
    #         print('%s = %g' % (sv.VarName, sv.X))

    solution = m.getAttr('X', e_vars)
    selected = [(i, j) for i in range(n) for j in range(n) if solution[i, j] > 0.5]

    route = []
    next_city = start

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
    RANDOM = False
    if RANDOM:
        # generate random problem
        n = 4
        points = np.random.randint(-50, 50, (n, 2))
    else:
        n = 4
        points = np.zeros((n, 2))
        points[1, :] = [1, 1]
        points[2, :] = [1, 10]
        points[3, :] = [0, 2]
        # points[4, :] = [1, 2]
        print(points)

    # standard cost
    distances = np.zeros((n, n))

    for k in xrange(n):
        for p in xrange(n):
            distances[k, p] = np.linalg.norm(points[k, :] - points[p, :])

    # Divide distances by maximum speed. To get time approximation.
    # distances = distances / 0.8

    print(distances)

    # solve using the Gurobi solver
    st = time.time()
    try:
        tsp_route, total_cost, model = ccovrp_problem(distances, 12)
    except:
        pass
    dt = time.time() - st



    print('Gurobi Solver')
    print('Time to Solve: %.2f secs' % dt)
    print('Cost: %.3f' % total_cost)
    print('TSP Route: %s\n' % tsp_route)


if __name__ == '__main__':
    main()
