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
    m.params.OutputFlag = 0  # kwargs.get('output_flag', 0)
    m.params.LazyConstraints = 1

    # optimize model
    m.optimize(subtour_callback)
    n = m._n

    solution = m.getAttr('x', m._eVars)
    u = m.getAttr('X', m._uVars)
    selected = [(i, j) for i in range(n) for j in range(n) if solution[i, j] > 0.5]

    route = subtour_calculate(n, selected)
    assert len(route) == n

    route = [n for n in route]

    total_cost = m.objVal

    return route, total_cost, m


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
    import time
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.style.use('bmh')
    mpl.rcParams['figure.figsize'] = (12, 6)
    np.random.seed(47)

    def __plot_problem(ips, tsp_route, total_cost):
        idx = [city for city in tsp_route]
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


    # solve using the Gurobi solver
    st = time.time()
    tsp_route, total_cost, model = tsp_problem(distances)
    dt = time.time() - st

    print('Gurobi Solver')
    print('Time to Solve: %.2f secs' % dt)
    print('Cost: %.3f' % total_cost)
    print('TSP Route: %s\n' % tsp_route)

    fig, ax = __plot_problem(ips, tsp_route, total_cost)
    plt.show()

if __name__ == '__main__':
    main()