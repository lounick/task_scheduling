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
Simple path optimiser that accepts an entry and exit point in literature it is defined as Open Vehicle Routing Problem
"""

from __future__ import division

import numpy as np
from gurobipy import *


def ovrp_problem(cost, start=None, finish=None, **kwargs):
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

    m = Model()

    # Create model variables
    e_vars = {}
    for i in range(n):
        for j in range(n):
            e_vars[i, j] = m.addVar(obj=cost[i, j], vtype=GRB.BINARY, name='e'+str(i)+'_'+str(j))

    m.update()

    for i in range(n):
        e_vars[i, i].ub = 0
    m.update()

    u_vars = {}
    for i in range(n):
        u_vars[i] = m.addVar(vtype=GRB.INTEGER, name='u'+str(i))
    m.update()

    # None exits the finish point
    m.addConstr(quicksum(e_vars[finish, j] for j in range(n)) == 0)
    m.update()

    # From all other points someone exits
    for i in range(n):
        if i != finish:
            m.addConstr(quicksum(e_vars[i, j] for j in range(n)) == 1)
    m.update()

    # None enters the starting point
    m.addConstr(quicksum(e_vars[j, start] for j in range(n)) == 0)
    m.update()

    # To all other points someone enters
    for i in range(n):
        if i != start:
            m.addConstr(quicksum(e_vars[j, i] for j in range(n)) == 1)
    m.update()

    # Sub-tour elimination constraint
    for i in range(n):
        for j in range(n):
            if i != j:
                m.addConstr(u_vars[i] - u_vars[j] + n * e_vars[i, j] <= n-1)
    m.update()

    m._vars = e_vars
    m._uVars = u_vars
    m.params.OutputFlag = 0
    m.optimize()

    solution = m.getAttr('X', e_vars)
    u = m.getAttr('X', u_vars)
    selected = [(i, j) for i in range(n) for j in range(n) if solution[i, j] > 0.5]

    route = np.zeros(n, dtype=np.int)

    for k, v in u.iteritems():
        route[v] = int(k)

    return route, m.objVal, m


def main():
    import time
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    mpl.style.use('bmh')
    mpl.rcParams['figure.figsize'] = (12, 6)
    np.random.seed(47)

    def __plot_problem(ips, tsp_route, total_cost):
        idx = tsp_route #[int(city.split('c_')[1]) for city in tsp_route]
        ips_route = ips[idx, :]

        fig, ax = plt.subplots()
        ax.plot(ips[:, 1], ips[:, 0], 'o', label='inspection points')
        ax.plot(ips_route[:, 1], ips_route[:, 0], 'r-', alpha=0.3)

        for n in xrange(len(idx)):
            x, y = ips[n, 1], ips[n, 0]
            xt, yt = x - 0.10 * np.abs(x), y - 0.10 * np.abs(y)

            ax.annotate('#%d' % n, xy=(x, y), xycoords='data', xytext=(xt,yt))

        for k, n in enumerate(idx):
            x, y = ips[n, 1], ips[n, 0]
            xt, yt = x + 0.05 * np.abs(x), y + 0.05 * np.abs(y)

            ax.annotate(str(k), xy=(x, y), xycoords='data', xytext=(xt,yt))

        ax.axis('equal')

        ax.set_xlabel('East (m)')
        ax.set_ylabel('North (m)')
        ax.set_title('TSP Problem')

        return fig, ax

    def __plot_problem3d(ips, tsp_route, total_cost):
        idx = tsp_route
        ips_route = ips[idx, :]
        fig, ax = plt.subplots()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(ips[:, 1], ips[:, 0], ips[:,2], 'o', label='inspection points')
        ax.plot(ips_route[:, 1], ips_route[:, 0], ips_route[:, 2], 'r-', alpha=0.3)
        return fig, ax

    # generate random problem
    n = 12
    points = np.random.randint(-50, 50, (n, 2))
    cities = ['c_{}'.format(k) for k in xrange(n)]

    # standard cost
    distances = np.zeros((n, n))

    for k in xrange(n):
        for p in xrange(n):
            distances[k, p] = np.linalg.norm(points[k, :] - points[p, :])

    # solve using the Gurobi solver
    st = time.time()
    tsp_route, total_cost, model = ovrp_problem(distances, 1, 2)
    dt = time.time() - st

    print('Gurobi Solver')
    print('Time to Solve: %.2f secs' % dt)
    print('Cost: %.3f' % total_cost)
    print('TSP Route: %s\n' % tsp_route)

    if points.shape[1] == 3:
        fig, ax = __plot_problem3d(points, tsp_route, total_cost)
    else:
        fig, ax = __plot_problem(points, tsp_route, total_cost)

    plt.show()


if __name__ == '__main__':
    main()