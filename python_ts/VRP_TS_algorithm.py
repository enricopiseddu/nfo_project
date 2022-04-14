########################################################################################################################
#                                          University of Cagliari                                                      #
#                                   Master Degree in Computer Science                                                  #
#                                  Course of Network flows Optimization                                                #
#                                          Academic Year 2021/2022                                                     #
#                                                                                                                      #
#                                           Author: Piseddu Enrico                                                     #
#                                  Instructor: Prof. Di Francesco Massimo                                              #
########################################################################################################################

# This project implements a simple tabu search for the Vehicle Routing Problem

import copy
import numpy as np
import matplotlib.pyplot as plt
import time


# ************ CLASSES **************
class Customer:
    def __init__(self, _id, x, y, request):
        self._id = _id
        self.x = x  # x axis position
        self.y = y  # y axis position
        self.request = request


class Solution:
    def __init__(self, r1, r2, del_1, del_2, cost):
        self.r1 = r1  # 1st route
        self.r2 = r2  # 2nd route
        self.del_1 = del_1  # quantities shipped in 1st route
        self.del_2 = del_2  # quantities shipped in 2nd route
        self.cost = cost


# *********** END CLASSES ***********


# ********* FUNCTIONS **************
# This function compute the cost matrix according the triangle inequality
def populate_cost_matrix():
    costs.itemset((0, 1), compute_euclidean_distance(dep, c_1))
    costs.itemset((0, 2), compute_euclidean_distance(dep, c_2))
    costs.itemset((0, 3), compute_euclidean_distance(dep, c_3))
    costs.itemset((0, 4), compute_euclidean_distance(dep, c_4))
    costs.itemset((0, 5), compute_euclidean_distance(dep, c_5))
    costs.itemset((0, 6), compute_euclidean_distance(dep, c_6))
    costs.itemset((0, 7), compute_euclidean_distance(dep, c_7))

    costs.itemset((1, 2), compute_euclidean_distance(c_1, c_2))
    costs.itemset((1, 3), compute_euclidean_distance(c_1, c_3))
    costs.itemset((1, 4), compute_euclidean_distance(c_1, c_4))
    costs.itemset((1, 5), compute_euclidean_distance(c_1, c_5))
    costs.itemset((1, 6), compute_euclidean_distance(c_1, c_6))
    costs.itemset((1, 7), compute_euclidean_distance(c_1, c_7))

    costs.itemset((2, 3), compute_euclidean_distance(c_2, c_3))
    costs.itemset((2, 4), compute_euclidean_distance(c_2, c_4))
    costs.itemset((2, 5), compute_euclidean_distance(c_2, c_5))
    costs.itemset((2, 6), compute_euclidean_distance(c_2, c_6))
    costs.itemset((2, 7), compute_euclidean_distance(c_2, c_7))

    costs.itemset((3, 4), compute_euclidean_distance(c_3, c_4))
    costs.itemset((3, 5), compute_euclidean_distance(c_3, c_5))
    costs.itemset((3, 6), compute_euclidean_distance(c_3, c_6))
    costs.itemset((3, 7), compute_euclidean_distance(c_3, c_7))

    costs.itemset((4, 5), compute_euclidean_distance(c_4, c_5))
    costs.itemset((4, 6), compute_euclidean_distance(c_4, c_6))
    costs.itemset((4, 7), compute_euclidean_distance(c_4, c_7))

    costs.itemset((5, 6), compute_euclidean_distance(c_5, c_6))
    costs.itemset((5, 7), compute_euclidean_distance(c_5, c_7))

    costs.itemset((6, 7), compute_euclidean_distance(c_6, c_7))


# This function compute the euclidean distance between two customers
def compute_euclidean_distance(a, b):
    return np.sqrt(np.power(b.x - a.x, 2) + np.power(b.y - a.y, 2))


# This function returns the cost of a route. The route in input is a list of n nodes
def compute_cost_route(route):
    dist = 0
    for i in range(0, route.__len__() - 1):
        dist = dist + costs[route[i]][route[i + 1]]
        # print('for ', route[i], ' to ', route[i + 1], ' is ', dist)
    return dist


# This function prints the initial feasible solution chosen by the user
def display_initial_solution():
    print('**********************************')
    print('Initial cost routing is: ', total_cost_routing)
    print('Initial routes are: ', routes)
    print('\n\n')


# Function that try to swap two nodes and return the new cost, new routes and the nodes swapped
def try_to_swap_nodes(route1, route2, i, j):
    temp_1_route = route1
    temp_2_route = route2

    node1 = route1.__getitem__(i)
    node2 = route2.__getitem__(j)

    # remove
    temp_1_route.pop(i)
    temp_2_route.pop(j)

    # inserting new nodes in routes
    temp_1_route.insert(i, node2)
    temp_2_route.insert(j, node1)

    temp_cost_1_route = compute_cost_route(temp_1_route)
    temp_cost_2_route = compute_cost_route(temp_2_route)

    new_cost = temp_cost_1_route + temp_cost_2_route

    return new_cost, temp_1_route, temp_2_route, node1, node2


# This function swaps the deliveries of node1 and node2 and return the new deliveries vector
def try_to_swap_deliveries(node1, node2, deliv):
    deliv[1][node1] = deliv[0][node1]
    deliv[0][node1] = 0
    deliv[0][node2] = deliv[1][node2]
    deliv[1][node2] = 0

    return deliv


# This function searches a local optimum avoiding tabu movements
def optimum_local_search(tot_cost, deliveries):
    best_cost = tot_cost

    # we use i and j to scroll first and second list of routes.

    i = 1  # index for first list: it starts from 1 to exclude the depot (node with index 0)
    j = 1  # index for second list

    # we only try to swap internal nodes (except depot)
    max_i = routes[0].__len__() - 2  # it ends here to exclude the depot (final node)
    max_j = routes[1].__len__() - 2

    # we try to swap a pair of nodes of different routes (1st and 2nd route)
    while i <= max_i:
        new_cost_swapping, new_route_1, new_route_2, n1, n2 = try_to_swap_nodes(copy.deepcopy(routes[0]),
                                                                                copy.deepcopy(routes[1]), i, j)
        new_deliveries = try_to_swap_deliveries(n1, n2, copy.deepcopy(deliveries))

        # common_nodes = intersection(routes[0], routes[1])

        print('try to swap ', n1, ' and ', n2, ' with new cost ', new_cost_swapping)
        print('new routes: ', new_route_1, new_route_2)
        print('new deliv ', new_deliveries)

        # we update the new solution only if we get a better solution and if the capacity constraints are respected
        # we follow the rule of first improvement
        if (n1, n2) not in tabus and new_cost_swapping < best_cost and np.asarray(
                new_deliveries[0]).sum() <= vehicle_capacity and np.asarray(
            new_deliveries[1]).sum() <= vehicle_capacity:

            print('upd new sol!')

            # we record the tabu move
            # tabus.append(((n1, 1), (n2, 2)))  # node 1 is moved to 2nd route, so we make tabu the reverse movement

            routes[0] = new_route_1
            routes[1] = new_route_2

            best_cost = new_cost_swapping
            deliveries = copy.deepcopy(new_deliveries)

            # i = i + 1
            # j = j + 1
            break
        else:
            j = j + 1

        if j > max_j:
            i = i + 1
            j = 1

        print('\n\n')

    return best_cost, deliveries


# This function try to obtain a solution worse with respect to the local optimum
# and record the move to exit as tabu
def exit_from_local_optimum(cost, deliveries):
    opt_loc_cost = cost

    # we use i and j to scroll first and second list of routes.

    i = 1  # index for first list: it starts from 1 to exclude the depot (node with index 0)
    j = 1  # index for second list

    # we only try to swap internal nodes (except depot)
    max_i = routes[0].__len__() - 2  # it ends here to exclude the depot (final node)
    max_j = routes[1].__len__() - 2

    # we try to swap a pair of nodes of different routes (1st and 2nd route)
    while i <= max_i:
        new_cost_swapping, new_route_1, new_route_2, n1, n2 = try_to_swap_nodes(copy.deepcopy(routes[0]),
                                                                                copy.deepcopy(routes[1]), i, j)
        new_deliveries = try_to_swap_deliveries(n1, n2, copy.deepcopy(deliveries))

        # common_nodes = intersection(routes[0], routes[1])

        print('try to swap leaving local optimum', n1, ' and ', n2, ' with new cost ', new_cost_swapping)
        print('new routes: ', new_route_1, new_route_2)
        print('new deliv ', new_deliveries)

        # we update the new solution only if we get a better solution and if the capacity constraints are respected
        # we follow the rule of first improvement
        if new_cost_swapping > opt_loc_cost and np.asarray(
                new_deliveries[0]).sum() <= vehicle_capacity and np.asarray(
            new_deliveries[1]).sum() <= vehicle_capacity:

            print('successfully exit from local optimum')

            if len(tabus) > 1:
                tabus.pop(0)

            # we record the tabu move
            tabus.append((n2, n1))

            routes[0] = new_route_1
            routes[1] = new_route_2

            cost_ex = new_cost_swapping
            deliveries_ex = copy.deepcopy(new_deliveries)

            # i = i + 1
            # j = j + 1
            break
        else:
            j = j + 1

        if j > max_j:
            i = i + 1
            j = 1

        print('\n\n')

    return cost_ex, deliveries_ex


# ***** END FUNCTIONS *****


# ****** MAIN DATA ************
num_nodes = 8  # including depot

# we use only two vehicles
vehicle_capacity = 40
demand = [0, 7, 9, 15, 8, 6, 12, 9]

dep = Customer(0, 0, 0, 0)  # id customer, x position, y position and demand
c_1 = Customer(1, 4, 12, 7)
c_2 = Customer(2, 12, 16, 9)
c_3 = Customer(3, 16, 14, 15)
c_4 = Customer(4, 8, 2, 8)
c_5 = Customer(5, 10, 8, 6)
c_6 = Customer(6, 18, 6, 12)
c_7 = Customer(7, 2, 16, 9)

customers = [dep, c_1, c_2, c_3, c_4, c_5, c_6, c_7]

solutions = []

# ****** initial feasible solution (two vehicle used) **********
deliveries = [[0, 0, 9, 0, 8, 0, 12, 9], [0, 7, 0, 15, 0, 6, 0, 0]]  # vehicle 1 send 10 to customers 1,2 and 4
routes = [[0, 2, 4, 7, 6, 0],
          [0, 3, 5, 1, 0]]  # two routes, one for each vehicle. Each route starts and ends at the depot 0
# *******************************************

dim_costs_matrix = (num_nodes, num_nodes)

# initialize cost matrix
costs = np.zeros(dim_costs_matrix)

# populate upper right costs matrix
populate_cost_matrix()

# make cost matrix symmetric for practical purpose
costs = costs + costs.T

# cost for initial solution
cost_1_route = compute_cost_route(routes[0])
cost_2_route = compute_cost_route(routes[1])
total_cost_routing = cost_1_route + cost_2_route

sol = Solution(routes[0], routes[1], deliveries[0], deliveries[1], total_cost_routing)
solutions.append(sol)

# This list contains 'tabu' moves
tabus = []

display_initial_solution()

max_iters = 20
iteration = 0

# we compute time
start = time.time()

while iteration < max_iters:
    iteration = iteration + 1

    # LOCAL SEARCH by FIRST IMPROVEMENT
    best_cost, new_delivs = optimum_local_search(total_cost_routing, deliveries)

    print('\n\n**************LOCAL SEARCH RESULT********************')
    print('Best cost founded is: ', best_cost)
    print('Local optimum routes are: ', routes)
    print('Deliveries are: ', new_delivs)
    print('Tabu list is: ', tabus)
    print('\n\n')

    # we record the new solution
    sol = Solution(routes[0], routes[1], new_delivs[0], new_delivs[1], best_cost)
    solutions.append(sol)

    new_cost_ex, new_delivs_ex = exit_from_local_optimum(best_cost, new_delivs)

    print('\n\n**************EXIT FROM LOCAL OPTIMUM********************')
    print('Another cost founded is: ', new_cost_ex)
    print('Routes are: ', routes)
    print('Deliveries are: ', new_delivs_ex)

    # we record the new solution
    sol = Solution(routes[0], routes[1], new_delivs_ex[0], new_delivs_ex[1], new_cost_ex)
    solutions.append(sol)

    total_cost_routing = new_cost_ex
    deliveries = new_delivs_ex

end = time.time()

best_cost = 99999
for s in solutions:
    if s.cost < best_cost:
        best_cost = s.cost

        opt_route_1 = s.r1
        opt_route_2 = s.r2

        opt_del_1 = s.del_1
        opt_del_2 = s.del_2

print('\n\n\nFINAL SOLUTION FOUNDED: ')
print('Best cost: ', best_cost)
print('Best routes: ', opt_route_1, ' and ', opt_route_2)
print('Best deliveries: ', opt_del_1, ' and ', opt_del_2)

print('\nBest solution is founded in ', end-start, 'seconds')

data = []

for s in solutions:
    data.append(s.cost)

fig, simple_chart = plt.subplots()
plt.xlabel('iterations')
plt.ylabel('objective fun cost')

simple_chart.plot(data)
plt.grid(visible=True)
plt.show()
