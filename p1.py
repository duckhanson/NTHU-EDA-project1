'''
Inputs:
start end value: the start vertex, the end vertex and the weight of the edge.
e.g. 0 1 10, from 0 to 1 costs 10 units.
'''
num_vtx = 6
edges = [{i: 0} for i in range(num_vtx + 1)]


def p1_add_edge(direct_graph = False):
    
    '''
    Inputs:
    edgeIndex edgeWeight nameOfVertexU nameOfVertexV
    e.g. (e1 1 v1 v2)
    1. idx: edgeIndex.
    2. w: edgeWeight. 
    3. u: nameOfVertexU.
    4. v: nameOfVertexV.
    e.g. e1 1 v1 v2, description: edge 1, starting from v1 to v2, weights 1 unit.

    Note:
    p1_add_edge ends while input is nothing, i.e. empty.
    '''

    while True:
        edge = input("Enter an edge, e.g.(e1 1 v1 v2): ")
        if edge == "":
            break
        idx, w, u, v = edge.split()
        # parse inputs to useful information
        w = int(w)
        u = int(u[1:]) # remove char 'v'
        v = int(v[1:]) # remove char 'v'

        edges[u][v] = w

        if not direct_graph:
            edges[v][u] = w


p1_add_edge(direct_graph=False)

# implement 1, using naive approach
# using permuations to find all possible paths

import math
import numpy as np

def create_graph(edges):
    graph = np.asarray([[np.inf] * (num_vtx)] * (num_vtx))

    for u, u_edges in enumerate(edges):
        if u == 0:
            continue
        for v, w in u_edges.items():
            graph[u - 1][v - 1] = w
    
    return graph

graph = create_graph(edges)

'''
From https://youtu.be/cY4HiiFHO1o & https://github.com/kristiansandratama/travelling-salesman-problem-dynamic-programming/blob/main/main.py
Implement TSP-DP
'''
class TravellingSalesmanProblem:
    def __init__(self, distance, start):
        '''
        Main Idea is to use "memo" to record calculated paths.
        
        para:
            distance - 2D adjacency matrix represents graph
            start - The start node (0 ≤ S < N) 
        '''

        self.distance_matrix = distance
        self.start_city = start
        self.total_cities = len(distance)

        # 1111...111 represents all cities are visited
        self.end_state = (1 << self.total_cities) - 1
        
        self.memo = np.full((self.total_cities, 1<<self.total_cities), None)
        self.shortest_path = []
        self.min_path_cost = float('inf')

    def solve(self):
        self.__initialize_memo()

        for num_element in range(3, self.total_cities + 1):
            # The __initiate_combination function generates all bit sets
            # of size N with r bits set to 1. 
            # e.g. __initiate_combination(3, 4) = {0111, 1011, 1101, 1110}
            for subset in self.__initiate_combination(num_element):
                # if s is not in subset, meaning that we traversed s node.
                if self.__is_not_in_subset(self.start_city, subset):
                    continue

                for next_city in range(self.total_cities):

                    if next_city == self.start_city or self.__is_not_in_subset(next_city, subset):
                        continue

                    subset_without_next_city = subset ^ (1 << next_city)
                    min_distance = float('inf')

                    for last_city in range(self.total_cities):

                        if last_city == self.start_city or \
                                last_city == next_city or \
                                self.__is_not_in_subset(last_city, subset):
                            continue

                        new_distance = \
                            self.memo[last_city][subset_without_next_city] + self.distance_matrix[last_city][next_city]

                        if new_distance < min_distance:
                            min_distance = new_distance

                    self.memo[next_city][subset] = min_distance

        self.__calculate_min_cost()
        self.__find_shortest_path()

    def __calculate_min_cost(self):
        for i in range(self.total_cities):

            if i == self.start_city:
                continue

            path_cost = self.memo[i][self.end_state] + self.distance_matrix[i][self.start_city]

            if path_cost < self.min_path_cost:
                self.min_path_cost = path_cost
        

    def __find_shortest_path(self):
        state = self.end_state
        last_index = self.start_city

        for i in range(1, self.total_cities):
            best_index = -1
            best_distance = float('inf')
            # try to find the best path for [i] to [last] via state
            for j in range(self.total_cities):

                if j == self.start_city or self.__is_not_in_subset(j, state):
                    continue

                new_distance = self.memo[j][state] + self.distance_matrix[j][last_index]
                # prv_distance =
                if new_distance <= best_distance:
                    best_index = j
                    best_distance = new_distance

            self.shortest_path.append(best_index)
            state = state ^ (1 << best_index)
            last_index = best_index

        self.shortest_path.append(self.start_city)
        self.shortest_path.reverse()
        self.shortest_path.append(self.start_city)

    def __initialize_memo(self):
        for destination_city in range(self.total_cities):

            if destination_city == self.start_city:
                continue
                # Store the optimal value from start_city to destination_city
                # LHS - to go to destination_city, there exists a path that 
                # traverses start_city and destination_city and costs m[start_city][destination_city].
                # RHS - the cost from start_city to destination_city.
            self.memo[destination_city][1 << self.start_city | 1 << destination_city] = \
                self.distance_matrix[self.start_city][destination_city]

    # This method generates all bit sets of size n where r bits
    # are set to one. The result is returned as a list of integer masks.
    def __initiate_combination(self, num_element):
        subset_list = []
        self.__initialize_combination(0, 0, num_element, self.total_cities, subset_list)
        return subset_list

    # To find all the combinations of size r we need to recurse until we have
    # selected r elements (aka r = 0), otherwise if r != 0 then we still need to select
    # an element which is found after the position of our last selected element    
    def __initialize_combination(self, subset, at, num_element, total_cities, subset_list):

        elements_left_to_pick = total_cities - at
        if elements_left_to_pick < num_element:
            return

        if num_element == 0:
            subset_list.append(subset)
        else:
            for i in range(at, total_cities):
                # Try including this element
                subset |= 1 << i
                self.__initialize_combination(subset, i + 1, num_element - 1, total_cities, subset_list)
                # Backtrack and try the instance where we did not include this element
                subset &= ~(1 << i)

    @staticmethod
    def __is_not_in_subset(element, subset):
        return ((1 << element) & subset) == 0

distance_matrix_test_3 = graph # our hw test case

start_city = 0

tour = TravellingSalesmanProblem(distance_matrix_test_3, start_city)
tour.solve()

def start_from_1(path):
    return [p + 1 for p in path]


print("Shortest path :", start_from_1(tour.shortest_path))
print("Minimum path cost :", tour.min_path_cost)
