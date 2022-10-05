
num_vtx = 6
edges = [{i: 0} for i in range(num_vtx + 1)]

def my_add_edge(direct_graph = False):
    '''
    Inputs:
    1. start: the start vertex.
    2. end: the end vertex. 
    3. value: the weight of the edge.
    e.g. 0 1 10, from 0 to 1 costs 10 units.
    '''
    while True:
        start = int(input("start vtx: "))
        if start == -1:
            break       
        end = int(input("end vtx: "))
        assert start > 0
        assert end < num_vtx + 1
        cost = int(input("cost: "))
        edges[start][end] = cost
        if not direct_graph:
            edges[end][start] = cost

# add_edge(direct_graph=False)

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


# p1_add_edge(direct_graph=False)

''' 
In hw 1, edges = [{0: 0},
 {1: 0, 2: 1, 5: 4, 6: 2},
 {2: 0, 5: 5, 4: 8, 1: 1},
 {3: 0, 4: 1, 5: 5, 6: 6},
 {4: 0, 6: 2, 2: 8, 3: 1},
 {5: 0, 6: 3, 1: 4, 2: 5, 3: 5},
 {6: 0, 1: 2, 3: 6, 4: 2, 5: 3}]
'''

edges = [{0: 0},
 {1: 0, 2: 1, 5: 4, 6: 2},
 {2: 0, 5: 5, 4: 8, 1: 1},
 {3: 0, 4: 1, 5: 5, 6: 6},
 {4: 0, 6: 2, 2: 8, 3: 1},
 {5: 0, 6: 3, 1: 4, 2: 5, 3: 5},
 {6: 0, 1: 2, 3: 6, 4: 2, 5: 3}]


# implement 1, using naive approach
# using permuations to find all possible paths
import math
import numpy as np
from itertools import permutations

def create_graph(edges):
    graph = np.asarray([[np.inf] * (num_vtx)] * (num_vtx))

    for u, u_edges in enumerate(edges):
        if u == 0:
            continue
        for v, w in u_edges.items():
            graph[u - 1][v - 1] = w
    
    return graph

create_graph(edges)

def path_begin_from_1(path):
    return [p+1 for p in path]


# implementation of traveling Salesman Problem
def travellingSalesmanProblem(graph, src):
    # store all vertex apart from src vertex
    vertex = [i for i in range(num_vtx)]

    # store minimum weight Hamiltonian Cycle
    min_cost = np.inf
    min_path = []
    perms = permutations(vertex)
    for perm in perms:

        # store current Path weight(cost)
        cur_path_weight = 0

        # compute current path weight
        
        s = perm[0] 
        k = s
        for v in perm:
            cur_path_weight += graph[k][v]
            k = v
        cur_path_weight += graph[k][s]

        # update minimum
        if min_cost > cur_path_weight:
            min_cost = cur_path_weight
            min_path = perm
    
    return min_path, min_cost

graph = create_graph(edges=edges)
min_path, min_cost = travellingSalesmanProblem(graph, 0)
print(f"from implement 1 [brute force]: min_path={path_begin_from_1(min_path)}, min_cost={min_cost}")

# implement 2
'''
From https://youtu.be/cY4HiiFHO1o & https://github.com/kristiansandratama/travelling-salesman-problem-dynamic-programming/blob/main/main.py
Implement TSP-DP
'''
import numpy as np

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

        for i in range(1, self.total_cities):
            best_index = -1
            best_distance = float('inf')
            # try to find the best path for [i] to [last] via state
            for j in range(self.total_cities):

                if j == self.start_city or self.__is_not_in_subset(j, state):
                    continue

                new_distance = self.memo[j][state]

                if new_distance <= best_distance:
                    best_index = j
                    best_distance = new_distance

            self.shortest_path.append(best_index)
            state = state ^ (1 << best_index)

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


distance_matrix_test_1 = [
    [0, 328, 259, 180, 314, 294, 269, 391],
    [328, 0, 83, 279, 107, 131, 208, 136],
    [259, 83, 0, 257, 70, 86, 172, 152],
    [180, 279, 257, 0, 190, 169, 157, 273],
    [314, 107, 70, 190, 0, 25, 108, 182],
    [294, 131, 86, 169, 25, 0, 84, 158],
    [269, 208, 172, 157, 108, 84, 0, 140],
    [391, 136, 152, 273, 182, 158, 140, 0],
]

n = 6
distance_matrix_test_2 = np.full((n, n), np.inf)
distance_matrix_test_2[5][0] = 10
distance_matrix_test_2[1][5] = 12
distance_matrix_test_2[4][1] = 2
distance_matrix_test_2[2][4] = 4
distance_matrix_test_2[3][2] = 6
distance_matrix_test_2[0][3] = 8

distance_matrix_test_3 = graph # our hw test case



start_city = 0

tour = TravellingSalesmanProblem(distance_matrix_test_3, start_city)
tour.solve()

print(f"from implement 2 [dynamic programming]: min_path={path_begin_from_1(tour.shortest_path)}, min_cost={tour.min_path_cost}")





