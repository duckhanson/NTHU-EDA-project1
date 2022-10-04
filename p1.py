
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
    graph = np.asarray([[np.inf] * (num_vtx + 1)] * (num_vtx + 1))

    for u, u_edges in enumerate(edges):

        for v, w in u_edges.items():
            graph[u][v] = w
    
    return graph


# implementation of traveling Salesman Problem
def travellingSalesmanProblem(graph, src):
    # store all vertex apart from src vertex
    vertex = [i for i in range(1, num_vtx + 1)]

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
    
    min_path = (*min_path, src)
    
    return min_path, min_cost

graph = create_graph(edges=edges)
min_path, min_cost = travellingSalesmanProblem(graph, 1)

print(f"from implement 1: min_path={min_path}, min_cost={min_cost}")

# implement 2, using dp approach
# record and update weights from each cities to other cities
# implementation of traveling Salesman Problem
def travellingSalesmanProblem(graph, src):

    # print(f"original graph: \n {graph}")
    # store all vertex apart from src vertex
    vertex = [i for i in range(1, num_vtx + 1)]

    # store minimum weight Hamiltonian Cycle
    min_path = np.inf
    
    # From i to j, if there exists a path via k whose cost is lower than directly from i to j,
    # we will choose path i -> k -> j, and update graph[i][j] = graph[i][k] + graph[k][j].
    for n_iter in range(num_vtx + 1):
        # print(f'#{n_iter}')
        # print(graph)
        for i in range(1, num_vtx + 1):
            for j in range(i, num_vtx + 1):
                for k in range(1, num_vtx + 1):
                    via_k = graph[i][k] + graph[k][j]

                    if graph[i][j] > via_k:
                        graph[i][j] = min(graph[i][j], via_k)
                        graph[j][i] = graph[i][j]
                        

	
    # print(f"after hamiltonian cycle solver: \n {graph}")
    
    path = [src]
    min_total_cost = 0
    cur = src
    for it in range(num_vtx-1):
        m_idx, m_cost = -1, np.inf
        for i, c in enumerate(graph[cur]):
            if i not in path and c < m_cost:
                m_cost = c
                m_idx = i
        path.append(m_idx)
        min_total_cost += m_cost
        # print(f"#iter{it}, graph[{cur}]={graph[cur]}, next city idx={m_idx}, cost={m_cost}, total_cost={min_total_cost}")
        cur = m_idx
    
    path.append(src)
    min_total_cost += graph[cur][src]
    # print(f"#iter{num_vtx-1}, graph[{cur}]={graph[cur]}, next city idx={src}, cost={graph[cur][src]}, total_cost={min_total_cost}\n\n")

    return path, min_total_cost

graph = create_graph(edges=edges)
min_path, min_cost = travellingSalesmanProblem(graph, 1)
print(f"from implement 2: min_path={min_path}, min_cost={min_cost}")







