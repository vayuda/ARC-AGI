import numpy as np
from typing import List
from queue import PriorityQueue
from source import *
from phog.synthetic_generator import _flatten_objects

class Node:
    def __init__(self, operations, obj, depth):
        self.operations = operations
        self.object = obj
        self.depth = depth
    
    def __lt__(self, other):
        return self.depth < other.depth

def distance(grid1, grid2):
    return np.count_nonzero((grid1 == grid2) == False)

def solve(base: ARC_Object, target: ARC_Object, input_objs: List[ARC_Object], relation_graph: RelationGraph, dsl_weights=None, input_objs_weights=None, max_depth: int=1) -> Node:
    if input_objs_weights is None:
        input_objs_weights = [1 / len(input_objs)] * len(input_objs)
    if dsl_weights is None:
        dsl_weights = [1 / len(dsl_operations)] * len(dsl_operations)

    frontier = PriorityQueue()
    start = Node([], base, 0)
    frontier.put((0, start))

    while not frontier.empty():
        _, current = frontier.get()
        if np.array_equal(current.object.grid, target.grid):
            return current
        
        if current.depth >= max_depth:
            continue
        
        dsl_order = np.flip(np.argsort(dsl_weights))
        # obj_order = np.flip(np.argsort(input_objs_weights))

        # for i in obj_order:
        #     for j in dsl_order:
        #         operations = relation_graph.get_args(dsl_operations[j], input_objs[i])
        #         new_objs = [f(input_objs[i]) for f in operations]
        #         new_lst = [input_objs] * len(new_objs)
        #         assert len(new_lst) == len(new_objs) and len(new_objs) == len(operations)
        #         for k in range(len(operations)):
        #             new_lst[k][i] = new_objs[k]
        #             new_grid = _flatten_objects(base, new_lst[k])
        #             dist = distance(new_grid, target.grid)
        #             new_op = current.operations
        #             new_op.append(operations[k])
        #             node = Node(new_op, new_grid, current.depth + 1)
        #             frontier.put((dist, node))

        for i in dsl_order:
            print(dsl_operations[i].__name__)
            new_objs = []
            for o in input_objs:
                functions = relation_graph.get_args(dsl_operations[i], o)
                new_o = [f(o) for f in functions]
                new_objs.extend(new_o)
            new_grid = _flatten_objects(base, new_objs)
            new_grid.plot_grid()
            dist = distance(new_grid.grid, target.grid)
            new_op = current.operations
            new_op.append(dsl_operations[i])
            node = Node(new_op, new_grid, current.depth + 1)
            frontier.put((dist, node))
    
    print('Reaches max depth. No solution found.')
    return None
