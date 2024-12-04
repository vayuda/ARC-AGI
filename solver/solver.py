import numpy as np
from typing import List
from queue import PriorityQueue
from source import *
from phog.synthetic_generator import _flatten_objects

class Node:
    def __init__(self, operations, grid, depth):
        self.operations = operations
        self.grid = grid
        self.depth = depth

def distance(grid1, grid2):
    return np.count_nonzero((grid1 != grid2) == False)

def solve(base: ARC_Object, target: ARC_Object, input_objs: List[ARC_Object], relation_graph: RelationGraph, dsl_weights=None, input_objs_weights=None, max_depth: int=3) -> Node:
    if input_objs_weights is None:
        input_objs_weights = [1 / len(input_objs)] * len(input_objs)
    if dsl_weights is None:
        dsl_weights = [1 / len(dsl_operations)] * len(dsl_operations)

    frontier = PriorityQueue()
    start = Node([], base.grid, 0)
    frontier.put((0, start))

    while not frontier.empty():
        _, current = frontier.get()
        if np.array_equal(current.grid, target.grid):
            return current
        
        if current.depth > max_depth:
            print('Exceeds max depth! No solution found.')
            return None
        
        dsl_order = np.flip(np.argsort(dsl_weights))
        obj_order = np.flip(np.argsort(input_objs_weights))

        for i in obj_order:
            for j in dsl_order:
                operations = relation_graph.get_args(dsl_operations[j], input_objs[i])
                new_objs = [f(input_objs[i]) for f in operations]
                new_lst = [input_objs] * len(new_objs)
                assert len(new_lst) == len(new_objs) and len(new_objs) == len(operations)
                for k in range(len(operations)):
                    new_lst[k][i] = new_objs[k]
                    new_grid = _flatten_objects(base, new_lst[k])
                    dist = distance(new_grid, target.grid)
                    new_op = current.operations
                    new_op.append(operations[k])
                    node = Node(new_op, new_grid, current.depth + 1)
                    frontier.put(dist, node)
