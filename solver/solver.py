import numpy as np
import multiprocessing
from multiprocessing import Process, Queue
from typing import List
from itertools import product
from queue import PriorityQueue, Empty
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

def _solve(base: ARC_Object, target: ARC_Object, input_objs: List[ARC_Object], relation_graph: RelationGraph, dsl_weights=None, input_objs_weights=None, max_depth: int=1) -> Node:
    if relation_graph is None:
        return None
    if len(input_objs) <= 0:
        return None
    if input_objs_weights is None:
        input_objs_weights = [1 / len(input_objs)] * len(input_objs)
    if dsl_weights is None:
        dsl_weights = [1 / len(dsl_operations)] * len(dsl_operations)

    frontier = PriorityQueue()
    start = Node([], base, 0)
    frontier.put((0, start))

    while not frontier.empty():
        _, current = frontier.get()
        if current.depth >= max_depth:
            continue
        
        dsl_order = np.flip(np.argsort(dsl_weights))

        for i in dsl_order:
            try:
                new_objs = []
                for o in input_objs:
                    functions = relation_graph.get_args(dsl_operations[i], o)
                    if len(functions) > 10:
                        raise Exception('Not gonna try')
                    new_o = [f(o) for f in functions]
                    new_objs.append(new_o)
                
                new_op = current.operations + [dsl_operations[i]]
                for comb in product(*new_objs):
                    new_grid = _flatten_objects(base, list(comb))
                    dist = distance(new_grid.grid, target.grid)
                    node = Node(new_op, new_grid, current.depth + 1)
                    if dist == 0:
                        return node
                    frontier.put((dist, node))
            except:
                continue

    return None

def _solve_with_result(queue, *args, **kwargs):
    try:
        result = _solve(*args, **kwargs)
        queue.put(result)
    except Exception as e:
        queue.put(e)

def solve(*args, timeout=30, **kwargs):
    queue = Queue()
    process = Process(target=_solve_with_result, args=(queue, *args), kwargs=kwargs)
    process.start()

    try:
        result = queue.get(timeout=timeout)
    except Empty:
        process.terminate()
        process.join()
        return None
    finally:
        if process.is_alive():
            process.terminate()
            process.join()

    if isinstance(result, Exception):
        raise result
    
    return result