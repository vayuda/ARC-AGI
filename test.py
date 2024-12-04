import numpy as np
from typing import List, Tuple
from source import *

# see also: 1e0a9b12
problem = quick_load('1caeab9d',dir='training')
# problem = quick_load('05f2a901',dir='training')
# problem = quick_load('1e0a9b12',dir='training')

arcobj = [(extract_objects(example['input'], method='monochrome_contour'),extract_objects(example['output'],'monochrome_contour')) for example in problem[0]]
relation_graph = RelationGraph(arcobj)
# relation_graph.visualize_graph()
# relation_graph.visualize_graph(relation_graph.get_io_graph())
# relation_graph.visualize_graph(relation_graph.get_input_graph())
# print(relation_graph.find_io_pairs())
# print(relation_graph.find_input_pairs())
o = relation_graph.objects[0][0][0]
print(relation_graph.get_args(color,o))
print(relation_graph.get_args(recolor,o))
print(relation_graph.get_args(translate,o))
print(relation_graph.get_args(draw_line,o))
print(relation_graph.get_args(rotate,o))
print(relation_graph.get_args(flip,o))
print(relation_graph.get_args(delete,o))
print(relation_graph.get_args(tile,o))