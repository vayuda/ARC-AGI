import numpy as np
from typing import List, Tuple
from load_data import quick_load
from util import visualize_problem
from segmentation import extract_objects
from embedding import load_ViT
from objects import ARC_Object
from dsl import *
from constraints import RelationGraph
# see also: 1e0a9b12
# problem = quick_load('1caeab9d',dir='training')
problem = quick_load('1e0a9b12',dir='training')

arcobj = [(extract_objects(example['input'], method='monochrome_contour'),extract_objects(example['output'],'monochrome_contour')) for example in problem[0]]
relation_graph = RelationGraph(arcobj)
# relation_graph.visualize_graph(relation_graph.get_subgraph(source='input',target='output'))
relation_graph.find_bijection()
 