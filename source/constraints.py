import numpy as np
from typing import List, Tuple
from .objects import ARC_Object
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

class RelationGraph():
    def __init__(self, objects: List[Tuple[List[ARC_Object], List[ARC_Object]]]):
        self.objects = objects
        self.id_to_objects = {}
        self.objects_to_id = {}
        self.nexamples = len(objects)
        self.graph = nx.MultiGraph()
        self.relation_types = ['color', 'shape', 'y_pos', 'x_pos']
        def add_relations(obj1, obj2):
            id1 = self.objects_to_id[obj1]
            id2 = self.objects_to_id[obj2]
            if id1 == id2:
                return
            
            if obj1.color == obj2.color and obj1.color != None:
                self.graph.add_edge(id1, id2, key='color', color=obj1.color)
            if obj1.shape.shape == obj2.shape.shape:
                self.graph.add_edge(id1, id2, key='shape', shape=obj1.shape)
            if obj1.top_left[0] == obj2.top_left[0] and obj1.start[0] == obj2.start[0]:
                self.graph.add_edge(id1, id2, key='y_pos',y_pos=obj1.top_left[0])
            if obj1.top_left[1] == obj2.top_left[1] and obj1.start[1] == obj2.start[1]:
                self.graph.add_edge(id1, id2, key='x_pos',x_pos=obj1.top_left[1])

        def register_objects(objects_list, prefix, example_n):
            for i, obj in enumerate(objects_list):
                id = f"{prefix}_ex{example_n}_{i}"
                self.objects_to_id[obj] = id
                self.id_to_objects[id] = obj
                self.graph.add_node(id)
            
            # Add relations between objects in the same list
            for obj in objects_list:
                for other_obj in objects_list:
                    add_relations(obj, other_obj)

        for example_n, (input_objects, output_objects) in enumerate(objects):
            # Register input and output objects
            register_objects(input_objects, "input", example_n)
            register_objects(output_objects, "output", example_n)
            
            # Add relations between input and output objects
            for input_obj in input_objects:
                for output_obj in output_objects:
                    add_relations(input_obj, output_obj)
                    
        # add relations between input objects in different examples
        for i1, _ in objects:
            for i2, _ in objects:
                if i1 != i2:
                    for obj1 in i1:
                        for obj2 in i2:
                            add_relations(obj1, obj2)

        
        
    def get_subgraph(self, relations=[], source=None, target=None):
        if not relations:
            relations = self.relation_types
        edges = [(u, v, k, d) for (u, v, k, d) in self.graph.edges(data=True, keys=True)
                if (source is None or source in u) 
                and (target is None or target in v) 
                and (k in relations)]
        sg = nx.MultiGraph()
        sg.add_edges_from(edges)
        return sg
    
    def visualize_graph(self, graph=None):
        # Get nodes from first input example
        if graph is None:
            graph= self.graph
        pos = nx.spring_layout(graph)
        # Draw only selected nodes
        nx.draw_networkx_nodes(graph, pos, node_color='lightblue', node_size=400)

        # Draw edges between selected nodes
        # red: color, beige: shape, pink: y_pos, cyan: x_pos
        edge_colors = ['#FFB3B3', '#FFFFB3', '#B3B3FF', '#B3FFB3']
        widths = [16,8,4,2]
        for (width, edge_color, attrib) in zip(widths, edge_colors, self.relation_types):
            edgelist = [(u, v, k, d) for (u, v, k, d) in graph.edges(data=True, keys=True) if k == attrib]
            nx.draw_networkx_edges(graph, pos, edgelist=edgelist, width=width, edge_color=edge_color)
        
        # Add labels
        nx.draw_networkx_labels(graph, pos, font_size=10, font_color='black')
        # Add legend
        legend_elements = [plt.Line2D([0], [0], color=color, lw=width, label=label)
                 for color, width, label in zip(edge_colors, widths, self.relation_types)]
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.axis('off')
        plt.show()
    
    def get_input_graph(self, relation_types=None):
        if relation_types is None:
            relation_types = self.relation_types      
        edge_list = []
        for n in range(self.nexamples):
            for m in range(self.nexamples):
                if n ==m:
                    continue
                for relation in relation_types:
                    edge_list += [(u, v, k, d) for (u, v, k, d) in self.graph.edges(data=True, keys=True) if f'input_ex{n}' in u and f'input_ex{m}' in v and k == relation]
        input_graph = nx.MultiGraph()
        input_graph.add_edges_from(edge_list)
        return input_graph
    
    def bijection_check(self, graph, src, tgt, expected_nodes):
        if len(graph.nodes()) != expected_nodes:
            return False
        # Get input and output nodes
        inputs = [n for n in graph.nodes() if src in n]
        outputs = [n for n in graph.nodes() if tgt in n]
        
        # Check if every node has exactly one edge
        for node in graph.nodes():
            if graph.degree(node) != 1:
                return False
        
        # Check if number of inputs equals number of outputs
        if len(inputs) != len(outputs):
            return False
            
        return True
                         
    def find_io_pairs(self):
        for relation in self.relation_types:
            is_bijection = True
            graphs = []
            for n in range(self.nexamples):
                src = f'input_ex{n}'
                tgt = f'output_ex{n}'
                subgraph = self.get_subgraph(relations=[relation], source=src, target=tgt)
                graphs.append(subgraph)
                if not self.bijection_check(subgraph, src, tgt, len(self.objects[n][0]) + len(self.objects[n][1])):
                    is_bijection = False
                    break
                
            if is_bijection:
                print('Bijection found', relation)
                # create a dictionary of bijections
                mapping = {}
                for graph in graphs:
                    for (u, v, k, d) in graph.edges(data=True, keys=True):
                        if 'input' in u and 'output' in v:
                            mapping[u] = v
                return (relation, mapping)
    
    def find_input_pairs(self):
        '''
        finds mappings between inputs across exmaples for each relation type
        currently has a hard requirement that the number of objects in each example is the same
        later we can work on clustering similar objects to reduce it to a bijection
        another easier step is to store partial mappings (works for subset of examples) because the whole point of this is to reduce the search space 
        '''
        for relation in self.relation_types:
            is_bijection = True
            graphs = []
            for n1, n2 in [(n1, n2) for n1 in range(self.nexamples) for n2 in range(self.nexamples) if n1 < n2]:
                src = f'input_ex{n1}'
                tgt = f'input_ex{n2}'
                subgraph = self.get_subgraph(relations=[relation], source=src, target=tgt)
                graphs.append(subgraph)
                if not self.bijection_check(subgraph, src, tgt, len(self.objects[n1][0]) + len(self.objects[n2][0])):
                    is_bijection = False
                    break
            
            if is_bijection:
                print('Bijection found', relation)
                # group all objects of the same relation
                groups = {}
                for graph in graphs:
                    for (u, v, k, d) in graph.edges(data=True, keys=True):
                        if d[k] not in groups:
                            groups[d[k]] = set()
                        groups[d[k]].add(u)
                        groups[d[k]].add(v)
                return (relation, groups)
                   

def find_translate_arguments(relation_graph: RelationGraph, object: ARC_Object) -> Tuple[str, str]:
    '''
    Uses the graph to determine how much to translate the input to get the output
    Assume (for now) that there exists a bijection between the input and output objects as well as between respective inputs/outputs amongst examples
    This means: every problem has the same number of objects and the same relations between them
    Later we can work on clustering 'similar objects' to reduce it to a bijection
    '''
    # assume object is not translated off screen as otherwise a bijection wouldnt be found :P
    obj_id = relation_graph.objects_to_id[object]
    # example_number = obj_id.split('_')[1]
    # grid_dim_x = relation_graph.objects[0][0][0].parent.grid.shape[1]
    # grid_dim_y = relation_graph.objects[0][0][0].parent.grid.shape[0]
    # max_dist_left = object.top_left[1]
    # max_dist_right = grid_dim_x - object.top_left[1] - object.width
    # max_dist_up = object.top_left[0]
    # max_dist_down = grid_dim_y - object.top_left[0] - object.height
    # options_x = set(range(-max_dist_left, max_dist_right + 1))
    # options_y = set(range(-max_dist_up, max_dist_down + 1))
    
        
    # find bijection relation across io and across inputs
    relation, io_mapping = relation_graph.find_io_pairs()
    assert relation, 'No io bijections found'
    target_id = io_mapping[obj_id]
    target_obj = relation_graph.id_to_objects[target_id]
    
    
    relation, related_input_examples = relation_graph.find_input_pairs()
    print(related_input_examples)
    if relation:
        for _, group in related_input_examples.items():
            if obj_id in group:
                related_ids = list(group)
                related_objects = [relation_graph.id_to_objects[id] for id in group]
                target_objects = [relation_graph.id_to_objects[io_mapping[id]] for id in group]
                break
    else:
        related_ids = [obj_id]
        related_objects = [object]
        target_objects = [target_obj]
    # find distance to target object(s)
    dist_x = [target_objects[n].top_left[1] for n in range(len(related_objects))]
    dist_y = [target_objects[n].top_left[0] for n in range(len(related_objects))]
    
    # get index of current object in the list of related objects
    idx = related_ids.index(obj_id)
    
    # loop through properties in order of deemed relevance and find one that matches the distance to the target object(s) 
    y_formula = []
    x_formula = []
    for prop in [get_latitude, get_longitude]:
        # get a property of an object in the input object example number
        for context in relation_graph.objects[int(related_ids[idx].split("_")[1][-1])][0]:
            context_id = relation_graph.objects_to_id[context]
            # check if this property works across all examples
            check_add_y = 0 # check when you add it
            check_sub_y = 0 # check when you subtract it
            print('context obj id: ', context_id)
            for i, robj_id in enumerate(related_ids):
                print('\t related object: ',robj_id)
                # get the same context object in the example we are checking
                c_obj_id = -1
                for _, group in related_input_examples.items():
                    if context_id in group:
                        c_obj_id = [id for id in group if robj_id.split('_')[1] == id.split('_')[1]][0]   
                print('\t\t context for related:',end=" ")
                if c_obj_id != -1:
                    print(c_obj_id, ' prop val', prop(relation_graph.id_to_objects[c_obj_id]),' dist ', dist_y[i])
                else:
                    print('didnt find context')
                if c_obj_id != -1 and prop(relation_graph.id_to_objects[c_obj_id]) == dist_y[i]:
                    check_add_y += 1
                elif c_obj_id != -1 and prop(relation_graph.id_to_objects[c_obj_id]) == -dist_y[i]:
                    check_sub_y += 1
                else:
                    break
            if check_add_y == len(related_ids):
                y_formula.append(('+', prop.__name__, context_id))
            if check_sub_y == len(related_ids):
                y_formula.append(('-', prop, context))
                
    for prop in [get_latitude, get_longitude]:
        # get a property of an object in the input object example number
        for context in relation_graph.objects[int(related_ids[idx].split("_")[1][-1])][0]:
            context_id = relation_graph.objects_to_id[context]
            # check if this property works across all examples
            check_add_y = 0 # check when you add it
            check_sub_y = 0 # check when you subtract it
            print('context obj id: ', context_id)
            for i, robj_id in enumerate(related_ids):
                print('\t related object: ',robj_id)
                # get the same context object in the example we are checking
                c_obj_id = -1
                for _, group in related_input_examples.items():
                    if context_id in group:
                        c_obj_id = [id for id in group if robj_id.split('_')[1] == id.split('_')[1]][0]   
                print('\t\t context for related:',end=" ")
                if c_obj_id != -1:
                    print(c_obj_id, ' prop val', prop(relation_graph.id_to_objects[c_obj_id]),' dist ', dist_y[i])
                else:
                    print('didnt find context')
                if c_obj_id != -1 and prop(relation_graph.id_to_objects[c_obj_id]) == dist_x[i]:
                    check_add_y += 1
                elif c_obj_id != -1 and prop(relation_graph.id_to_objects[c_obj_id]) == -dist_x[i]:
                    check_sub_y += 1
                else:
                    break
            if check_add_y == len(related_ids):
                x_formula.append(('+', prop.__name__, context_id))
            if check_sub_y == len(related_ids):
                x_formula.append(('-', prop, context))
                
    return (y_formula, x_formula)
        
        
                    
# filter objects (List -> smaller List or single object)
def filter_by_color(objects: List[ARC_Object], color: int) -> List[ARC_Object]:
    """
    Filter a set of objects based on their color.
    
    """
    return [obj for obj in objects if np.all(obj.grid == color)]

def most_common(objs: list[ARC_Object]) -> ARC_Object:
    try:
        unique, count = np.unique([o.grid for o in objs], axis=0, return_counts=True)
        image = unique[np.argmax(count)]
        return ARC_Object(image, np.ones_like(image))
    except:
        return None

# filter a set of objects based on their properties
def filter_by_shape(objects: List[ARC_Object], target: ARC_Object) -> List[ARC_Object]:
    """
    Filter a set of objects based on their shape.
    
    """
    return [obj for obj in objects if (obj.grid !=0) == (target.grid !=0)]

def filter_by_size(objects: List[ARC_Object], target: ARC_Object) -> List[ARC_Object]:
    """
    Filter a set of objects based on their size (number of pixels).
    
    """
    return [obj for obj in objects if obj.active_pixels == target.active_pixels]

def lattitude_dist(obj1: ARC_Object, obj2: ARC_Object) -> int:
    """
    Returns the difference in height locations of the top-left corners of two objects.
    """
    return abs(obj1.top_left[0] - obj2.top_left[0])

def longitude_dist(obj1: ARC_Object, obj2: ARC_Object) -> isinstance:
    """
    Returns the difference in width locations of the top-left corners of two objects.
    """
    return abs(obj1.top_left[1] - obj2.top_left[1])

def get_width(obj: ARC_Object) -> int:
    """
    Get the width of the object.
    
    """
    return obj.width

def get_height(obj: ARC_Object) -> int:
    """
    Get the height of the object.
    """
    return obj.height

def get_latitude(obj: ARC_Object) -> int:
    '''
    Get the latitude position of the object in the parent grid
    '''
    return obj.top_left[0]

def get_longitude(obj: ARC_Object) -> int:
    '''
    Get the longitude position of the object in the parent grid
    '''
    return obj.top_left[1]

# retrieve useful integer properties of objects
def get_color(obj: ARC_Object) -> int:
    """
    Get the color of the object.
    np.unique[1] returns the unique values in the object's grid, excluding 0.
    """
    return np.unique(obj.grid)[1]

def get_size(obj: ARC_Object) -> int:
    """
    Get the size of the object (number of pixels).
    
    """
    return np.sum(obj.grid != 0)

def count_objects(objects: List[ARC_Object]) -> int:
    """
    Count the number of objects in the list.
    
    """
    return len(objects)

def count_mask(mask: np.ndarray) -> int:
    """
    Count the number of pixels in the mask.
    
    """
    return np.sum(mask != 0)

# retrieve useful shape properties of objects
def detect_symmetry(obj: ARC_Object) -> np.ndarray:
    matrix = obj.grid
    rows, cols = matrix.shape
    
    # Check for symmetry
    horizontal_symmetry = np.all(matrix == matrix[::-1])
    vertical_symmetry = np.all(matrix == matrix[:, ::-1])
    point_symmetry = np.all(matrix == np.flip(matrix, axis=(0, 1)))
    main_diagonal_symmetry = np.all(matrix == matrix.T)
    anti_diagonal_symmetry = np.all(matrix == np.flip(matrix.T, axis=1))
    
    if point_symmetry:
        # Generate mask for point symmetry (upper-left quadrant excluding center row and column if odd)
        half_rows = rows // 2
        half_cols = cols // 2
        mask = np.zeros_like(matrix, dtype=bool)
        mask[:half_rows, :half_cols] = True
        return mask
    
    elif horizontal_symmetry:
        # Generate mask for horizontal symmetry (upper half excluding center row if odd)
        half_rows = rows // 2
        mask = np.zeros_like(matrix, dtype=bool)
        mask[:half_rows, :] = True
        return mask
    
    elif vertical_symmetry:
        # Generate mask for vertical symmetry (left half excluding center column if odd)
        half_cols = cols // 2
        mask = np.zeros_like(matrix, dtype=bool)
        mask[:, :half_cols] = True
        return mask
    
    elif main_diagonal_symmetry:
        # Generate mask for main diagonal symmetry (upper triangle excluding diagonal)
        mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
        return mask

    elif anti_diagonal_symmetry:
        # Generate mask for anti-diagonal symmetry (lower triangle flipped along anti-diagonal)
        mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)[::-1, ::-1]
        return mask
    else:
        # No symmetry detected
        return np.zeros_like(matrix, dtype=bool)
    
def get_shape(obj: ARC_Object) -> np.ndarray:
    """
    Get the shape of the object as a bit mask.
    
    """
    return obj.mask

def get_contour(obj: ARC_Object) -> np.ndarray:
    """
    Get the contour of the object as a bit mask.
    
    """
    grid = obj.grid
    rows, cols = grid.shape
    contour_mask = np.zeros_like(grid)
    
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] != 0:
                # Check the neighbors
                neighbors = [
                    (i-1, j),  # Up
                    (i+1, j),  # Down
                    (i, j-1),  # Left
                    (i, j+1)   # Right
                ]
                for x, y in neighbors:
                    if x < 0 or x >= rows or y < 0 or y >= cols or grid[x, y] == 0:
                        contour_mask[i, j] = 1
                        break
    
    return contour_mask

# retrieve useful boolean properties of objects
def is_color(obj: ARC_Object, color: int) -> bool:
    return np.all(obj.grid == color)

def isAdjacent(obj1: ARC_Object, obj2: ARC_Object) -> bool:
    mask1 = obj1.grid != 0
    mask2 = obj2.grid != 0
    for i in range(obj1.height):
        for j in range(obj1.width):
            if mask1[i, j]:
                if i > 0 and mask2[i - 1, j]:
                    return True
                if i < obj1.height - 1 and mask2[i + 1, j]:
                    return True
                if j > 0 and mask2[i, j - 1]:
                    return True
                if j < obj1.width - 1 and mask2[i, j + 1]:
                    return True
    return False

def getOverlap(obj1: ARC_Object, obj2: ARC_Object) -> bool:
    mask1 = obj1.grid != 0
    mask2 = obj2.grid != 0
    return mask1 & mask2

def dominant_color(obj: ARC_Object) -> int:
    flattened = obj.grid.flatten()
    color_counts = Counter(flattened[flattened != 0])
    
    if color_counts:
        dominant = color_counts.most_common(1)[0][0]
        return dominant
    else:
        return -1