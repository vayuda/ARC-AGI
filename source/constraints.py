import numpy as np
from typing import List, Tuple
from .objects import ARC_Object
import networkx as nx
import matplotlib.pyplot as plt
from . import dsl
from collections import Counter

class RelationGraph():
    def __init__(self, objects: List[Tuple[List[ARC_Object], List[ARC_Object]]]):
        self.objects = objects
        self.id_to_object = {}
        self.object_to_id = {}
        self.nexamples = len(objects)
        self.graph = nx.MultiGraph()
        self.relation_types = ['color', 'shape', 'y_pos', 'x_pos']
        def add_relations(obj1, obj2):
            id1 = self.object_to_id[obj1]
            id2 = self.object_to_id[obj2]
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
                self.object_to_id[obj] = id
                self.id_to_object[id] = obj
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
                # print('IO Bijection found', relation)
                # create a dictionary of bijections
                mapping = {}
                for graph in graphs:
                    for (u, v, k, d) in graph.edges(data=True, keys=True):
                        if 'input' in u and 'output' in v:
                            mapping[u] = v
                return (relation, mapping)
    
    def find_input_groups(self, key_example='0'):
        '''
        finds mappings between inputs across exmaples for each relation type
        currently has a hard requirement that the number of objects in each example is the same
        later we can work on clustering similar objects to reduce it to a bijection
        another easier step is to store partial mappings (works for subset of examples) because the whole point of this is to reduce the search space 
        '''
        all_mappings = {}
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
                # print('Example Bijection found', relation)
                # group all objects of the same relation
                groups = {}
                mapping = {}
                for graph in graphs:
                    for (u, v, k, d) in graph.edges(data=True, keys=True):
                        if d[k] not in groups:
                            groups[d[k]] = set()
                        groups[d[k]].add(u)
                        groups[d[k]].add(v)
                # reformat groups so that key is object from example 0 and value is a set of objects from other examples
                for group in groups.values():
                    key_obj = [obj for obj in group if key_example in obj][0]
                    mapping[key_obj] = group
                all_mappings[relation] = mapping
        return all_mappings
    
    def find_important_objects(self, object_id):
        '''
        returns a tuple (output_object_id, (input_group_ids))
        
        '''
        io_relation, io_mapping = self.find_io_pairs()
        target_id = io_mapping[object_id]
        possible_mappings = self.find_input_groups(object_id.split('_')[1])
        for relation, input_groups in possible_mappings.items():
            if relation == io_relation:
                break
        return (target_id, input_groups[object_id])
                 
    def get_args(self, dsl_function, object)-> List[callable]:
        # simple heuristics
        if dsl_function == dsl.color:
            functions = []
            colors_in_problem = set([obj.color for id, obj in self.id_to_object.items() if "output" in id])
            for color in colors_in_problem:
                functions.append(lambda obj: dsl.color(obj, color))
                
            return functions
        
        if dsl_function == dsl.recolor:
            functions = []
            colors_in_input = set([obj.color for id, obj in self.id_to_object.items() if "input" in id])
            colors_in_output = set([obj.color for id, obj in self.id_to_object.items() if "output" in id])
            for color1 in colors_in_input:
                for color2 in colors_in_output:
                    functions.append(lambda obj: dsl.color(obj, color1, color2))
                
            return [lambda obj: dsl.recolor(obj, i, j) for i in range(1,10) for j in range(1,10)]
        
        if dsl_function == dsl.rotate:
            return [lambda obj: dsl.rotate(obj, i) for i in range(1,4)]
        
        # not sure if this is needed but just for standardization may be helpful
        if dsl_function == dsl.flip:
            return [dsl.flip]
        
        if dsl_function == dsl.delete:
            return [dsl.delete]
        
        # assumes a line is drawn between two objects or the borders of the grid
        if dsl_function == dsl.draw_line:
            coords = [coord for obj in self.id_to_object.values() for coord in (obj.E, obj.W, obj.N, obj.S)]
            coords.extend([0, object.parent.width-1, object.parent.height-1])
            coords = set(coords)
            colors = set([obj.color for id, obj in self.id_to_object.items() if "output" in id])
            functions = []
            for s0 in coords:
                for s1 in coords:
                    for e0 in coords:
                        for e1 in coords:
                            for color in colors:
                                start = [s0, s1]
                                end = [e0, e1]
                                if start != end:
                                    functions.append(lambda obj: dsl.draw_line(obj, start, end, 1))
            return functions
        # im  not sure if this actually works
        if dsl_function == dsl.translate:
            return self.find_translate_arguments(object)
        
        if dsl_function == dsl.tile:
            return self.find_tile_arguments(object)
        
    def find_tile_arguments(self, tile_obj: ARC_Object) -> List[callable]:
        x_scale = tile_obj.width
        y_scale = tile_obj.height
        n = self.object_to_id[tile_obj].split('_')[1]
        output_objects = [obj for id, obj in self.id_to_object.items() if f"output_{n}" in id]
        same_shape_objects = filter_by_shape(output_objects, tile_obj)
        same_shape_objects.sort(key=lambda x: abs(x.top_left[0] - tile_obj.top_left[0]) + abs(x.top_left[1] - tile_obj.top_left[1]))
        if len(same_shape_objects) > 1:
            directions = [[same_shape_objects[1].top_left[0] - same_shape_objects[0].top_left[0],
                           same_shape_objects[1].top_left[1] - same_shape_objects[0].top_left[1]]]
        else: # its doomed try everything...
            directions = [[0, x_scale], [y_scale, 0], [0, -y_scale], [-x_scale, 0], 
                          [x_scale, y_scale], [-x_scale, -y_scale], [x_scale, -y_scale], [-x_scale, y_scale]]
        
        ends = [0,1, len(same_shape_objects)]
        return [lambda obj, direction=direction: dsl.tile(obj, direction, end) for direction in directions for end in ends]
        
    def find_translate_arguments(self, object: ARC_Object) -> List[callable]:
        '''
        Uses the graph to determine how much to translate the input to get the output
        Assume (for now) that there exists a bijection between the input and output objects as well as between respective inputs/outputs amongst examples
        This means: every problem has the same number of objects and the same relations between them
        Later we can work on clustering 'similar objects' to reduce it to a bijection
        '''
        # assume object is not translated off screen as otherwise a bijection wouldnt be found :P
        id2object = self.id_to_object
        obj_id = self.object_to_id[object]
        example_id = obj_id.split('_')[1]
        # print(object.grid)
        # find corresponding object in output
        relation, io_mapping = self.find_io_pairs()
        assert relation, 'No io bijections found'
        target_id = io_mapping[obj_id]
        target_obj = id2object[target_id]
        
        # find corresponding objects in other examples
        possible_mappings = self.find_input_groups(key_example = example_id)
        # prefer when they are related by color
        for relation, input_groups in possible_mappings.items():
            if 'color' in relation:
                break
        
        # find x,y of target object for each related object across examples
        # print(input_groups)
        # print(io_mapping)
        target_x = {id: id2object[io_mapping[id]].top_left[1] for id in input_groups[obj_id]}
        target_y = {id: id2object[io_mapping[id]].top_left[0] for id in input_groups[obj_id]}
        # print(target_y, target_x)
        
        # loop through properties in order of deemed relevance and find one that matches the distance to the target object(s) 
        y_formula = []
        x_formula = []
        
        def validate_property(object_id, prop: callable, target_map, input_groups, argument):
            formula = []

            def create_translate_function(obj, offset, axis):
                if axis == 'y':
                    return lambda obj: dsl.translate(obj, (offset, 0))
                else:
                    return lambda obj: dsl.translate(obj, (0, offset))

            # for all possible objects in the current example
            for context_id in input_groups.keys():
                # print('context_id', context_id)
                check_add = 0
                # for all related objects in other examples
                for related_id in input_groups[object_id]:
                    # get the corresponding context object in the other example
                    related_context_ids = input_groups[context_id]
                    n = related_id.split('_')[1]
                    for related_context_id in related_context_ids:
                        if n in related_context_id:
                            break
                    # check if property matches the target
                    
                    if prop(id2object[related_context_id]) == target_map[related_id]:
                        # print('\t',prop.__name__, context_id, prop(id2object[related_context_id]), target_map[related_id])
                        check_add += 1
                    else:
                        break
                if check_add == len(input_groups[object_id]):
                    if argument == 'y':
                        offset = prop(id2object[context_id]) - id2object[object_id].top_left[0]
                        formula.append(create_translate_function(obj=None, offset=offset, axis='y'))
                    else:
                        offset = prop(id2object[context_id]) - id2object[object_id].top_left[1]
                        formula.append(create_translate_function(obj=None, offset=offset, axis='x'))
            return formula

        # try based on x/y position of other objects in the same example
        for prop in [get_N, get_E, get_S, get_W]:
            y_formula += validate_property(obj_id, prop, target_y, input_groups, 'y')
            x_formula += validate_property(obj_id, prop, target_x, input_groups, 'x')
        
        # WIP but maybe it works for other problems..
        # dist_props = [lambda x: object.top_left[0] - lattitude_dist(object, x),
        #             lambda x: object.top_left[0] + lattitude_dist(object, x),
        #             lambda x: object.top_left[1] - lattitude_dist(object, x),
        #             lambda x: object.top_left[1] + lattitude_dist(object, x),
        #             lambda x: object.top_left[0] - longitude_dist(object, x),
        #             lambda x: object.top_left[0] + longitude_dist(object, x),
        #             lambda x: object.top_left[1] - longitude_dist(object, x),
        #             lambda x: object.top_left[1] + longitude_dist(object, x)]
        
        # for prop in dist_props:
        #     y_formula += validate_property(obj_id, prop, target_y, input_groups, self)
        #     x_formula += validate_property(obj_id, prop, target_x, input_groups, self)
            
        # try based on a constant value
        # check if dist_y is constant across all examples
        if len(set(target_y.values())) == 1:
            y_formula.append(('constant', target_obj.top_left[0]))     
        
        # check if dist_x is constant across all examples
        if len(set(target_x.values())) == 1:
            x_formula.append(('constant', target_obj.top_left[1]))
        
        combined_formulas = []
        for y_func in y_formula:
            for x_func in x_formula:
                combined_formulas.append(lambda obj: x_func(y_func(obj)))
        return combined_formulas
        
    #check distance to borders for current object and related ones
                    
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
    return [obj for obj in objects if ((obj.grid !=0) == (target.grid !=0)).all()]

def filter_by_size(objects: List[ARC_Object], target: ARC_Object) -> List[ARC_Object]:
    """
    Filter a set of objects based on their size (number of pixels).
    
    """
    return [obj for obj in objects if obj.active_pixels == target.active_pixels]

def lattitude_dist(obj1: ARC_Object, obj2: ARC_Object) -> int:
    """
    Assuming we cannot have overlapping objects, 
    returns min(dist(o1.N-1,o2.S+1), dist(o1.S+1, o1.N-1))
    """
    if abs(obj1.N - obj2.S) < abs(obj2.N - obj1.S):
        return obj1.N - obj2.S
    else:
        return obj2.N - obj1.S

def longitude_dist(obj1: ARC_Object, obj2: ARC_Object) -> int:
    """
    Assuming we cannot have overlapping objects,
    returns min(dist(o1.E+1,o2.W-1), dist(o1.W-1, o1.E+1))
    """
    if abs(obj1.E - obj2.W) < abs(obj2.E - obj1.W):
        return obj1.E - obj2.W
    else:
        return obj2.E - obj1.W

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

def get_N(obj: ARC_Object) -> int:
    '''
    Get the north most y position of the object in the parent grid
    '''
    return obj.N

def get_S(obj: ARC_Object) -> int:
    '''
    Get the south most y position of the object in the parent grid
    '''
    return obj.S

def get_E(obj: ARC_Object) -> int:
    '''
    Get the east most x position of the object in the parent grid
    '''
    return obj.E

def get_W(obj: ARC_Object) -> int:
    '''
    Get the west most x position of the object in the parent grid
    '''
    return obj.W


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