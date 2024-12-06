import numpy as np
from functools import reduce
from typing import List
from . import *
from phog.synthetic_generator import _flatten_objects

def same_grid(grid1, grid2):
    grid1_normalized = np.where(grid1 == 12, 0, grid1)
    grid2_normalized = np.where(grid2 == 12, 0, grid2)
    return np.array_equal(grid1_normalized, grid2_normalized)

def same_obj(obj1: ARC_Object, obj2: ARC_Object):
    if obj1 is None or obj2 is None:
        return False
    return same_grid(obj1.grid, obj2.grid)

def rot_flip(matrix1, matrix2):
    for k in range(4):
        rotated_matrix = np.rot90(matrix1, k=k)
        if same_grid(rotated_matrix, matrix2):
            return (k, -1)
        if same_grid(np.flip(rotated_matrix, axis=0), matrix2):
            return (k, 0)
        if same_grid(np.flip(rotated_matrix, axis=1), matrix2):
            return (k, 1)
        if same_grid(np.flip(rotated_matrix, axis=(0, 1)), matrix2):
            return (k, (0, 1))
    return None

class ListProperties:
    def __init__(self, objs: List[ARC_Object]):
        self.objs = sorted(objs, key=lambda o: o.top_left)
        self.num_objs = len(self.objs)
        self.most_common = most_common(self.objs)
        self.least_common = least_common(self.objs)
        self.majority = majority(self.objs)
        try:
            self.and_all = reduce(and_obj, self.objs)
        except:
            self.and_all = None
        try:
            self.or_all = reduce(or_obj, self.objs)
        except:
            self.or_all = None
        try:
            self.xor_all = reduce(xor_obj, self.objs)
        except:
            self.xor_all = None

class CompareObjects:
    def __init__(self, obj1: ARC_Object, obj2: ARC_Object):
        self.obj1 = obj1
        self.obj2 = obj2

        sym_mask1 = detect_symmetry(obj1)
        sym_mask2 = detect_symmetry(obj2)
        sym_grid1, sym_grid2 = None, None
        if np.any(sym_mask1):
            rows, cols = np.nonzero(sym_mask1)
            row_min, row_max = rows.min(), rows.max()
            col_min, col_max = cols.min(), cols.max()
            sym_grid1 = obj1.grid[row_min:row_max+1, col_min:col_max+1]
        if np.any(sym_mask2):
            rows, cols = np.nonzero(sym_mask2)
            row_min, row_max = rows.min(), rows.max()
            col_min, col_max = cols.min(), cols.max()
            sym_grid2 = obj2.grid[row_min:row_max+1, col_min:col_max+1]

        if sym_grid2 is not None and same_grid(obj1.grid, sym_grid2):
            self.sym = 1
        elif sym_grid1 is not None and same_grid(obj2.grid, sym_grid1):
            self.sym = 2
        else:
            self.sym = 0

        self.same_grid = same_obj(obj1, obj2)
        self.same_pos = obj1.top_left == obj2.top_left
        self.same_color = dominant_color(obj1) == dominant_color(obj2)
        self.same_size = (obj1.height == obj2.height) and (obj1.width == obj2.width)
        self.rot_flip = rot_flip(obj1.grid, obj2.grid)
    
    def guess_transform(self):
        if self.same_grid:
            return [lambda obj: recolor(obj, 12, 0)] # Swap all padding to black
        
        transformations = []
        if self.same_size and self.same_pos and not self.same_color:
            color1 = dominant_color(self.obj1)
            color2 = dominant_color(self.obj2)
            transformations.append(lambda obj: recolor(obj, color1, color2))
        
        if self.rot_flip is not None:
            rot_arg, flip_arg = self.rot_flip
            transformations.append(lambda obj: rotate(obj, rot_arg))
            transformations.append(lambda obj: flip(obj, flip_arg))
        
        return transformations
    
    def __str__(self):
        return f'same grid: {self.same_grid}, same pos: {self.same_pos}, same color: {self.same_color}, same size: {self.same_size}, rot flip: {self.rot_flip}, sym: {self.sym}'

class CompareObjectList:
    def __init__(self, obj: ARC_Object, lst: ListProperties):
        self.obj = obj
        self.lst = lst
        self.is_only = lst.num_objs == 1 and same_obj(obj, lst.objs[0])
        self.is_most_common = same_obj(obj, lst.most_common)
        self.is_least_common = same_obj(obj, lst.least_common)
        self.is_majority = same_obj(obj, lst.majority)
        self.is_and_all = same_obj(obj, lst.and_all)
        self.is_or_all = same_obj(obj, lst.or_all)
        self.is_xor_all = same_obj(obj, lst.xor_all)
    
    def guess_transform(self):
        if self.is_only:
            return lambda lst: lst[0]
        if self.is_most_common:
            return lambda lst: most_common(lst)
        if self.is_least_common:
            return lambda lst: least_common(lst)
        if self.is_majority:
            return lambda lst: majority(lst)
        if self.is_and_all:
            return lambda lst: reduce(and_obj, lst)
        if self.is_or_all:
            return lambda lst: reduce(or_obj, lst)
        if self.is_xor_all:
            return lambda lst: reduce(xor_obj, lst)
        return None
    
    def __str__(self):
        return f'is most common: {self.is_most_common}, is majority: {self.is_majority}, is and: {self.is_and_all}, is or: {self.is_or_all}, is xor: {self.is_xor_all}'

def build_problems_dict(dataloader):
    problems = dict()

    for _, data in enumerate(dataloader):
        id = data['id'][0]
        last_id = id
        train = data['train']
        test = data['test']
        problems[id] = dict()
        problems[id]['train'] = dict()
        seg_method = 'monochrome_contour'
        for i in range(len(train)):
            input_image = train[i]['input'].squeeze(0).numpy()
            output_image = train[i]['output'].squeeze(0).numpy()
            input_object = ARC_Object(image=input_image, mask=np.ones_like(input_image), parent=None)
            output_object = ARC_Object(image=output_image, mask=np.ones_like(output_image), parent=None)
            problems[id]['train'][f'ex_{i}'] = {'input': input_object, 'output': output_object}
            input_extraction = extract_objects(input_object, method=seg_method)
            output_extraction  = extract_objects(output_object, method=seg_method)
            if len(input_extraction) > 10 or len(output_extraction) > 10:
                seg_method = 'contour_scale'
                input_extraction = extract_objects(input_object, method=seg_method)
                output_extraction  = extract_objects(output_object, method=seg_method)
            # HACK HACK HACK
            if len(input_extraction) > 10 or len(output_extraction) > 10 or (seg_method == 'contour_scale' and len(input_extraction) <= 2 and len(output_extraction) <= 2):
                seg_method = 'color'
                input_extraction = extract_objects(input_object, method=seg_method)
                output_extraction  = extract_objects(output_object, method=seg_method)
            problems[id]['train'][f'ex_{i}']['extracted'] = (input_extraction, output_extraction)

        input_image = test[0]['input'].squeeze(0).numpy()
        output_image = test[0]['output'].squeeze(0).numpy()
        input_object = ARC_Object(image=input_image, mask=np.ones_like(input_image), parent=None)
        output_object = ARC_Object(image=output_image, mask=np.ones_like(output_image), parent=None)
        problems[id]['test'] = {'input': input_object, 'output': output_object}
        input_extraction = extract_objects(input_object, method=seg_method)
        output_extraction  = extract_objects(output_object, method=seg_method)
        problems[id]['test']['extracted'] = (input_extraction, output_extraction)
        problems[id]['seg_method'] = seg_method

    return problems

def build_analyze(prob):
    analyze = dict()

    for i in range(len(prob['train'])):
        analyze[f'ex_{i}'] = dict()
        in_obj = prob['train'][f'ex_{i}']['input']
        out_obj = prob['train'][f'ex_{i}']['output']
        in_to_out = CompareObjects(in_obj, out_obj)
        seg_in = ListProperties(prob['train'][f'ex_{i}']['extracted'][0])
        seg_out = ListProperties(prob['train'][f'ex_{i}']['extracted'][1])
        out_to_seg_in = CompareObjectList(out_obj, seg_in)
        seg_in_to_seg_out = [[CompareObjects(i, j) for i in prob['train'][f'ex_{i}']['extracted'][0]] for j in prob['train'][f'ex_{i}']['extracted'][1]]
        analyze[f'ex_{i}']['in_to_out'] = in_to_out.guess_transform()
        analyze[f'ex_{i}']['out_to_seg_in'] = out_to_seg_in.guess_transform()
        analyze[f'ex_{i}']['seg_in_to_seg_out'] = [[i.guess_transform() for i in j] for j in seg_in_to_seg_out]
    
    return analyze

def solve(prob, analyze):
    test_input = prob['test']['input']
    seg_in = extract_objects(test_input, method=prob['seg_method'])

    in_to_out = []
    out_to_seg_in = []
    seg_in_to_seg_out = []

    for key, val in analyze.items():
        for f in val['in_to_out']:
            in_to_out.append(f)
        if val['out_to_seg_in'] is not None:
            out_to_seg_in.append(val['out_to_seg_in'])
        for f_lst in val['seg_in_to_seg_out']:
            for f in f_lst:
                seg_in_to_seg_out.extend(f)

    for f in in_to_out:
        no_good = False
        out = f(test_input)
        for i in range(len(prob['train'])):
            in_obj = prob['train'][f'ex_{i}']['input']
            out_obj = prob['train'][f'ex_{i}']['output']
            train_out = f(in_obj)
            if not same_obj(train_out, out_obj):
                no_good = True
                break
        if not no_good:
            return out

    for f in out_to_seg_in:
        no_good = False
        out = f(seg_in)

        for i in range(len(prob['train'])):
            in_obj = prob['train'][f'ex_{i}']['input']
            out_obj = prob['train'][f'ex_{i}']['output']
            seg_train_in = prob['train'][f'ex_{i}']['extracted'][0]
            train_out = f(seg_train_in)
            if not same_obj(train_out, out_obj):
                no_good = True
                break
        if not no_good:
            return out

    transformed = []
    for o in seg_in:
        curr = deepcopy(o)
        for f in seg_in_to_seg_out:
            new = f(curr)
            if not same_obj(new, curr):
                curr = new
                break
        transformed.append(curr)
    if len(transformed) > 0:
        out = _flatten_objects(test_input, transformed)
        return out
    
    return None