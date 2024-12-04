import numpy as np
from source import *
from torch.utils.data import DataLoader
from phog.synthetic_generator import _flatten_objects
from solver.solver import solve
from tqdm import tqdm

if __name__ == '__main__':
    train_dataset = load_dataset(mode='evaluation')
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    problems = dict()

    for _, data in enumerate(train_dataloader):
        id = data['id'][0]
        last_id = id
        train = data['train']
        test = data['test']

        problems[id] = dict()
        relation_lst = []
        seg_method = 'monochrome_contour'

        for i in range(len(train)):
            input_image = train[i]['input'].squeeze(0).numpy()
            output_image = train[i]['output'].squeeze(0).numpy()

            input_object = ARC_Object(image=input_image, mask=np.ones_like(input_image), parent=None)
            output_object = ARC_Object(image=output_image, mask=np.ones_like(output_image), parent=None)
            problems[id][f'ex_{i}'] = {'input': input_object, 'output': output_object}

            input_extraction = extract_objects(input_object, method=seg_method)
            output_extraction  = extract_objects(output_object, method=seg_method)
            if len(input_extraction) > 10 or len(output_extraction) > 10:
                seg_method = 'color'
                input_extraction = extract_objects(input_object, method=seg_method)
                output_extraction  = extract_objects(output_object, method=seg_method)
        
            problems[id][f'ex_{i}']['extracted'] = (input_extraction, output_extraction)
            relation_lst.append(problems[id][f'ex_{i}']['extracted'])
    
        input_image = test[0]['input'].squeeze(0).numpy()
        output_image = test[0]['output'].squeeze(0).numpy()

        input_object = ARC_Object(image=input_image, mask=np.ones_like(input_image), parent=None)
        output_object = ARC_Object(image=output_image, mask=np.ones_like(output_image), parent=None)
        problems[id]['test'] = {'input': input_object, 'output': output_object}

        input_extraction = extract_objects(input_object, method=seg_method)
        output_extraction  = extract_objects(output_object, method=seg_method)
        problems[id]['test']['extracted'] = (input_extraction, output_extraction)

        try:
            problems[id]['relation_graph'] = RelationGraph(relation_lst)
        except:
            problems[id]['relation_graph'] = None
            print(f'Cannot build relation graph for {id}')

    count = 0

    for pid, prob in tqdm(problems.items()):
        if len(prob['ex_0']['extracted'][0]) > 10 or len(prob['ex_0']['extracted'][1]) > 10 or prob['relation_graph'] is None:
            continue
        solution = solve(prob['ex_0']['input'], prob['ex_0']['output'], prob['ex_0']['extracted'][0], prob['relation_graph'])
        if solution is not None:
            print(pid)
            count += 1

    print(count)