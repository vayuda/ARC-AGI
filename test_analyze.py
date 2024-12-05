import numpy as np
from source import *
from torch.utils.data import DataLoader

if __name__ == '__main__':
    train_dataset = load_dataset(mode='evaluation')
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    problems = dict()

    for _, data in enumerate(train_dataloader):
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

    count = 0
    for pid, prob in problems.items():
        analyze = dict()

        for i in range(len(prob['train'])):
            analyze[f'ex_{i}'] = dict()
            # prob['train'][f'ex_{i}']['input'].plot_grid()
            # prob['train'][f'ex_{i}']['output'].plot_grid()
            in_obj = prob['train'][f'ex_{i}']['input']
            out_obj = prob['train'][f'ex_{i}']['output']
            in_to_out = CompareObjects(in_obj, out_obj)
            seg_in = ListProperties(prob['train'][f'ex_{i}']['extracted'][0])
            seg_out = ListProperties(prob['train'][f'ex_{i}']['extracted'][1])
            out_to_seg_in = CompareObjectList(output_object, seg_in)
            seg_in_to_out = [CompareObjects(i, out_obj) for i in prob['train'][f'ex_{i}']['extracted'][0]]
            seg_in_to_seg_out = [[CompareObjects(i, j) for i in prob['train'][f'ex_{i}']['extracted'][0]] for j in prob['train'][f'ex_{i}']['extracted'][1]]
            analyze[f'ex_{i}']['in_to_out'] = in_to_out.guess_transform()
            analyze[f'ex_{i}']['out_to_seg_in'] = out_to_seg_in.guess_transform()
            analyze[f'ex_{i}']['seg_in_to_out'] = [i.guess_transform() for i in seg_in_to_out]
            # if len(analyze[f'ex_{i}']['seg_in_to_out']) > 0:
            #     print(f'Need analyze: {pid}')
            analyze[f'ex_{i}']['seg_in_to_seg_out'] = [[i.guess_transform() for i in j] for j in seg_in_to_seg_out]

        test_out = solve(prob, analyze)
        if same_obj(test_out, prob['test']['output']):
            print(pid)
            # test_out.plot_grid()
            count += 1
    print(count)