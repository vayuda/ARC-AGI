import numpy as np
from source import *
from torch.utils.data import DataLoader

if __name__ == '__main__':
    train_dataset = load_dataset(mode='training')
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    problems = build_problems_dict(train_dataloader)

    count = 0
    for pid, prob in problems.items():
        analyze = build_analyze(prob)
        sln = solve(prob, analyze)
        if same_obj(sln, prob['test']['output']):
            print(pid)
            count += 1
    print(count)