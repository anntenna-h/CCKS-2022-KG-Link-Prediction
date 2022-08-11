import argparse
from typing import Dict
import os
import torch
from torch import optim
import time
import numpy as np
import random
from datasets import Dataset
from model import ComplEx
from regularizers import F2, N3
from optimizers import KBCOptimizer

def avg_both(mrrs: Dict[str, float], mrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
    mr = mrs
    mrr = mrrs
    h = hits
    return {'MRR': mrr, 'MR': mr, 'hits@[1,3,10]': h}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

setup_seed(666)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True



# ----
# CFG
# ----

parser = argparse.ArgumentParser(
    description="Relational learning contraption"
)


regularizers = ['N3', 'F2']
parser.add_argument(
    '--regularizer', choices=regularizers, default='N3',
    help="Regularizer in {}".format(regularizers)
)

optimizers = ['Adagrad', 'Adam', 'SGD']
parser.add_argument(
    '--optimizer', choices=optimizers, default='Adagrad',
    help="Optimizer in {}".format(optimizers)
)

parser.add_argument(
    '--max_epochs', default=50, type=int,
    help="Number of epochs."
)

parser.add_argument(
    '--valid', default=1, type=float,
    help="Number of epochs before valid."
)

parser.add_argument(
    '--rank', default=1000, type=int,
    help="Factorization rank."
)

parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Factorization rank."
)

parser.add_argument(
    '--reg', default=0, type=float,
    help="Regularization weight"
)

parser.add_argument(
    '--init', default=1e-3, type=float,
    help="Initial scale"
)

parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)

parser.add_argument(
    '--decay1', default=0.9, type=float,
    help="decay rate for the first moment estimate in Adam"
)

parser.add_argument(
    '--decay2', default=0.999, type=float,
    help="decay rate for second moment estimate in Adam"
)

parser.add_argument(
    '--alpha', default=0.999, type=float,
    help="weight for image embeddings"
)

parser.add_argument(
    '--constant', default=False, action="store_true",
    help="whether to use single constant for weight of image embeddings"
)


args = parser.parse_args()
print(args)


# --------
# trainer
# --------

datasets = ['OpenBG-IMG']
dataset = Dataset(datasets[0])
examples = torch.from_numpy(dataset.get_examples('train').astype('int64'))
print(dataset.get_shape())

model = ComplEx(dataset.get_shape(), rank = args.rank, init_size = args.init,
                alpha = args.alpha, constant = args.constant, 
                sig_alpha='../../data/OpenBG-IMG/alphas.pickle')

regularizer = {
    'F2': F2(args.reg),
    'N3': N3(args.reg),
}[args.regularizer]

device = 'cuda'
model.to(device)

optim_method = {
    'Adagrad': lambda: optim.Adagrad(model.parameters(), lr=args.learning_rate),
    'Adam': lambda: optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.decay1, args.decay2)),
    'SGD': lambda: optim.SGD(model.parameters(), lr=args.learning_rate)
}[args.optimizer]()

optimizer = KBCOptimizer(model, regularizer, optim_method, args.batch_size)

cur_loss = 0
curve = {'train': [], 'valid': [], 'test': []}
for e in range(args.max_epochs):
    cur_loss = optimizer.epoch(examples)

    if (e + 1) % args.valid == 0: 
        valid, train = [
            avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000)) 
            for split in ['valid', 'train']
        ]
        
        curve['valid'].append(valid)
        curve['train'].append(train)

        print("\t TRAIN: ", train)
        print("\t VALID : ", valid)

if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints/')

PATH = './checkpoints/model.ckpt'
torch.save(model.state_dict(), PATH)

model.load_state_dict(torch.load(PATH))

model.eval()
dataset.predict(model, 'test', -1)