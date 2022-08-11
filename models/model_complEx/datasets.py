from pathlib import Path
import pickle
from typing import Dict, Tuple, List
import numpy as np
import torch
import os
from model import KBCModel

DATA_PATH=Path('../../data')

class Dataset(object):
    def __init__(self, name: str):
        self.root = DATA_PATH / name

        self.data = {}
        for f in ['train', 'test', 'valid']:
            in_file = open(str(self.root / (f + '.pickle')), 'rb')
            self.data[f] = pickle.load(in_file)
            in_file.close()
        maxis = np.max(self.data['train'], axis=0)
        self.n_predicates = int(maxis[1] + 1)
        
        res = []
        for f in ['lhs_id_tot', 'rhs_id_tot']:
            file = open(str(self.root / (f + '.pickle')), 'rb')
            dic = pickle.load(file)
            res.append(np.max(dic))
            file.close()
        self.n_lhs = res[0] + 1
        self.n_rhs = res[1] + 1
        
        inp_f = open(str(self.root / f'to_skip.pickle'), 'rb')
        self.to_skip: Dict[Tuple[int, int], List[int]] = pickle.load(inp_f)
        inp_f.close()

    def get_examples(self, split):
        return self.data[split]

    def eval(self, 
             model: KBCModel, 
             split: str, 
             n_queries: int = -1, 
             at: Tuple[int] = (1, 3, 10)
             ):
        if not os.path.exists('../../scores/'):
            os.mkdir('../../scores/')
        test = self.get_examples(split) 
        examples = torch.from_numpy(test.astype('int64')).cuda()

        mean_reciprocal_rank = {}
        mean_rank = {}
        hits_at = {}
        
        q = examples.clone()
        if n_queries > 0: 
            permutation = torch.randperm(len(examples))[:n_queries] 
            q = examples[permutation]
    
        ranks = model.get_ranking(q, self.to_skip, batch_size=500)

        mean_rank = torch.mean(ranks).item() 
        mean_reciprocal_rank = torch.mean(1. / ranks).item() 
        hits_at = torch.FloatTensor((list(map(
            lambda x: torch.mean((ranks <= x).float()).item(),
            at
        )))) 

        return mean_reciprocal_rank, mean_rank, hits_at

    def predict(self, 
                model: KBCModel, 
                split: str,
                n_queries: int = -1, 
                at: Tuple[int] = (1, 3, 10)
                ):
        if not os.path.exists('../../scores/'):
            os.mkdir('../../scores/')
        test = self.get_examples(split) 
        examples = torch.from_numpy(test.astype('int64')).cuda()

        mean_reciprocal_rank = {}
        mean_rank = {}
        hits_at = {}

        q = examples.clone()
        if n_queries > 0:
            permutation = torch.randperm(len(examples))[:n_queries]
            q = examples[permutation] 
            
        model.get_ranking(q, self.to_skip, batch_size=500, predict = True)
        return

    def get_shape(self):
        return self.n_lhs, self.n_predicates, self.n_rhs