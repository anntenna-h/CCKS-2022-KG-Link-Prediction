from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import torch
from torch import nn
import pickle
import torch.nn.functional as F
import numpy as np

    
class KBCModel(nn.Module, ABC): 
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    def get_ranking(self, 
            queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, 
            chunk_size: int = -1, 
            predict = False
    ):
        scores_tot = torch.zeros([0, 6676]).cuda()
        if chunk_size < 0:
            chunk_size = self.sizes[2] 
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0 
            while c_begin < self.sizes[2]:
                b_begin = 0 
                while b_begin < len(queries):
                    
                    these_queries = queries[b_begin:b_begin + batch_size]
                    rhs = self.get_rhs(c_begin, chunk_size) 
                    q = self.get_queries(these_queries) 
                    scores = q @ rhs # (batch_size, 6676)
 
                    if not predict:
                        targets = self.score(these_queries) 
                        for i, query in enumerate(these_queries):
                            filter_out = filters[(query[0].item(), query[1].item())] 
                            if query[2].item() != -1:
                                filter_out += [query[2].item()] 
                            if chunk_size < self.sizes[2]:
                                filter_in_chunk = [int(x - c_begin) for x in filter_out
                                                   if c_begin <= x < c_begin + chunk_size]
                                scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                            else:
                                scores[i, torch.LongTensor(filter_out)] = -1e6
                        ranks[b_begin:b_begin + batch_size] += torch.sum((scores >= targets).float(), dim=1).cpu()
                    else:
                        scores_tot = torch.cat([scores_tot, scores])
                    b_begin += batch_size
                c_begin += chunk_size
        if not predict:
            return ranks
        else:
            scores_tot.cpu()
            out = open('../../scores/scores.pickle', 'wb')
            pickle.dump(scores_tot, out)
            out.close()
            return

        
class ComplEx(KBCModel):
    def __init__(self, 
                 sizes: Tuple[int, int, int],  # (n_lhs, n_rel, n_rhs)
                 rank, alpha, constant, init_size,
                 img_info='../../data/OpenBG-IMG/img_vec_id_openbg_vit.pickle',
                 lhs_tot_id='../../data/OpenBG-IMG/lhs_id_tot.pickle', 
                 sig_alpha = None
    ):
        super(ComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.alpha = alpha
        self.constant = constant
        
        self.lhs_embedding = nn.Embedding(sizes[0], rank * 2, sparse=True)
        self.rel_embedding = nn.Embedding(sizes[1], rank * 2, sparse=True)
        self.rhs_embedding = nn.Embedding(sizes[2], rank * 2, sparse=True)

        self.lhs_embedding.weight.data *= init_size # (|lhs|, 2000) / （21234, 2000)
        self.rel_embedding.weight.data *= init_size # (|rel|, 2000) / （136, 2000)
        self.rhs_embedding.weight.data *= init_size # (|rhs|, 2000) / （6677, 2000)
        
        if not self.constant:
            self.alpha = pickle.load(open(sig_alpha, 'rb')).cuda()
        else:
            self.alpha = nn.Parameter(torch.tensor(self.alpha), requires_grad=False)  
        
        self.img_dimension = 1000
        self.img_info = pickle.load(open(img_info, 'rb'))
        self.img_vec = torch.from_numpy(self.img_info).float().cuda() 
        
        self.post_mats = nn.Parameter(torch.Tensor(self.img_dimension, 2 * rank), requires_grad=True)
        nn.init.xavier_uniform(self.post_mats)
        self.post_bias = nn.Parameter(torch.Tensor(1,2 * rank), requires_grad=True)
        nn.init.xavier_uniform(self.post_bias)
        
        self.lhs_tot_id = torch.from_numpy(pickle.load(open(lhs_tot_id, 'rb'))).cuda()

    def score(self, x):
        img_embeddings = self.img_vec.mm(self.post_mats) + self.post_bias 
        if not self.constant:
            
            lhs = (1 - self.alpha[(x[:, 0])]) * self.lhs_embedding(x[:, 0]) + self.alpha[(x[:, 0])] * img_embeddings[(x[:, 0])]
        else:
            embedding = (1 - self.alpha) * self.lhs_embedding.weight + self.alpha * img_embeddings
            lhs = embedding[(x[:, 0])]
            
        rel = self.rel_embedding(x[:, 1])
        rhs = self.rhs_embedding(x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:] # ((batch_size, 1000),（batch_size, 1000))
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        
        score = torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1, keepdim=True
        )
        # (batch_size, 1)
        return score

    def forward(self, x):
        img_embeddings = self.img_vec.mm(self.post_mats) + self.post_bias
        if not self.constant:
            lhs = (1 - self.alpha[(x[:, 0])]) * self.lhs_embedding(x[:, 0]) + self.alpha[(x[:, 0])] * img_embeddings[(x[:, 0])] #（|x|， 2000）
        else:
            embedding = (1 - self.alpha) * self.lhs_embedding.weight + self.alpha * img_embeddings
            lhs = embedding[(x[:, 0])]
            
        rel = self.rel_embedding(x[:, 1])
        rhs = self.rhs_embedding.weight

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        return  (
                    (lhs[0] * rel[0] - lhs[1] * rel[1]) @ rhs[0].transpose(0, 1) +
                    (lhs[0] * rel[1] + lhs[1] * rel[0]) @ rhs[1].transpose(0, 1)
                ), (
                    torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                    torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                    torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
                )

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.rhs_embedding.weight[chunk_begin:chunk_begin + chunk_size].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        img_embeddings = self.img_vec.mm(self.post_mats) + self.post_bias
        if not self.constant:             
            lhs = (1 - self.alpha[(queries[:, 0])]) * self.lhs_embedding(queries[:, 0]) + \
                self.alpha[(queries[:, 0])] * img_embeddings[(queries[:, 0])]
        else:
            embedding = (1 - self.alpha) * self.lhs_embedding.weight + self.alpha * img_embeddings
            lhs = embedding[(queries[:, 0])]
            
        rel = self.rel_embedding(queries[:, 1])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:] 
        rel = rel[:, :self.rank], rel[:, self.rank:]

        return torch.cat([
            lhs[0] * rel[0] - lhs[1] * rel[1],
            lhs[0] * rel[1] + lhs[1] * rel[0]
        ], 1)

