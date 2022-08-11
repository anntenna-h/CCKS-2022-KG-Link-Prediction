import os
import errno
from pathlib import Path
import pickle
import numpy as np
from collections import defaultdict

DATA_PATH = '../../data'

def prepare_dataset(path, name):
    files = ['train', 'valid', 'test']
    left_entities, right_entities, relations = set(), set(), set()
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        for line in to_read.readlines():
            lhs, rel, rhs = line.strip().split('\t')
            left_entities.add(lhs)
            right_entities.add(rhs)
            relations.add(rel)
        to_read.close()
    
    right_entities.remove('ent_999999')
    
    left_entities_to_id = {x: i for (i, x) in enumerate(sorted(left_entities))} 
    right_entities_to_id = {x: i for (i, x) in enumerate(sorted(right_entities))}
    relations_to_id = {x: i for (i, x) in enumerate(sorted(relations))}
    print("{} left entities, {} right entities and {} relations".format(len(left_entities), len(right_entities), len(relations)))
    n_relations = len(relations)
    n_left_entities = len(left_entities)
    n_right_entities = len(right_entities)

    if not os.path.exists(os.path.join(DATA_PATH, name)):
        os.makedirs(os.path.join(DATA_PATH, name))
    # write ent to id / rel to id
    for (dic, f) in zip([left_entities_to_id, right_entities_to_id, relations_to_id], ['lhs_id', 'rhs_id', 'rel_id']):
        ff = open(os.path.join(DATA_PATH, name, f), 'w+')
        for (x, i) in dic.items():
            ff.write("{}\t{}\n".format(x, i))
        ff.close()

    # map train/test/valid with the ids
    rhs_id_tot = set([])
    lhs_id_tot = set([])
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        examples = []

        for line in to_read.readlines():
            lhs, rel, rhs = line.strip().split('\t')
            try:
                examples.append([left_entities_to_id[lhs], relations_to_id[rel], right_entities_to_id[rhs]])
                rhs_id_tot.add(right_entities_to_id[rhs])
                lhs_id_tot.add(left_entities_to_id[lhs])
                                
            except KeyError: 
                examples.append([left_entities_to_id[lhs], relations_to_id[rel], -1])
                continue
        out_1 = open(Path(DATA_PATH) / name / (f + '.pickle'), 'wb')
        pickle.dump(np.array(examples).astype('int64'), out_1)
        out_1.close()
    
    print('Number of tail entities =',len(rhs_id_tot))                         
    out_2 = open(Path(DATA_PATH) / name / 'lhs_id_tot.pickle', 'wb')
    pickle.dump(np.array(list(lhs_id_tot)).astype('int64'), out_2)
    out_2.close()
    out_3 = open(Path(DATA_PATH) / name / 'rhs_id_tot.pickle', 'wb')
    pickle.dump(np.array(list(rhs_id_tot)).astype('int64'), out_3)
    out_3.close()  

    print("creating filtering lists")

    # create filtering files
    to_skip = defaultdict(set)
    for f in files:
        examples = pickle.load(open(Path(DATA_PATH) / name / (f + '.pickle'), 'rb'))
        for lhs, rel, rhs in examples:
            if rhs != 'ent_999999':
                to_skip[(lhs, rel)].add(rhs)
            
    to_skip_final = {}
    for k, v in to_skip.items():
        to_skip_final[k] = sorted(list(v))

    out = open(Path(DATA_PATH) / name / 'to_skip.pickle', 'wb')
    pickle.dump(to_skip_final, out)
    out.close()


if __name__ == "__main__":
    datasets = ['OpenBG-IMG']
    for d in datasets:
        print("Preparing dataset {}".format(d))
        prepare_dataset(os.path.join(os.path.dirname(os.path.realpath(DATA_PATH)), 'data', d), d)