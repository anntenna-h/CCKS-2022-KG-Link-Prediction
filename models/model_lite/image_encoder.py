import torch.nn
import torchvision.models as models
from torch.autograd import Variable
import torch.cuda
import torchvision.transforms as transforms
from PIL import Image
import pickle
import pytorch_pretrained_vit
# import sys
# sys.path.append("../../../autodl-nas")
# from pytorch_pretrained_vit.model import ViT

import os
import imagehash
from tqdm import tqdm
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

class ImageEncoder():
    
    TARGET_IMG_SIZE = 224
    img_to_tensor = transforms.ToTensor()     
    Normalizer = transforms.Normalize((0.5,), (0.5,))
    

    @staticmethod 
    def get_embedding(self): 
        pass

    
    def extract_feature(self, base_path):
        self.model.eval()         
        # -------------------------------------------------------------------------------------------
        # 遍历图片，保存'ent_000905': '../../data/OpenBG-IMG/images/ent_000905/image_0.jpg' 这样的dict
        # -------------------------------------------------------------------------------------------
        best_imgs = {}
        ents = os.listdir(base_path)
        pbar = tqdm(total=len(ents))
        
        while len(ents)>0:
            ent=ents.pop() 
            if 'DS_' in ent:
                continue 
            imgs = os.listdir(base_path + ent + '/') 
            n_img=len(imgs) 
            
            if n_img == 0:
                pbar.update(1)
                continue
             
            best_imgs[ent] = base_path + ent + '/' + imgs[0]
            pbar.update(1)
        pbar.close()
        
        # -----------
        # 喂图片进模型
        # -----------
        dic = {}

        img_paths = list(best_imgs.values())
        pbart = tqdm(total=len(img_paths))
        while len(img_paths) > 0:
            ents_50 = []
            ents_50_ok = [] 
            for i in range(5): 
                if len(img_paths) > 0:
                    ent = img_paths.pop()
                    try:
                        ents_50.append(ent)
                    except Exception as e: 
                        print(ent, e)
                        continue

            tensors = []
            for imgpath in ents_50:
                try:
                    img = Image.open(imgpath).resize((384, 384))
                except Exception as e:
                    print(e)
                    continue
                img_tensor = self.img_to_tensor(img)
                img_tensor = self.Normalizer(img_tensor)
                

                if img_tensor.size()[0] == 3: 
                    tensors.append(img_tensor) 
                    ents_50_ok.append(imgpath) 


            tensor = torch.stack(tensors, 0)
            tensor = tensor.cuda()
           
            result = self.model(Variable(tensor)) 
            result_npy = result.data.cpu().numpy()
            for i in range(len(result_npy)):
                dic[ents_50_ok[i]] = result_npy[i] # dic: key: 图的路径；value：图的嵌入向量
            pbart.update(5)

        pbart.close()
        return dic

class VisionTransformer(ImageEncoder):
    def __init__(self):
        super(VisionTransformer, self).__init__()
        # self.model = ViT('B_16_imagenet1k', pretrained=True, weights_path='../../data/B_16_imagenet1k.pth')
        self.model = ViT('B_16_imagenet1k', pretrained=True)
        
    def get_embedding(self,base_path):
        self.model.eval()
        self.model.cuda()
        self.d=self.extract_feature(base_path) 
        return self.d

    def save_embedding(self,output_file):
        with open(output_file, 'wb') as out:
            pickle.dump(self.d, out)


def get_img_vec_array(img_vec_path, output_file, dim=1000):
    
    img_vec = pickle.load(open(img_vec_path,'rb'))
    img_vec = {k.split('/')[-2]:v for k,v in img_vec.items()} # key - 实体编码； value - 图片嵌入向量
    
    f=open('../../data/OpenBG-IMG/lhs_id', 'r')
    Lines=f.readlines()
    
    img_vec_array=[]
    for l in Lines:
        ent, id = l.strip().split()
        
        if ent.replace('/','.') in img_vec.keys():
            img_vec_array.append(img_vec[ent.replace('/','.')])
        else:
            img_vec_array.append([0 for i in range(dim)])
    img_vec_by_id = np.array(img_vec_array) # key - 头实体id； value - 图片嵌入向量
    
    out = open(output_file,'wb')
    pickle.dump(img_vec_by_id,out)
    out.close()

if __name__ == '__main__':

#     model = VisionTransformer()
#     base_path = '../../data/images/'
#     model.get_embedding(base_path)
#     model.save_embedding('../../data/OpenBG-IMG/openbg_vit_best_img_vec.pickle')
    
    get_img_vec_array(
                      img_vec_path='../../data/OpenBG-IMG/openbg_vit_best_img_vec.pickle', 
                      output_file='../../data/OpenBG-IMG/img_vec_id_openbg_vit.pickle')