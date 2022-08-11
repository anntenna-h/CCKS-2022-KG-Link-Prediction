# CCKS2022 多模态商品知识图谱链接预测 方案文档

by 整点薯条

# 解决方案

## 知识图谱模型

- ComplEx 

  $$\operatorname{Re}\left(\left\langle\mathbf{e}_{s}, \mathbf{w}_{r}, \overline{\mathbf{e}}_{o}\right\rangle\right)$$

- TuckER

  $$\mathcal{W} \times{ }_{1} \mathrm{e}_{s} \times_{2} \mathbf{w}_{r} \times_{3} \mathrm{e}_{o}$$

## 多模态信息嵌入

- 用ViT提取图片信息
- 用OCR和BERT提取文字信息

## 多模态融合

- 生成权重后相加

  ComplEx模型基础上将图片向量的权重分别设为0和1，再在验证集上，将得到的对应真实结果的$score_1$和$score_2$通过softmax求得图片向量的权重

- 注意力融合

  * 针对头实体和图片做注意力融合：将图片视作context sequence，针对头实体的embedding求注意力然后更新在头实体的embedding上，此处借鉴transformer的结构，采用多头多层进行头实体的注意力更新，此阶段主要是讲图片信息编码融合到头实体的embedding中，模型效果提升显著
  * 针对关系embedding和图片做注意力融合：同上，该阶段可以针对不同图片，让关系的embedding进行自适应的更新，模型更加精准

- 自适应融合

  把结构化嵌入向量、图片嵌入向量和文字嵌入向量三个模态分别通过一个线性层做了自适应的调整，此后拼接成一个向量。该向量分别过了两个线性层去提取信息，相加，然后再过了一个线性层转换成与关系、尾实体嵌入向量相同大小的向量。

## 损失函数

- CE_loss `nn.CrossEntropyLoss(reduction='mean')`

- BCE_loss `nn.BCELoss()`

  - 对于ComplEx，在参数较多的情况下，表现不如交叉熵损失。
  - 在加入注意力融合的TuckER模型中表现良好，收敛能得到单模型中最好的效果
  
  

# 运行

## 模型结果

> rank为实体的维度，dr为关系的维度

| 模型代号            | 具体方法                                            | 超参                                        | 设备                 | 时间   | 线上结果 |
| :------------------ | :-------------------------------------------------- | :------------------------------------------ | -------------------- | ------ | -------- |
| model_complEx       | ComplEx + 生成权重后相加 + CE_loss                  | 见`train_model.sh`                          | RTX 3090             | 1h10m  | 0.5558   |
| model_lite_bumbling | ComplEx + 生成权重后相加 + CE_loss + OCR            | 见`train_model.sh`                          | RTX 3090             | 1h40m  |          |
| model_lite_polar    | ComplEx + 生成权重后相加 + CE_loss + OCR            | 见`train_model.sh`                          | RTX 3090             | 1h20m  |          |
| model_0.5583        | TuckER+注意力融合头实体 + BCE_loss                  | rank = 1000 dr = 200 4头2层 其余见`CFG`变量 | Tesla P100-PCIE-16GB | 5h50m  | 0.5583   |
| model_0.5580        | TuckER+注意力融合头实体 + BCE_loss                  | rank = 1000 dr = 200 4头2层 其余见`CFG`变量 | Tesla P100-PCIE-16GB | 8h     | 0.5580   |
| model_0.5545        | TuckER+注意力融合头实体 + BCE_loss                  | rank = 500 dr = 200 4头2层 其余见`CFG`变量  | Tesla P100-PCIE-16GB | 4h     | 0.5545   |
| model_flowing       | TuckER+注意力融合头实体 + BCE_loss                  | rank = 1300 dr = 500 4头2层 其余见`CFG`变量 | Tesla V100-SXM2-32GB | 9h     |          |
| model_fast          | TuckER+注意力融合头实体 + BCE_loss                  | rank = 1000 dr = 500 4头2层 其余见`CFG`变量 | Tesla V100-SXM2-32GB | 5h40m  |          |
| model_woven         | TuckER+注意力融合头实体 + BCE_loss                  | rank = 1500 dr = 500 8头2层 其余见`CFG`变量 | Tesla V100-SXM2-32GB | 12h50m |          |
| model_vocal         | TuckER+注意力融合头实体 + 注意力融合关系 + BCE_loss | rank = 500 dr = 200 4头2层 其余见`CFG`变量  | Tesla V100-SXM2-32GB | 4h     |          |

## 安装依赖

```bash
bash requirements.sh
```

## 输入数据

分为两部分，存在了`./data/` 与`./data/OpenBG-IMG`

- `./data/` 官方提供的`train.tsv`, `test_final.tsv`, `images`
- `./data/OpenBG-IMG`数据处理过程中需要的重复数据，如`openbg_vit_best_img_vec.pickle`, `alpha.pickle`

- `./scores/`跳过模型训练这一步直接得到的模型融合所需的结果

```bash
project
 |-- data
 |    |-- images                                # 图片集
 |    |    |-- ent_xxxxxx                       # 实体对应图片
 |    |    |-- ...
 |    |-- OpenBG-IMG                            # 生成的数据
 |    |	   |-- train
 |    |	   |-- valid
 |    |	   |-- openbg_vit_best_img_vec.pickle
 |    |	   |-- alpha.pickle
 |    |	   |-- ...
 |    |-- train.tsv                             # 训练集数据
 |    |-- test_final.tsv                        # 测试集数据
 |-- models
 |	  |-- model_1
 |	  |	   |-- checkpoints
 |	  |	   |    |-- checkpoints.ckpt
 |	  |	   |-- train_model.sh
 |	  |    |-- splitdata.py
 |	  |    |-- ...
 |	  |-- model_2
 |	  |-- ...
 |-- results
 |	  |-- results.txt
 |	  |-- results.tsv
 |-- scores                                     # test.tsv头实体与关系的组合对于所有尾实体的分数（已保存结果）
 |	  |-- scores_1.pickle
 |	  |-- ...
 |-- train.sh 
 |-- predict.sh
 |-- requirements.sh
```

## 训练模型

> 模型融合的命令为：
>
> ```bash
> bash train.sh ensemble
> ```

### model_ComplEx

```bash
bash train.sh complEx
```

- `splitdata.py` 将存放在`../../data/`中的原始`train.tsv`, `test_final.tsv`三元组导入，结果保存在`../../data/OpenBG-IMG/`
- `process_datasets.py`把实体和关系的编码改为了id，并保存了查找表，结果保存在`../../data/OpenBG-IMG/`
- `image_encoder.py`把所有图片转为(1，1000)的向量后，保存为一个以头实体编码为key，图片嵌入向量为value的字典；该文件生成需要约20分钟，`train_model.sh`里把生成该文件的代码注释掉了，是直接调用已经保存好的。
- `learn.py`训练模型
  - `model.py`模型，各个函数的作用与RSME源码类似，结构上把实体的embedding改成了两个，头尾实体分开
  - `dataset.py`完成数据导入及eval、predict等步骤，其中predict生成`scores.pickle`，保存在`./scores/`
  - `train_model.sh` 中指定超参

### model_lite

```bash
bash train.sh lite
```

`train_model.sh`

```bash
# python splitdata.py         
# python process_datasets.py
# python utils.py              # 用paddleOCR提取图片上的文字
# python text_encoder.py       # 用BERT提取文字信息
# python image_encoder.py      # 用ViT提取图片信息

python learn.py \
    --reg=0.0067989365082463905 \
    --rank=2000 \
    --batch_size=400 \
    --learning_rate=0.0017917896516710923 \
    --max_epochs=70 \
    --init=0.33583147294586274 \
    --dropout_prob_fusion=0.1 \

mv ../../scores/scores.pickle ../../scores/score_lite_bumbling.pickle

python learn.py \
    --reg=0.006677529187675686 \
    --rank=2000 \
    --batch_size=400 \
    --learning_rate=0.002003764224288829 \
    --max_epochs=50 \
    --init=0.2685887919625858 \
    --dropout_prob_fusion=0.1 \

mv ../../scores/scores.pickle ../../scores/score_lite_polar.pickle
```

- `utils.py` 其中`get_text_paddleOCR`这个公式根据提取图片上的文字，保存'头实体id': '文字描述' 这样的dict；如果没有文字，该头实体就对应了一个空的list，保存在`../../data/OpenBG-IMG/text_by_id.pickle`，提取OCR的时间在3小时左右(RTX 3090)

- `text_encoder.py`

  `TextBertEmbeddings`这个类使用了`hfl/chinese-roberta-wwm-ext`的预训练BERT，将文字转化为嵌入向量，按照头实体顺序读取、处理、加入嵌入矩阵中，没有文字的用[PAD]对应的word_embedding代替，结果保存在`../../data/OpenBG-IMG/text_token_emb.pickle`，所需时间大概10分钟(RTX 3090)

- `image_encoder.py`

  功能与model_1的`image_encoder.py`类似，但是取的是没有经过classifier层的hidden_layer， 没有图像的同样用[PAD]对应的word_embedding代替，处理逻辑与`text_token_emb.pickle`类似，结果保存在`../../data/OpenBG-IMG/img_token_emb.pickle`中，所需时间大概15分钟(RTX 3090)

### model_0.5583/model_0.5580/model_0.5545/model_flowing/model_fast/model_woven/model_vocal

```bash
bash train.sh 0_5583/0_5580/0_5545/flowing/fast/woven/vocal
```

## 预测

 预测部分直接调用`./scores/`中保存的pickle文件。如果使用`train.sh`从头训练了模型，`./scores/`中保存的pickle文件将会被刷新。

> 模型融合的命令为
>
> ```bash
> bash predict.sh ensemble
> ```
>
> 单模型的命令为
>
> ```bash
> bash predict.sh complEx\lite_bumbling\lite_polar\0.5583\0.5580\0.5545\flowing\fast\woven\vocal
> ```

| 模型代号            | 权重 |
| :------------------ | :--- |
| model_complEx       | 0.15 |
| model_lite_bumbling | 0.08 |
| model_lite_polar    | 0.07 |
| model_0.5583        | 0.1  |
| model_0.5580        | 0.1  |
| model_0.5545        | 0.1  |
| model_flowing       | 0.1  |
| model_fast          | 0.1  |
| model_woven         | 0.1  |
| model_vocal         | 0.1  |











