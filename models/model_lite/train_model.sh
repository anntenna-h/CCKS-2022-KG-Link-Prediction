# python splitdata.py
# python process_datasets.py
# python utils.py
# python text_encoder.py
# python image_encoder.py


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
