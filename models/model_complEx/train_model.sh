python splitdata.py
python process_datasets.py
# python image_encoder.py

python learn.py \
    --max_epochs=20 \
    --reg=0.003 \
    --rank=2500 \
    --batch_size=100 \
    --learning_rate=5e-2 \
    --max_epochs=20 \
#     --constant

mv ../../scores/scores.pickle ../../scores/score_complEx.pickle
# python gen_result.py
