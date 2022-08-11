model_name=$1
cd models
python gen_result.py --model=$model_name
echo '================================='
echo 'results are saved in ./results/'
echo '================================='