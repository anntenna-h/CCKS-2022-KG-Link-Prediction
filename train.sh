model=$1
if [ $model != 'ensemble' ]
then
    cd models/model_$model
    pwd
    bash train_model.sh
else
    echo 'Now training model_0_5545'
    bash train.sh 0_5545
    echo 'Now training model_0_5580'
    bash train.sh 0_5580
    echo 'Now training model_0_5583'
    bash train.sh 0_5583
    echo 'Now training model_complEx'
    bash train.sh complEx
    echo 'Now training model_lite'
    bash train.sh lite
    echo 'Now training model_fast'
    bash train.sh fast
    echo 'Now training model_flowing'
    bash train.sh flowing
    echo 'Now training model_vocal'
    bash train.sh vocal
    echo 'Now training model_woven'
    bash train.sh woven
fi