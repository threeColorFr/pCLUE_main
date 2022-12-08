export CUDA_VISIBLE_DEVICES=4,5 # 控制gpu数量
output_dir='outputs/general_test'

if [ ! -d $output_dir ]; then
    mkdir -p $output_dir
else
    rm -rf $output_dir/*
fi

PLM_path='PLM/mengziT5MT' # 可以换成huggingface在线名字
train_batch_per_gpu=12
lr=1e-4
epoches=4
can_prompt=false # 是否使用prompt，prompt具体相关参数在下面直接修改

# ---data---
trian_file_path='./pclue_data/train.json' # 训练数据。
#trian_file_path='None'
# 训练文件如果不存在，则根据任务指令生成文件（至少四个任务的分训练集要存在e.g. trian_classify.json
if [ ! -f "$trian_file_path" ]; then
    echo "file not exist, will make a file"

    # classify nli generate mrc
    arr=('classify' 'nli' 'generate' 'mrc') # 需要融合的任务, parameter

    echo ${#arr[@]}
    for task in ${arr[@]};do
        name="pclue_data/train_${task}.json"
        echo $name
        cat $name >> $output_dir/temp_train.json
    done
    trian_file_path=$output_dir/temp_train.json
    shuf $trian_file_path -o $trian_file_path
fi
echo "train data:$trian_file_path"

# ---start---
nohup python -u ./run_pclue.py \
    --model_name_or_path $PLM_path\
    --do_train \
    --train_file $trian_file_path \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --per_device_train_batch_size $train_batch_per_gpu \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --text_column input \
    --summary_column target \
    --learning_rate $lr \
    --num_train_epochs $epoches \
    --max_source_length 512 \
    --max_target_length 64 \
    --save_steps 20000 \
    --save_total_limit 10 \
    --pad_to_max_length true \
    --use_prompt $can_prompt \
    --prompt_paradigm single \
    --num_tasks 4 \
    --freeze \
    --prompt_type random \
    --max_prompt_length 10 > $output_dir/train.log  2>&1 &
#--max_train_samples 1000 \
#--max_eval_samples 100 \
#--max_predict_samples 100 \
#--per_device_eval_batch_size 8 \
#--save_steps 5000 \
#--save_total_limit 3 \
#--continue_trian \ prompt用
#--freeze  prompt用
#--test_file ./pclue_data/test.json \
#--validation_file  ./pclue_data/test_public.json \