export CUDA_VISIBLE_DEVICES=5,6
output_dir='outputs/multi_prompt_10_2gpu_12b'
rm -rf $output_dir
mkdir -p $output_dir

nohup python -u ./run_pclue.py \
    --model_name_or_path ./PLM/mengziT5MT \
    --do_train \
    --test_file ./pclue_data/test.json \
    --train_file ./pclue_data/train.json \
    --validation_file  ./pclue_data/test_public.json \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 8 \
    --predict_with_generate \
    --text_column input \
    --summary_column target \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --max_source_length 512 \
    --max_target_length 64 \
    --save_steps 10000 \
    --save_total_limit 1 \
    --pad_to_max_length true \
    --use_prompt true \
    --prompt_paradigm multi \
    --num_tasks 4 \
    --prompt_type from_vocab \
    --max_prompt_length 10 > $output_dir/train.log  2>&1 &
#--max_train_samples 1000 \
#--max_eval_samples 100 \
#--max_predict_samples 100 \
#--save_steps 5000 \
#--save_total_limit 3 \