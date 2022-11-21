export CUDA_VISIBLE_DEVICES=6
output='outputs/multi_prompt_10_2gpu_12b'
#nohup python -u baseline_test.py --output_dir $output > $output/test_public_baseline.log 2>&1 &
nohup python -u baseline_test.py --output_dir $output \
    --data_dir pclue_data \
    --test_file pCLUE_test.json \
    --tokenizer_path PLM/mengziT5MT \
    --model_path $output \
    --max_prompt_length 10 \
    --select_top -1 \
    --prompt_type from_vocab \
    --use_prompt \
    --prompt_paradigm multi > $output/test.log 2>&1 &