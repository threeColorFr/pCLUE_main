export CUDA_VISIBLE_DEVICES=0
output='./outputs/general_test/'
#nohup python -u baseline_test.py --output_dir $output > $output/test_public_baseline.log 2>&1 &
python baseline_test.py --output_dir $output \
    --data_dir pclue_data \
    --test_file pCLUE_test_public.json \
    --tokenizer_path PLM/mengziT5MT \
    --model_path $output \
    --max_prompt_length 10 \
    --select_top -1 \
    --do_metrics