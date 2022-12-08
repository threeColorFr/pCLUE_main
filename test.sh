export CUDA_VISIBLE_DEVICES=5
output='outputs/base_mrc'
#nohup python -u baseline_test.py --output_dir $output > $output/test_public_baseline.log 2>&1 &
nohup python -u baseline_test.py --output_dir $output \
    --data_dir pclue_data \
    --test_file pCLUE_test_public.json \
    --tokenizer_path PLM/mengziT5MT \
    --model_path $output \
    --select_top -1 \
    --do_metrics \
    --max_prompt_length 10 \
    --prompt_type random \
    --prompt_paradigm single > $output/test_public.log 2>&1 &
#    --use_prompt \
#    --do_metrics \