export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# model=/root/paddlejob/workspace/env_run/output/pretrained_models/QwQ-32B
# model=/root/paddlejob/workspace/env_run/output/pretrained_models/DeepSeek-R1-Distill-Qwen-32B
model=/root/paddlejob/workspace/env_run/output/pretrained_models/AceReason-Nemotron-14B
model_name=$(basename ${model})

for task in "aime2024" "aime2025" "amc23" ; do
    
    pkill -f occupy.py

    python run_sglang_softthinking.py \
        --dataset ${task} \
        --model_name ${model} \
        --max_topk 10 \
        --max_generated_tokens 32768 \
        --temperature 0.6 \
        --top_p 0.95 \
        --top_k 30 \
        --min_p 0.0 \
        --after_thinking_temperature 0.6 \
        --after_thinking_top_p 0.95 \
        --after_thinking_top_k 30 \
        --after_thinking_min_p 0.0 \
        --early_stopping_entropy_threshold 0.1 \
        --early_stopping_length_threshold 32768 \
        --mem_fraction_static 0.7 \
        --start_idx 0 \
        --end_idx 10000 \
        --num_gpus 8 \
        --num_samples 16 \
        --output_dir outputs/${model_name}
done

for task in "math500" "gpqa_diamond"; do
    
    pkill -f occupy.py

    python run_sglang_softthinking.py \
        --dataset ${task} \
        --model_name ${model} \
        --max_topk 10 \
        --max_generated_tokens 32768 \
        --temperature 0.6 \
        --top_p 0.95 \
        --top_k 30 \
        --min_p 0.0 \
        --after_thinking_temperature 0.6 \
        --after_thinking_top_p 0.95 \
        --after_thinking_top_k 30 \
        --after_thinking_min_p 0.0 \
        --early_stopping_entropy_threshold 0.1 \
        --early_stopping_length_threshold 32768 \
        --mem_fraction_static 0.7 \
        --start_idx 0 \
        --end_idx 10000 \
        --num_gpus 8 \
        --num_samples 5 \
        --output_dir outputs/${model_name}
done

for task in "humaneval" "mbpp" "livecodebench" ; do
    
    pkill -f occupy.py

    python run_sglang_softthinking.py \
        --dataset ${task} \
        --model_name ${model} \
        --max_topk 10 \
        --max_generated_tokens 32768 \
        --temperature 0.6 \
        --top_p 0.95 \
        --top_k 30 \
        --min_p 0.0 \
        --after_thinking_temperature 0.6 \
        --after_thinking_top_p 0.95 \
        --after_thinking_top_k 30 \
        --after_thinking_min_p 0.0 \
        --early_stopping_entropy_threshold 0.1 \
        --early_stopping_length_threshold 32768 \
        --mem_fraction_static 0.7 \
        --start_idx 0 \
        --end_idx 10000 \
        --num_gpus 8 \
        --num_samples 5 \
        --output_dir outputs/${model_name}

done

for task in "humaneval" "mbpp" "livecodebench" ; do

    python run_sglang_softthinking.py \
        --dataset ${task} \
        --model_name ${model} \
        --max_topk 10 \
        --max_generated_tokens 32768 \
        --temperature 0.6 \
        --top_p 0.95 \
        --top_k 30 \
        --min_p 0.0 \
        --after_thinking_temperature 0.6 \
        --after_thinking_top_p 0.95 \
        --after_thinking_top_k 30 \
        --after_thinking_min_p 0.0 \
        --early_stopping_entropy_threshold 0.1 \
        --early_stopping_length_threshold 32768 \
        --mem_fraction_static 0.7 \
        --start_idx 0 \
        --end_idx 10000 \
        --num_gpus 8 \
        --num_samples 5 \
        --output_dir outputs/${model_name} \
        --reeval

done