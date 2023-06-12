#!/usr/bin/env python
# coding: utf-8

# # Tutorial on pre-training of Chinese-LLaMA-7B
# 
# More info: https://github.com/ymcui/Chinese-LLaMA-Alpaca

# ## Install Dependencies

# In[3]:


get_ipython().system('pip install transformers==4.28.1')
get_ipython().system('pip install git+https://github.com/huggingface/peft.git@13e53fc')
get_ipython().system('pip install datasets')
get_ipython().system('pip install sentencepiece')
get_ipython().system('pip install deepspeed')


# ## Clone our repository
# 
# 
# 
# 

# In[4]:


get_ipython().system('git clone https://github.com/ymcui/Chinese-LLaMA-Alpaca.git')


# ## Pre-training for LLaMA-7B
# 
# This follows the setting in https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/Pretraining-Script, except that to simplify the tutorial,
# - only train 100 steps
# - use a sample data file built from alpaca_data_zh_51k.json

# In[ ]:


get_ipython().system('mkdir Chinese-LLaMA-Alpaca/pt_data')
get_ipython().system('cp Chinese-LLaMA-Alpaca/data/pt_sample_data.txt Chinese-LLaMA-Alpaca/pt_data')


# In[9]:


get_ipython().system('cd Chinese-LLaMA-Alpaca/scripts && torchrun --nnodes 1 --nproc_per_node 1 run_clm_pt_with_peft.py     --deepspeed ds_zero2_no_offload.json     --model_name_or_path decapoda-research/llama-7b-hf     --tokenizer_name_or_path ziqingyang/chinese-llama-lora-7b     --dataset_dir /content/Chinese-LLaMA-Alpaca/pt_data     --data_cache_dir data_cache     --validation_split_percentage 0.001     --per_device_train_batch_size 1     --do_train     --fp16     --seed $RANDOM     --max_steps 100     --lr_scheduler_type cosine     --learning_rate 2e-4     --warmup_ratio 0.05     --weight_decay 0.01     --logging_strategy steps     --logging_steps 10     --save_strategy steps     --save_total_limit 3     --save_steps 50     --gradient_accumulation_steps 1     --preprocessing_num_workers 8     --block_size 512     --output_dir /content/output_model     --overwrite_output_dir     --ddp_timeout 30000     --logging_first_step True     --lora_rank 8     --lora_alpha 32    --trainable q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj     --modules_to_save embed_tokens,lm_head     --lora_dropout 0.05     --torch_dtype float16     --gradient_checkpointing     --ddp_find_unused_parameters False')


# After training, rename saved `pytorch_model.bin` to `adapter_model.bin`

# In[10]:


get_ipython().system('mkdir output_model/peft_model')
get_ipython().system('mv output_model/pytorch_model.bin output_model/peft_model/adapter_model.bin')


# Lastly, you need to manually create an `adapter_config.json` under `peft_model` and fill in the hyperparamters such as `lora_rank`, `lora_alpha` etc., whose content and 
# format can be referenced from the corresponding file in Chinese-LLaMA-LoRA.
