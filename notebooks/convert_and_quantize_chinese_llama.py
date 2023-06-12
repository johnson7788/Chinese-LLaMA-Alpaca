#!/usr/bin/env python
# coding: utf-8

# # 转换并量化中文LLaMA/Alpaca模型
# 
# 🎉🎉🎉 **新：现在免费用户也有机会能够转换7B和13B模型了！**
# 
# 💡 提示和小窍门：
# - 免费用户默认的内存只有12G左右，**笔者用免费账号实测选择TPU的话有机会随机出35G内存**，建议多试几次。如果能随机出25G内存以上的机器就可以了转换7B模型了，35G内存以上机器就能转换13B模型了
# - Pro(+)用户请选择 “代码执行程序” -> “更改运行时类型” -> “高RAM”
# - 实测：转换7B级别模型，25G内存的机器就够了；转换13B级别模型需要30G以上的内存（程序莫名崩掉或断开连接就说明内存爆了）
# - 如果选了“高RAM”之后内存还是不够大的话，选择以下操作，有的时候会分配出很高内存的机器，祝你好运😄！
#     - 可以把GPU或者TPU也选上（虽然不会用到）
#     - 选GPU时，Pro用户可选“高级”类型GPU
# 
# 以下信息配置信息供参考（Pro订阅下测试），运行时规格设置为“高RAM”时的设备配置如下（有随机性）：
# 
# | 硬件加速器  |  RAM  |  硬盘  |
# | :-- | :--: | :--: |
# | None | 25GB | 225GB |
# | TPU | 35GB | 225GB |
# | GPU（标准，T4）| 25GB | 166GB |
# | GPU（高性能，V100）| 25GB | 166GB |
# | GPU（高性能，A100）| **80GB** | 166GB |
# 
# *温馨提示：用完之后注意断开运行时，选择满足要求的最低配置即可，避免不必要的计算单元消耗（Pro只给100个计算单元）。*

# ## 安装相关依赖

# In[1]:


get_ipython().system('pip install transformers')
get_ipython().system('pip install peft')
get_ipython().system('pip install sentencepiece')


# ## 克隆目录和代码

# In[2]:


get_ipython().system('git clone https://github.com/ymcui/Chinese-LLaMA-Alpaca')
get_ipython().system('git clone https://github.com/ggerganov/llama.cpp')


# ## 合并模型（以Alpaca-7B为例）
# 
# **⚠️ 再次提醒：7B模型需要25G内存，13B模型需要35G+内存。**
# 
# 此处使用的是🤗模型库中提供的基模型（已是HF格式），而不是Facebook官方的LLaMA模型，因此略去将原版LLaMA转换为HF格式的步骤。
# 
# **这里直接运行第二步：合并LoRA权重**，生成全量模型权重。可以直接指定🤗模型库的地址，也可以是本地存放地址。
# - 基模型：`decapoda-research/llama-7b-hf` *（use at your own risk）*
# - LoRA模型：`ziqingyang/chinese-alpaca-lora-7b`
# 
# 💡 转换13B模型提示：
# - 请将参数`--base_model`和`--lora_model`中的的`7b`改为`13b`即可
# - **免费用户必须增加一个参数`--offload_dir`以缓解内存压力**，例如`--offload_dir ./offload_temp`
# 
# 该过程比较耗时（下载+转换），需要几分钟到十几分钟不等，请耐心等待。
# 转换好的模型存放在`alpaca-combined`目录。
# 如果你不需要量化模型，那么到这一步就结束了。

# In[3]:


get_ipython().system("python ./Chinese-LLaMA-Alpaca/scripts/merge_llama_with_chinese_lora.py     --base_model 'decapoda-research/llama-7b-hf'     --lora_model 'ziqingyang/chinese-alpaca-lora-7b'     --output_dir alpaca-combined")


# ## 量化模型
# 接下来我们使用[llama.cpp](https://github.com/ggerganov/llama.cpp)工具对上一步生成的全量版本权重进行转换，生成4-bit量化模型。
# 
# ### 编译工具
# 
# 首先对llama.cpp工具进行编译。

# In[4]:


get_ipython().system('cd llama.cpp && make')


# ### 模型转换为ggml格式（FP16）
# 
# 这一步，我们将模型转换为ggml格式（FP16）。
# - 在这之前需要把`alpaca-combined`目录挪个位置，把模型文件放到`llama.cpp/zh-models/7B`下，把`tokenizer.model`放到`llama.cpp/zh-models`
# - tokenizer在哪里？
#     - `alpaca-combined`目录下有
#     - 或者从以下网址下载：https://huggingface.co/ziqingyang/chinese-alpaca-lora-7b/resolve/main/tokenizer.model （注意，Alpaca和LLaMA的`tokenizer.model`不能混用！）
# 
# 💡 转换13B模型提示：
# - tokenizer可以直接用7B的，13B和7B的相同
# - Alpaca和LLaMA的`tokenizer.model`不能混用！
# - 以下看到7B字样的都是文件夹名，与转换过程没有关系了，改不改都行

# In[6]:


get_ipython().system('cd llama.cpp && mkdir zh-models && mv ../alpaca-combined zh-models/7B')
get_ipython().system('mv llama.cpp/zh-models/7B/tokenizer.model llama.cpp/zh-models/')
get_ipython().system('ls llama.cpp/zh-models/')


# In[7]:


get_ipython().system('cd llama.cpp && python convert.py zh-models/7B/')


# ### 将FP16模型量化为4-bit
# 
# 我们进一步将FP16模型转换为4-bit量化模型。

# In[8]:


get_ipython().system('cd llama.cpp && ./quantize ./zh-models/7B/ggml-model-f16.bin ./zh-models/7B/ggml-model-q4_0.bin 2')


# ### （可选）测试量化模型解码
# 至此已完成了所有转换步骤。
# 我们运行一条命令测试一下是否能够正常加载并进行对话。
# 
# FP16和Q4量化文件存放在./llama.cpp/zh-models/7B下，可按需下载使用。

# In[10]:


get_ipython().system('cd llama.cpp && ./main -m ./zh-models/7B/ggml-model-q4_0.bin --color -f ./prompts/alpaca.txt -p "详细介绍一下北京的名胜古迹：" -n 512')

