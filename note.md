python run.py \
    --datasets <你的数据集配置名> \  # 例如 siqa_gen，但更推荐方式二指定自定义数据集
    --hf-path /path/to/your/local/model \  # 本地模型路径
    --tokenizer-path /path/to/your/local/model \  # 通常与模型路径相同
    --tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \ # Tokenizer参数
    --model-kwargs device_map='auto' trust_remote_code=True \  # 加载模型的参数
    --max-seq-len 2048 \  # 模型最大序列长度
    --max-out-len 100 \   # 生成的最大token数
    --batch-size 64 \     # 批量大小
    --num-gpus 1 \        # 每个模型副本所需的GPU数量
    --debug               # 调试模式，实时输出日志



