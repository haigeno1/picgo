# 在 configs/eval_local_model.py 中

from opencompass.models import HuggingFace
# 如果你的模型是Chat模型，可能需要使用 HuggingFaceCausalLM 或其它适合的类

model = HuggingFace(
    path='/path/to/your/local/model',  # 本地模型路径
    model_kwargs=dict(
        device_map='auto',
        trust_remote_code=True
    ),
    tokenizer_path='/path/to/your/local/model',
    tokenizer_kwargs=dict(
        padding_side='left',
        truncation='left',
        trust_remote_code=True
    ),
    max_seq_len=2048,
    max_out_len=100,
    batch_size=64,
    run_cfg=dict(num_gpus=1),  # 所需的GPU数量
)

models = [model]










# 在 configs/datasets/mydataset/mydataset_gen.py 中
from opencompass.openicl.icl_eval import BaseEvaluator
from opencompass.datasets import BaseDataset
from opencompass.registry import DATASETS
from typing import Optional, List, Dict, Any
import json

# 定义一个简单的评估器（示例，需根据你的任务类型调整）
class MyEvaluator(BaseEvaluator):
    def score(self, predictions: List[str], references: List[Any]) -> Dict[str, float]:
        # 实现你的评估逻辑，例如计算准确率
        correct = 0
        for pred, ref in zip(predictions, references):
            # 这里假设你的参考答案是字符串，或者是包含正确答案的结构
            # 你需要根据你的数据集格式和任务类型编写具体的判断逻辑
            if pred.strip() == ref.strip(): 
                correct += 1
        accuracy = correct / len(predictions) if predictions else 0
        return {'accuracy': accuracy}

# 定义数据集
@DATASETS.register_module()
class MyDataset(BaseDataset):
    # 实现数据加载和预处理
    def load(self, path: str, name: Optional[str] = None):
        # 从path加载你的数据，例如读取JSONL文件
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

# 在配置中指定数据路径和评估方式
mydataset_eval_cfg = dict(
    evaluator=dict(type=MyEvaluator), # 使用自定义的评估器
    pred_role='BOT', # 根据你的任务调整
    pred_postprocessor=dict(type='first-capital'), # 可选，后处理预测
)

mydataset_datasets = []
# 假设你的数据集文件是 data/mydataset/test.jsonl
mydataset_datasets.append(
    dict(
        type=MyDataset,
        path='data/mydataset/test.jsonl', # 本地数据集文件路径
        name='mydataset', # 数据集名称
        reader_cfg=dict(
            input_columns=['question'], # 输入列的名称
            output_column='answer',     # 输出列（标准答案）的名称
        ),
        eval_cfg=mydataset_eval_cfg
    )
)





# 在 configs/eval_local_model.py 中
with read_base():
    from .datasets.mydataset.mydataset_gen import mydataset_datasets

datasets = [*mydataset_datasets]

