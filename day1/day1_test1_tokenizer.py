from transformers import AutoTokenizer

# 1. 加载 GPT-2 的分词器 (经典 BPE) 如果无法访问 HuggingFace 仓库，则需要手动下载GPT2 文件
tokenizer_gpt2 = AutoTokenizer.from_pretrained("gpt2")

# 2. 加载一个中文模型分词器 (如 BERT-base-Chinese) 同理
tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-chinese")

text_en = "Learning NLP is unhappily difficult but exciting!"
text_cn = "我喜欢学习自然语言处理。"

# 打印 GPT-2 的英文分词结果
print("GPT-2 English Tokens:", tokenizer_gpt2.tokenize(text_en))
print("GPT-2 English IDs:", tokenizer_gpt2.encode(text_en))

# 打印 BERT 的中文分词结果
print("\nBERT Chinese Tokens:", tokenizer_bert.tokenize(text_cn))
print("BERT Chinese IDs:", tokenizer_bert.encode(text_cn))

# 观察 [CLS] 和 [SEP]
encoded_input = tokenizer_bert(text_cn)
print("\nBERT Full Encoded (with special tokens):", encoded_input['input_ids'])

