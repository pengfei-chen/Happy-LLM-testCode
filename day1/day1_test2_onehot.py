import numpy as np

# 1. 原始数据
sentences = [
    "I love AI",
    "I love NLP",
    "AI is cool"
]

# 2. 构建词表 (Vocabulary)
# 将所有句子分词并去重
words = set()
for s in sentences:
    for word in s.split():
        words.add(word)

vocab = sorted(list(words))
word_to_idx = {word: i for i, word in enumerate(vocab)}

print("词表:", vocab)
print("词到索引的映射:", word_to_idx)

# 3. 生成 One-hot 向量函数
def get_one_hot(word):
    vector = np.zeros(len(vocab))
    if word in word_to_idx:
        vector[word_to_idx[word]] = 1
    return vector

# 4. 测试
word1 = "AI"
word2 = "NLP"
v1 = get_one_hot(word1)
v2 = get_one_hot(word2)

print(f"\n'{word1}' 的向量: {v1}")
print(f"\n'{word2}' 的向量: {v2}")
print(f"两个向量的点积 (相似度): {np.dot(v1, v2)}") # 结果一定是 0
