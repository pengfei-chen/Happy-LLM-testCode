import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import gensim
import numpy as np

def load_glove_vectors(glove_file_path):
    """
    加载 GloVe 词向量文件
    :param glove_file_path: GloVe 文件路径（如 glove.6B.100d.txt）
    :return: gensim的KeyedVectors对象（可直接调用词向量）
    """
    # 初始化词向量字典
    word_vectors = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 拆分每行：词 + 向量值
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            word_vectors[word] = vector
    
    # 转换为Gensim的KeyedVectors（方便后续操作）
    vocab_size = len(word_vectors)
    vector_dim = len(next(iter(word_vectors.values())))
    kv = gensim.models.KeyedVectors(vector_dim)
    kv.add_vectors(list(word_vectors.keys()), list(word_vectors.values()))
    return kv

# 示例：加载GloVe 6B 100维向量（需先下载：https://nlp.stanford.edu/projects/glove/）
glove_path = "./glove.6B.100d.txt"  # 替换为你的GloVe文件路径
wv = load_glove_vectors(glove_path)

# 选择几组词
words_to_plot = ["king", "queen", "prince", "princess", 
                 "apple", "banana", "orange", "fruit",
                 "computer", "laptop", "software", "keyboard"]

# 获取这些词的向量
word_vectors = np.array([wv[w] for w in words_to_plot])

# 使用 PCA 降维
pca = PCA(n_components=2)
coords = pca.fit_transform(word_vectors)

# 绘图
plt.figure(figsize=(10, 8))
plt.scatter(coords[:, 0], coords[:, 1], edgecolors='k', c='r')

for i, word in enumerate(words_to_plot):
    plt.annotate(word, xy=(coords[i, 0], coords[i, 1]), size=12)

plt.title("Word Embedding 2D Visualization (PCA)")
plt.grid(True)
plt.show()
plt.savefig("./word_embedding_pca.png")
