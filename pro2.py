import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from gensim.models import KeyedVectors

model_100d = KeyedVectors.load_word2vec_format("glove.6B.100d.word2vec.txt", binary=False,limit=500000)

words = ['football', 'soccer', 'basketball', 'tennis','engineer','information', 'baseball', 'coach', 'goal', 'player', 'referee', 'team']
word_vectors = np.array([model_100d[word] for word in words])

pca = PCA(n_components=2)
pca_result = pca.fit_transform(word_vectors)

plt.figure(figsize=(10, 8))
for i, word in enumerate(words):
  plt.scatter(pca_result[i, 0], pca_result[i, 1])
  plt.text(pca_result[i, 0] + 0.02, pca_result[i, 1], word, fontsize=12)
plt.title("PCA Visualization of Sports-related Word Embeddings (100d)")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.show()

def get_similar_words(word, model, topn=5):
  similar_words = model.similar_by_word(word, topn=topn)
  return similar_words
similar_words_football = get_similar_words('football', model_100d, topn=5)
print(f"Words similar to 'football': {similar_words_football}")

words_to_print = ['football', 'soccer']

for word in words_to_print:
    if word in model_100d:
        print(f"Vector embedding for '{word}':\n{model_100d[word]}\n")
    else:
        print(f"Word '{word}' not found in the embeddings model.")