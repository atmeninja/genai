from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

glove_input_file = "glove.6B.100d.txt"
word2vec_output_file = "glove.6B.100d.word2vec.txt"

glove2word2vec(glove_input_file, word2vec_output_file)

model = KeyedVectors.load_word2vec_format(word2vec_output_file,
binary=False)

print(model.most_similar("king"))

similar_to_mysore = model.similar_by_vector(model['mysore'], topn=5)
print(f"Words similar to 'mysore': {similar_to_mysore}")

result_vector_1 = model['actor'] - model['man'] + model['woman']

result_1 = model.similar_by_vector(result_vector_1, topn=1)
print(f"'actor - man + woman' = {result_1}")

result_vector_2 = model['india'] - model['delhi'] + model['washington']

result_2 = model.similar_by_vector(result_vector_2, topn=3)
print(f"'India - Delhi + Washington' = {result_2}")

scaled_vector = model['hotel'] * 2
result_2 = model.similar_by_vector(scaled_vector, topn=3)
result_2

import numpy as np
normalized_vector = model['fish'] / np.linalg.norm(model['fish'])
result_2 = model.similar_by_vector(normalized_vector, topn=3)
result_2

average_vector = (model['king'] + model['woman'] + model['man']) / 3
result_2 = model.similar_by_vector(average_vector, topn=3)
result_2

glove_input_file = "glove.6B.50d.txt"
word2vec_output_file = "glove.6B.50d.word2vec.txt"

glove2word2vec(glove_input_file, word2vec_output_file)

model_50d = KeyedVectors.load_word2vec_format(word2vec_output_file,
binary=False)

glove_input_file = "glove.6B.100d.txt"
word2vec_output_file = "glove.6B.100d.word2vec.txt"

glove2word2vec(glove_input_file, word2vec_output_file)

model_100d = KeyedVectors.load_word2vec_format(word2vec_output_file,
binary=False)

word1 = "hospital"
word2 = "doctor"

similarity_50d = model_50d.similarity(word1, word2)

similarity_100d = model_100d.similarity(word1, word2)

print(f"Similarity (50d) between '{word1}' and '{word2}': {similarity_50d:.4f}")
print(f"Similarity (100d) between '{word1}' and '{word2}': {similarity_100d:.4f}")

distance_50d = model_50d.distance(word1, word2)
distance_100d = model_100d.distance(word1, word2)
print(f"Distance (50d) between '{word1}' and '{word2}': {distance_50d:.4f}")
print(f"Distance (100d) between '{word1}' and '{word2}': {distance_100d:.4f}")