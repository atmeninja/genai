from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

glove_input_file = "glove.6B.100d.txt"
word2vec_output_file = "glove.6B.100d.word2vec.txt"

glove2word2vec(glove_input_file, word2vec_output_file)

model = KeyedVectors.load_word2vec_format(word2vec_output_file,
binary=False)

print(model.most_similar("king"))

original_prompt = "Explain the importance of vaccinations in healthcare."

key_terms = ["vaccinations", "healthcare"]

similar_terms = []

for term in key_terms:

    if term in model.key_to_index:
        similar_terms.extend({word for word, _ in model.most_similar(term, topn=3)})

if similar_terms:
    enriched_prompt = f"{original_prompt} Consider aspects like: {', '.join(similar_terms)}."
else:
    enriched_prompt = original_prompt

print("Original Prompt:", original_prompt)
print("Enriched Prompt:", enriched_prompt)

import getpass
import os
GOOGLE_API_KEY= os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0,
    api_key=GOOGLE_API_KEY,
    max_tokens=256,
    timeout=None,
    max_retries=2,
)

llm.invoke("Hi")

print(llm.invoke(original_prompt).content)