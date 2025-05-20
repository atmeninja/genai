from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

corpus = [
    "The stock market saw significant gains today, driven by strong earnings reports.",
    "Investing in diversified portfolios helps mitigate risk and maximize returns.",
    "The Federal Reserve's decision to raise interest rates could impact market liquidity.",
    "Cryptocurrency has become an increasingly popular asset class among investors.",
    "Financial analysts predict that the global economy will face challenges in the coming years.",
    "Bonds are considered a safer investment option compared to stocks.",
    "Banks are adopting blockchain technology to improve the efficiency of financial transactions.",
    "The economic impact of the pandemic has been severe, but recovery is underway.",
    "Inflation rates have been rising steadily, leading to higher costs for consumers.",
    "Corporate governance and transparency are crucial for investor confidence.",
    "The real estate market is experiencing a boom as demand outstrips supply in many areas.",
    "Investors should be aware of market volatility and adjust their strategies accordingly.",
    "Diversification is a key principle in reducing risk in investment portfolios.",
    "Hedge funds use complex strategies to generate high returns, often with higher risks.",
    "Stock buybacks are often seen as a sign of confidence by corporate executives."
]

corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
corpus_embeddings

def generate_paragraph(seed_word, corpus, corpus_embeddings, model, top_n=5):
    seed_sentence = f"Tell me more about {seed_word} in finance."
    seed_embedding = model.encode(seed_sentence, convert_to_tensor=True)


    similarities = util.pytorch_cos_sim(seed_embedding, corpus_embeddings)[0]
    top_results = similarities.topk(top_n)
    print('top_results:',top_results)
    story = f"The topic of '{seed_word}' is crucial in the finance industry. "

    for idx in top_results.indices:
        similar_sentence = corpus[idx]
        story += f"{similar_sentence} "
    story += f"These concepts highlight the importance of {seed_word} in understanding financial markets and investment strategies."
    return story

seed_word = "bonds"
story = generate_paragraph(seed_word, corpus, corpus_embeddings, model, top_n=5)
print(story)