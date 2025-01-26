from src.model.embedding import Embedding
from .prompts import EXTRACT_KEYWORDS
from src.model.inference_endpoints import LLM
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import ast
import time

def extract_keyword(
    model: LLM,
    question=None,
    hint=None,
    few_shot_examples=None
):
    completion = model(
        response_type="single",
        prompts=EXTRACT_KEYWORDS.format(
            few_shot_examples=few_shot_examples,
            question=question,
            hint=hint
        )
    ).result

    completion = completion.strip().lstrip('ANSWER: ').lstrip('assistant: ')
    print('COMPLETION', completion)
    # Remove multiple newlines
    completion = re.sub(r'\n+', '\n', completion)
    completion = re.sub(r'“|”', '"', completion).replace("‘", "'").replace("’", "'").replace("*", "")
    return ast.literal_eval(completion)

def semantic_rerank(
    embed_obj: Embedding,
    strings,
    keyword
):
    def embed_with_retry(text, retries=3, delay=60):
        for attempt in range(retries):
            try:
                return embed_obj.embed_query(text)
            except Exception as e:
                print(f"Encountered exception during embedding: {e}, retrying")
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    raise Exception("An error occurred while embedding the text.")

    # Embed the keyword
    keyword_embedding = embed_with_retry(keyword)

    # Embed the strings
    embeddings = [embed_with_retry(string) for string in strings]

    cosine_similarities = cosine_similarity([keyword_embedding], embeddings)[0]
    sorted_indices = np.argsort(cosine_similarities)[::-1]
    sorted_embeddings = [strings[i] for i in sorted_indices]

    return sorted_embeddings[:10]
