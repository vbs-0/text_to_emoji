
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from emoji_data import emojis, emo_list
import torch

# Sentence-BERT model
print("Loading Sentence-BERT model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Sentence-BERT model loaded successfully.")


e_l = [str(i.replace("_", " ")).lower() for i in emo_list]
emoji_mapping = {k.lower().strip(':').replace('_', ' '): v for k, v in emojis.items()}


print("Encoding emoji keywords...")
doc_vectors = model.encode(e_l, show_progress_bar=True)


output_dir = './saved_models'
model.save(f'{output_dir}/sbert_model')
np.save(f'{output_dir}/sbert_vectors.npy', doc_vectors)
print("Models and vectors saved successfully.")


def text_to_emoji(text, top_k=5):
    
    text_vector = model.encode([text])[0]
    
    
    similarities = np.dot(doc_vectors, text_vector) / (np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(text_vector))
    
    
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    top_emojis = []
    for i in top_indices:
        emoji_key = e_l[i]
        if emoji_key in emoji_mapping:
            top_emojis.append(emoji_mapping[emoji_key])
        else:
            print(f"Warning: No emoji found for '{emoji_key}'")
    
    return top_emojis
# Example usage
'''
example_text = "I'm feeling very happy today!"
result = text_to_emoji(example_text)
print(f"Input: {example_text}")
print(f"Top 5 emojis: {' '.join(result)}")
'''
#debug information
'''
print("\nDebug Information:")
print(f"Number of items in e_l: {len(e_l)}")
print(f"Number of items in emoji_mapping: {len(emoji_mapping)}")
print(f"First 5 items in e_l: {e_l[:5]}")
print(f"First 5 items in emoji_mapping: {list(emoji_mapping.items())[:5]}")
example_text = "eyes heart alien india"
result = text_to_emoji(example_text)
print(f"Input: {example_text}")
print(f"Top 5 emojis: {' '.join(result)}")
'''
