from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
import numpy as np
from emoji_data import emojis, emo_list

app = Flask(__name__)

print("Loading Sentence-BERT model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Sentence-BERT model loaded successfully.")

e_l = [str(i.replace("_", " ")).lower() for i in emo_list]
emoji_mapping = {k.lower().strip(':').replace('_', ' '): v for k, v in emojis.items()}


doc_vectors = np.load('./saved_models/sbert_vectors.npy')

def text_to_emoji(text):
    
    text_vector = model.encode([text])[0]
    
    
    similarities = np.dot(doc_vectors, text_vector) / (np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(text_vector))
    
    
    top_index = np.argmax(similarities)
    emoji_key = e_l[top_index]
    
    if emoji_key in emoji_mapping:
        return emoji_mapping[emoji_key]
    else:
        print(f"Warning: No emoji found for '{emoji_key}'")
        return "❓"  # Return a question mark emoji if no match is found but maybe it was not working
        

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        input_text = request.form['input_text']
        words = input_text.split()
        emoji_result = ' '.join(text_to_emoji(word) for word in words)
        return render_template('index.html', result=emoji_result, original_text=input_text)
    return render_template('index.html')

if __name__ == '__main__':     #explicitly disabling watchdog so no extra reloading
    app.run(debug=True, use_reloader=False)
