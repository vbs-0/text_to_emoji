# 🎭 Emoji Prediction Using BERT

Enhance your digital communication with AI-powered emoji suggestions! This project uses state-of-the-art natural language processing to predict the most relevant emojis for any given text.

## 🌟 Features

- Utilizes Sentence-BERT for advanced semantic understanding
- Predicts most relevant emojis for input text
- Supports multiple languages
- Includes training script for model fine-tuning
 
## 🛠 Technology Stack

- Python 3.7+
- Sentence-Transformers
- NumPy
- PyTorch

## 📁 Project Structure

- `emoji_data.py`: Contains the emoji dataset and processing functions
- `run0.py`: Main script to run the emoji prediction
- `train0.py`: Script for training or fine-tuning the model
- `templates/`: Directory containing HTML templates (if you have a web interface)

## 🚀 Quick Start

1. Clone the repository:
   ```
   git clone https://github.com/vbs-0/text_to_emoji.git
   cd text_to_emoji
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. To train or fine-tune the model:
   ```
   python train0.py
   ```

4. To run the prediction:
   ```
   python run0.py
   ```

## 📚 How It Works

1. Training (`train0.py`):
   - Loads and preprocesses emoji data from `emoji_data.py`
   - Fine-tunes a pre-trained Sentence-BERT model on emoji descriptions
   - Saves the fine-tuned model for later use

2. Prediction (`run0.py`):
   - Loads the fine-tuned Sentence-BERT model
   - Encodes user input text
   - Calculates semantic similarity between input and emojis
   - Returns most relevant emojis

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/vbs-0/text_to_emoji/issues).

## 📄 License

This project is [MIT](https://choosealicense.com/licenses/mit/) licensed.

## 🙏 Acknowledgements
by_vbs🙃

- [Sentence-Transformers](https://www.sbert.net/)
- [Hugging Face](https://huggingface.co/)
- Emoji data from [Unicode.org](https://unicode.org/emoji/charts/full-emoji-list.html)
