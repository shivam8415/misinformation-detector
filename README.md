# 🤖 TruthGuard AI - Misinformation Detector

## 🎯 Hackathon Project: AI-Powered Fake News Detection

An intelligent system that detects potential misinformation in news content and educates users on identifying credible information. Built with state-of-the-art BERT transformer models.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![AI](https://img.shields.io/badge/AI-BERT-yellow)
![Accuracy](https://img.shields.io/badge/Accuracy-96%25-brightgreen)

## ✨ Features

- **Real-time Analysis**: Instantly analyzes news headlines and text
- **Educational Explanations**: Explains why content might be misleading
- **High Accuracy**: 96%+ accuracy on test datasets
- **User-Friendly Interface**: Clean web interface built with Gradio
- **Trustworthy AI**: Transparent confidence scoring

## 🚀 How to Run

1. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   python app.py
   ```

3. **Open your browser** to:
   ```
   http://127.0.0.1:7860
   ```

## 🛠️ Technology Stack

- **AI Model**: Hugging Face Transformers + BERT-base
- **Framework**: Gradio for web interface
- **Language**: Python 3.8+
- **Training Data**: FakeNewsNet Dataset
- **Version Control**: Git + Git LFS for large files

## 📊 Performance

- **Accuracy**: 96.3% on test set
- **Inference Time**: < 2 seconds
- **Model Size**: ~438MB
- **Training Time**: ~45 minutes on CPU

## 🎥 Demo

https://github.com/shivam8415/misinformation-detector/assets/.../demo.mp4

*Try these examples:*
- `"Miracle cure discovered in common fruit!"` → 🚨 FAKE NEWS
- `"School board meeting scheduled for next week"` → ✅ LIKELY REAL

## 📁 Project Structure

```
misinformation-detector/
├── app.py              # Web interface
├── train.py            # Model training code
├── make_big_dataset.py # Data preparation
├── my_fake_news_model/ # Trained AI model (Git LFS)
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## 👨‍💻 Developer

**Shivam**  
- 🎓 Computer Science Student
- 🏆 Hackathon Participant
- 🤖 AI Enthusiast

## 📝 License

This project was created for educational purposes and hackathon participation.

---

**⭐ If you find this project useful, please give it a star on GitHub!**