# app.py
from transformers import pipeline
import gradio as gr

print("Loading your AI model...")
classifier = pipeline("text-classification", model="./my_fake_news_model")

def check_news(text):
    if not text.strip():
        return "Please enter some text to analyze."
    
    result = classifier(text)[0]
    label = result['label']
    confidence = result['score'] * 100
    
    if label == "FAKE":
        reasons = [
            "• Uses emotionally charged or sensational language",
            "• Makes extraordinary claims without evidence", 
            "• Creates urgency or prompts immediate sharing",
            "• References unnamed 'experts' or 'secret sources'"
        ]
        reasons_text = "\n".join(reasons[:2])
        return f"""🚨 FAKE NEWS ({confidence:.1f}% confidence)

This content shows characteristics of misinformation:

{reasons_text}

🔍 Tip: Check multiple reputable sources to verify."""
    else:
        reasons = [
            "• Reports measurable facts and events",
            "• Uses balanced, factual language",
            "• Cites specific sources or evidence",
            "• Avoids extreme emotional language"
        ]
        reasons_text = "\n".join(reasons[:2])
        return f"""✅ LIKELY REAL ({confidence:.1f}% confidence)

This content appears trustworthy:

{reasons_text}

📚 Tip: Still good practice to verify with known reputable sources."""

examples = [
    ["JUST RELEASED: One fruit melts belly fat overnight while you sleep!"],
    ["City council approves new park construction budget unanimously"],
    ["SHOCKING: Government hiding alien technology from public!"],
    ["University study finds daily walking reduces heart disease risk by 30%"]
]

demo = gr.Interface(
    fn=check_news,
    inputs=gr.Textbox(
        label="📰 Enter News Headline", 
        placeholder="Paste news content here...",
        lines=3
    ),
    outputs=gr.Textbox(
        label="🔍 Analysis Result",
        lines=6
    ),
    title="🤖 TruthGuard AI - Misinformation Detector",
    description="AI-powered tool to detect potential misinformation and educate users.",
    examples=examples
)

demo.launch()