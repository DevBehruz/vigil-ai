import re

def preprocess_text_basic(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '[URL]', text)
    text = re.sub(r'@\w+', '[USER]', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text_aggressive(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-Z\s!?.]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
