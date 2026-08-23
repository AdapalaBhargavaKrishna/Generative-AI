with open("document.txt", "r", encoding="utf-8") as file:
    text = file.read()


import os
filename = 'report.pdf'
name , extension = os.path.splitext(filename)

print(name)
print(extension.lstrip('.'))

def ingest_document(filename):

    extension = filename.split(".")[-1].lower()

    if extension not in ["txt", "md"]:
        raise ValueError("Unsupported file type")

    with open(filename, "r", encoding="utf-8") as file:
        return file.read()

def preprocess_text(text):
    text = text.lower()
    text = text.strip()

    return text

text = "Python, is useful! AI?"

import re
text = re.sub(r"[^\w\s]", "", text)

import re
words = re.findall(r"\b\w+\b", text.lower())