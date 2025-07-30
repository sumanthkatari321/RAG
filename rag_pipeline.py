import openai
import numpy as np

def generate_answer(question, relevant_texts):
    context = "\n".join(relevant_texts)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']
