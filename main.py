from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

@app.route('/')
def generate_text():
    prompt = request.args.get('prompt', default='', type=str)

    if prompt:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text
    else:
        return jsonify({"error": "Please provide a 'prompt' parameter in the URL."})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=4000)