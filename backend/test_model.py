from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load your trained model
model_path = "../model"

tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Test input
input_text = "Summarize: The accused was found guilty of theft under IPC section 378 and sentenced to 2 years imprisonment."

inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

outputs = model.generate(**inputs, max_length=150)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n✅ Generated Summary:\n")
print(result)