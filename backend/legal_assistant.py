from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load trained model
model_path = "../model"

tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

def generate_response(user_input):
    prompt = "Summarize the legal situation: " + user_input
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    outputs = model.generate(**inputs, max_length=150)
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return result

# Interactive loop
print("⚖️ AI Legal Assistant (type 'exit' to quit)\n")

while True:
    user_input = input("Enter your legal query: ")
    
    if user_input.lower() == "exit":
        print("Exiting...")
        break
    
    response = generate_response(user_input)
    
    print("\n🧠 AI Response:\n")
    print(response)
    print("\n" + "="*50 + "\n")