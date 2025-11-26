from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

#Select the model (uncomment the one you want to use)
#model_name = "google/gemma-3-1b-it"
model_name = "google/gemma-3-1b-it-emotion"

#Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).to("cuda")

#Role instruction for the LLM
#role_instruction = "You are a an upset customer for a delivery service, you are annoyed at your package being delivered late. Respond like the customer."
role_instruction = "You are an angry customer who missed his train due to maintenance work on the platform."

def generate_model_reply(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
    )
    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

print("Chat started. YOU are the assistant. The LLM is the USER.")
print("Type 'exit' to quit.\n")

conversation_history = ""
while True:
    assistant_text = input("\nAssistant (you): ")
    
    if assistant_text.strip().lower() == 'exit':
        break
    
    conversation_history += f"<start_of_turn>user\n{assistant_text}<end_of_turn>\n"
    
    full_prompt = f"{role_instruction}\n\n{conversation_history}<start_of_turn>model\n"
    
    #Generate LLM's response
    model_reply = generate_model_reply(full_prompt)
    print(f"\nLLM (user): {model_reply}")
    conversation_history += f"<start_of_turn>model\n{model_reply}<end_of_turn>\n"