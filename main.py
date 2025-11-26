import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

base_model_name = "google/gemma-3-1b-it"
adapter_name = "google/gemma-3-1b-it-emotion-adapter"

# Load model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    trust_remote_code=True
)

# Set pad token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load and format dataset
dataset = load_dataset("dair-ai/emotion", split="train")
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

# Emotion label mapping
label_map = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}

def format_to_gemma_chat(example):
    text = example["text"]
    label = label_map[example["label"]]
    
    # Format: <start_of_turn>user\nprompt<end_of_turn>\n<start_of_turn>model\nresponse<end_of_turn>\n
    formatted = (
        f"<start_of_turn>user\n"
        f"Respond with the following emotion: {label}.<end_of_turn>\n"
        f"<start_of_turn>model\n"
        f"{text}<end_of_turn>\n"
    )
    
    return {"text": formatted}

# Apply formatting
tokenized_dataset = split_dataset.map(format_to_gemma_chat)

# Training configuration
training_arguments = SFTConfig(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=48,
    gradient_accumulation_steps=1,
    optim="adamw_torch",
    save_steps=1000,
    logging_steps=200,
    learning_rate=1e-6,
    weight_decay=0.001,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard",
    dataset_text_field="text",
    max_length=256,
    packing=False,
)

# LoRA configuration
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# Create trainer
sft_trainer = SFTTrainer(
    model=base_model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    peft_config=peft_config,
    processing_class=tokenizer,
    args=training_arguments,
)

# Print trainable parameters
sft_trainer.model.print_trainable_parameters()

# Train
sft_trainer.train(resume_from_checkpoint=False)

# Save adapter
sft_trainer.model.save_pretrained(adapter_name)

# Merge and save
merged_model = sft_trainer.model.merge_and_unload()
merged_model.save_pretrained("google/gemma-3-1b-it-emotion")
tokenizer.save_pretrained("google/gemma-3-1b-it-emotion")