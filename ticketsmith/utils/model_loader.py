from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

def load_optimized_model(model_id):
    # The "Magic" for 8GB GPUs - 4-bit Quantization (NF4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 # Using bfloat16 for modern GPUs like 3070
    )
    
    print(f"Loading {model_id} in 4-bit (NF4)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # Critical for 3070 survival: Gradient Checkpointing
    print("Enabling gradient checkpointing...")
    model.gradient_checkpointing_enable() 
    
    # Prepare for k-bit training (required for PEFT/LoRA usually, but good for stability here too)
    # from peft import prepare_model_for_kbit_training
    # model = prepare_model_for_kbit_training(model)

    return model

if __name__ == "__main__":
    # Test loading
    model_id = "meta-llama/Llama-3.2-1B-Instruct" 
    # Note: User might need to be logged into Hugging Face via `huggingface-cli login` for gated models
    # If not logged in, this script might fail or ask for token.
    try:
        model = load_optimized_model(model_id)
        print(f"Successfully loaded {model_id}")
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"VRAM used: {mem:.2f} GB")
    except Exception as e:
        print(f"Error loading model: {e}")
