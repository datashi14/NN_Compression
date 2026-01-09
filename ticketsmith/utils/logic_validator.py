import torch
import torch.nn.functional as F
from tqdm import tqdm

def evaluate_logic_loss(dense_model, pruned_model, tokenizer, eval_text_list, device="cuda"):
    """
    Measures KL-Divergence: KL(Dense || Pruned).
    Lower is better. 0.0 means the logic is identical.
    """
    dense_model.eval()
    pruned_model.eval()
    
    total_kl = 0
    total_tokens = 0
    
    print("🔬 Running Logic Validation (KL-Divergence)...")
    
    with torch.no_grad():
        for text in tqdm(eval_text_list, desc="KL Evaluation"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            
            # 1. Get Logits from both models
            # We use the same inputs to ensure a "fair" logic comparison
            dense_outputs = dense_model(**inputs)
            pruned_outputs = pruned_model(**inputs)
            
            dense_logits = dense_outputs.logits  # Shape: [batch, seq_len, vocab_size]
            pruned_logits = pruned_outputs.logits
            
            # 2. Convert to Probabilities
            # Target (P): The Dense model's distribution
            # Input (Q): The Pruned model's log-probabilities (required by PyTorch KLDiv)
            p = F.softmax(dense_logits, dim=-1)
            log_q = F.log_softmax(pruned_logits, dim=-1)
            
            # 3. Calculate KL Divergence
            # reduction='batchmean' gives the mathematically correct average
            kl_div = F.kl_div(log_q, p, reduction='batchmean')
            
            total_kl += kl_div.item()
            total_tokens += 1

    avg_kl = total_kl / (total_tokens + 1e-8)
    
    # Senior Engineering Insight: Logic Integrity Score
    # We invert the KL to show a "Consistency %"
    # KL is unbounded [0, Inf), but generally small for good models.
    # Heuristic: If KL > 1.0, logic is very broken.
    logic_integrity = max(0, (1 - avg_kl) * 100)
    
    print(f"\n--- TicketSmith Logic Report ---")
    print(f"Mean KL-Divergence: {avg_kl:.4f}")
    print(f"Logic Integrity:    {logic_integrity:.2f}%")
    
    return avg_kl, logic_integrity
