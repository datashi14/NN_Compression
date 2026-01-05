import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

class QualityGate:
    def __init__(self, config):
        self.config = config.get('quality_gate', {})
        self.enabled = self.config.get('enabled', False)
        # Thresholds
        self.max_loss_delta_pct = self.config.get('max_loss_delta_pct', 0.10) # 10%
        self.max_acc_drop_absolute = self.config.get('max_acc_drop_absolute', 2.0) # 2% drop
        
    def evaluate(self, current_metrics, baseline_metrics):
        """
        Evaluates gate signals.
        Returns (passed: bool, reasons: list[str], signals: dict)
        """
        if not self.enabled:
            return True, ["Gate disabled"], {}
            
        if not baseline_metrics:
            return False, ["Missing baseline metrics"], {}
            
        reasons = []
        signals = {}
        passed = True
        
        # Signal 1: Loss Delta
        # (val_loss_variant - val_loss_base) / val_loss_base
        curr_loss = current_metrics.get('val_loss', float('inf'))
        base_loss = baseline_metrics.get('val_loss', 0.0001)
        
        loss_delta = (curr_loss - base_loss) / base_loss
        signals['loss_delta_pct'] = loss_delta
        
        if loss_delta > self.max_loss_delta_pct:
            passed = False
            reasons.append(f"Loss delta {loss_delta:.2%} exceeds max {self.max_loss_delta_pct:.2%}")
            
        # Signal 2: Metric Drop (Accuracy)
        curr_acc = current_metrics.get('val_acc', 0.0)
        base_acc = baseline_metrics.get('val_acc', 0.0)
        
        acc_drop = base_acc - curr_acc
        signals['acc_drop'] = acc_drop
        
        if acc_drop > self.max_acc_drop_absolute:
            passed = False
            reasons.append(f"Accuracy drop {acc_drop:.2f}% exceeds max {self.max_acc_drop_absolute:.2f}%")
            
        if passed:
            reasons.append("All signals within thresholds")
            
        return passed, reasons, signals

def generate_sample_grid(model, device, loader, save_path, samples=16):
    """
    Generates a grid of predictions vs ground truth for classifiers.
    For Diffusion, this would be actual generation.
    """
    model.eval()
    
    images_list = []
    labels_list = []
    preds_list = []
    
    # Get a batch
    data, target = next(iter(loader))
    data, target = data.to(device), target.to(device)
    
    with torch.no_grad():
        output = model(data)
        preds = output.argmax(dim=1)
        
    # Collect n samples
    for i in range(min(samples, len(data))):
        img_tensor = data[i].cpu()
        # Unnormalize for visualization: (0.1307,), (0.3081,)
        img_tensor = img_tensor * 0.3081 + 0.1307
        images_list.append(img_tensor.squeeze().numpy())
        labels_list.append(target[i].item())
        preds_list.append(preds[i].item())
        
    # Plot
    # Grid size sqrt(samples)
    rows = int(samples**0.5)
    cols = int(samples / rows) + (1 if samples % rows != 0 else 0)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2.5))
    axes = axes.flatten()
    
    for i in range(len(images_list)):
        ax = axes[i]
        ax.imshow(images_list[i], cmap='gray')
        color = 'green' if preds_list[i] == labels_list[i] else 'red'
        ax.set_title(f"T:{labels_list[i]} P:{preds_list[i]}", color=color)
        ax.axis('off')
        
    # Hide unused subplots
    for i in range(len(images_list), len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Sample grid saved to {save_path}")
