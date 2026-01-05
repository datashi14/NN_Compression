import torch
import torch.nn as nn
import numpy as np

class Pruner:
    def __init__(self, model, pruning_rate=0.2, method='global'):
        self.model = model
        self.pruning_rate = pruning_rate
        self.method = method
        self.masks = {}
        
    def _is_prunable(self, module):
        # We prune Conv2d and Linear layers
        return isinstance(module, (nn.Conv2d, nn.Linear))

    def compute_mask(self, current_masks=None):
        """
        Computes the mask for the current model weights.
        If current_masks is provided, we prune FROM the current active weights 
        (iterative pruning), effectively removing pruning_rate % of *remaining* weights.
        OR we can treat pruning_rate as the target density?
        
        Usually IMP is: remove p% of *surviving* weights each round.
        So target_sparsity = current_sparsity + (1 - current_sparsity) * rate?
        
        Let's implement: remove p% of the *remaining* weights.
        """
        
        # 1. Gather all weights
        all_weights = []
        for name, module in self.model.named_modules():
            if self._is_prunable(module):
                # If we have a mask already, we should only consider currently unpruned weights?
                # A simpler way is to just look at the weights themselves.
                # If a weight is 0, it's already pruned. 
                # But due to floating point, we should check against the mask if possible.
                # For this implementation, we assume that if we are doing iterative pruning,
                # the weights that were pruned are already zero (or we ignore them).
                
                # We will gather absolute values of ALL weights, but we need to mask out
                # the already pruned ones so they don't count towards the threshold calculation
                # (they are 0, so they are at the bottom anyway).
                
                # Wait, if we have 0s, and we take bottom 20%, we include the 0s. 
                # We want to remove 20% of the NON-ZERO weights.
                
                tensor = module.weight.data.cpu().abs()
                if current_masks and name in current_masks:
                   mask = current_masks[name]
                   tensor = tensor[mask.bool()] # Select only active weights
                else:
                   tensor = tensor.flatten()
                
                all_weights.append(tensor)
                
        all_weights = torch.cat(all_weights)
        
        # 2. Find threshold
        # We want to prune `pruning_rate` of these weights.
        number_to_prune = int(len(all_weights) * self.pruning_rate)
        if number_to_prune == 0:
            print("Warning: pruning rate too small, no weights to prune.")
            threshold = 0.0
        else:
            # torch.kthvalue finds the k-th smallest element
            # k is 1-indexed
            threshold = torch.kthvalue(all_weights.flatten(), number_to_prune)[0].item()
            
        print(f"Pruning Threshold: {threshold:.6f} (Pruning {number_to_prune}/{len(all_weights)} weights)")
            
        # 3. Create new masks
        new_masks = {}
        total_params = 0
        active_params = 0
        
        for name, module in self.model.named_modules():
            if self._is_prunable(module):
                weight = module.weight.data
                # Keep weights > threshold
                # AND keep weights that were already active (if we respect previous mask? 
                # Actually if we assume previous pruned weights are 0, they are <= threshold.
                # But to be safe, we usually intersect (though here we calculated threshold based on active only).
                # Simpler global approach:
                # Just mask = |w| > threshold. 
                # BUT this re-enables weights if they grew? No, because we masked them to 0 and froze them usually?
                # In IMP, we rewind. So the weights are non-zero again?
                # NO. In IMP:
                # 1. Train.
                # 2. Prune (create mask).
                # 3. Rewind (reset weights to init).
                # 4. Apply Mask (zero out pruned weights).
                # 5. Train.
                
                # So when we compute mask, we are at step 2. All weights are non-zero (or previously pruned ones are 0).
                # If we are in round 2, the weights that were pruned in round 1 are 0.
                # They will be <= threshold.
                # So `abs(weight) > threshold` correctly keeps them pruned.
                
                mask = (weight.abs() > threshold).float()
                
                # Enforce that previously pruned weights stay pruned?
                # If a weight happened to be 0 but we want to keep it? Unlikely.
                # If we computed threshold on active weights only, we need to be careful.
                # If we use global threshold on ALL weights including zeros:
                # We are removing bottom X%. Since zeros are at bottom, we remove them again + more.
                # This works for iterative pruning naturally.
                
                new_masks[name] = mask
                
                active = mask.sum().item()
                total = mask.numel()
                active_params += active
                total_params += total
                
                print(f"Layer {name}: sparsity {100*(1-active/total):.2f}%")
                
        global_sparsity = 1.0 - (active_params / total_params)
        print(f"Global Sparsity: {100*global_sparsity:.2f}%")
        
        stats = {
            'global_sparsity': global_sparsity,
            'active_params': active_params,
            'total_params': total_params
        }
        return new_masks, stats

    def apply_mask(self, masks):
        """
        Applies mask to model weights (sets them to 0).
        Also registers a hook or makes sure gradients are 0?
        For simple LTH, we just set data to 0 before forward/after step.
        """
        for name, module in self.model.named_modules():
            if name in masks:
                mask = masks[name].to(module.weight.device)
                module.weight.data *= mask
                # To prevent update, we can zero grad, or register hook.
                # A simple way in loop is `weight.grad *= mask`
                
    def get_mask_hook(self, masks):
        def hook(grad):
             # We need to identifying which module this grad belongs to? 
             # Pytorch hooks are on tensor or module.
             pass 
        pass 
