"""
Gradient diagnosis script for CTCLossWithLabelPriors training.

Add this to your training script temporarily to check if gradients are flowing correctly.
"""

import torch

def diagnose_gradients(model, loss_module, loss_value, step_num=0):
    """
    Run this after loss.backward() but before optimizer.step()
    
    Args:
        model: Your PhnMonoSSLModel or the hparams.model
        loss_module: The CTCLossWithLabelPriors instance (hparams.ctc_cost)
        loss_value: The loss tensor returned from compute_objectives
        step_num: Current training step (optional)
    """
    print(f"\n{'='*80}")
    print(f"GRADIENT DIAGNOSIS - Step {step_num}")
    print(f"{'='*80}")
    
    # 1. Check loss properties
    print(f"\n1. LOSS PROPERTIES:")
    print(f"   - Type: {type(loss_value)}")
    print(f"   - Is tensor: {isinstance(loss_value, torch.Tensor)}")
    print(f"   - Requires grad: {getattr(loss_value, 'requires_grad', 'N/A')}")
    print(f"   - Value: {loss_value.item() if isinstance(loss_value, torch.Tensor) else loss_value}")
    print(f"   - Device: {loss_value.device if isinstance(loss_value, torch.Tensor) else 'N/A'}")
    
    # 2. Check if loss_module has parameters
    print(f"\n2. LOSS MODULE INFO:")
    if hasattr(loss_module, 'parameters'):
        loss_params = list(loss_module.parameters())
        print(f"   - Has parameters(): {len(loss_params)} params")
        if loss_params:
            print(f"   - Total params: {sum(p.numel() for p in loss_params)}")
            for i, p in enumerate(loss_params[:3]):  # Show first 3
                print(f"     Param {i}: shape={p.shape}, requires_grad={p.requires_grad}")
    else:
        print(f"   - No parameters() method")
    
    # Check for stateful attributes (like log_priors_sum)
    if hasattr(loss_module, 'log_priors_sum'):
        lps = loss_module.log_priors_sum
        print(f"   - log_priors_sum: {type(lps)}, is_tensor={isinstance(lps, torch.Tensor)}")
        if isinstance(lps, torch.Tensor):
            print(f"     shape={lps.shape}, requires_grad={lps.requires_grad}, device={lps.device}")
    
    if hasattr(loss_module, 'log_priors'):
        lp = loss_module.log_priors
        print(f"   - log_priors: {type(lp)}, is_tensor={isinstance(lp, torch.Tensor)}")
        if isinstance(lp, torch.Tensor):
            print(f"     shape={lp.shape}, requires_grad={lp.requires_grad}, device={lp.device}")
    
    # 3. Check model parameters and their gradients
    print(f"\n3. MODEL PARAMETER GRADIENTS:")
    
    # Get all model components
    components = {}
    if hasattr(model, 'modules'):
        if hasattr(model.modules, 'enc'):
            components['enc'] = model.modules.enc
        if hasattr(model.modules, 'ctc_lin'):
            components['ctc_lin'] = model.modules.ctc_lin
        if hasattr(model.modules, 'perceived_ssl'):
            components['perceived_ssl'] = model.modules.perceived_ssl
    
    for comp_name, component in components.items():
        if component is None:
            continue
        params_with_grad = []
        params_without_grad = []
        grad_norms = []
        
        for name, param in component.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_norms.append(grad_norm)
                    params_with_grad.append((name, grad_norm))
                else:
                    params_without_grad.append(name)
        
        print(f"\n   [{comp_name}]")
        print(f"   - Params with grad: {len(params_with_grad)}")
        print(f"   - Params without grad (but requires_grad=True): {len(params_without_grad)}")
        
        if grad_norms:
            print(f"   - Grad norm stats: min={min(grad_norms):.6f}, max={max(grad_norms):.6f}, mean={sum(grad_norms)/len(grad_norms):.6f}")
        
        # Show first 3 params with gradients
        if params_with_grad:
            print(f"   - Sample grads:")
            for name, gnorm in params_with_grad[:3]:
                print(f"     {name}: {gnorm:.6f}")
        
        # Warn about params without grad
        if params_without_grad:
            print(f"   ⚠️  WARNING: {len(params_without_grad)} params require_grad but have no gradient!")
            for name in params_without_grad[:3]:
                print(f"     {name}")
    
    # 4. Check optimizer configuration
    print(f"\n4. OPTIMIZER INFO:")
    if hasattr(model, 'adam_optimizer'):
        opt = model.adam_optimizer
        total_params = sum(len(group['params']) for group in opt.param_groups)
        print(f"   - adam_optimizer: {total_params} parameters")
        print(f"     LR: {opt.param_groups[0]['lr']}")
    
    if hasattr(model, 'pretrained_opt_class'):
        opt = model.pretrained_opt_class
        total_params = sum(len(group['params']) for group in opt.param_groups)
        print(f"   - pretrained_opt_class: {total_params} parameters")
        print(f"     LR: {opt.param_groups[0]['lr']}")
    
    # 5. Final recommendations
    print(f"\n5. DIAGNOSIS:")
    if not isinstance(loss_value, torch.Tensor):
        print(f"   ❌ CRITICAL: Loss is not a tensor! Check if you called .item() or .detach() too early.")
    elif not getattr(loss_value, 'requires_grad', False):
        print(f"   ⚠️  WARNING: Loss doesn't require grad. Check if computed under torch.no_grad().")
    else:
        print(f"   ✅ Loss is a valid tensor with computation graph.")
    
    # Check if any gradients exist
    has_any_grad = False
    for comp_name, component in components.items():
        if component is None:
            continue
        for param in component.parameters():
            if param.grad is not None:
                has_any_grad = True
                break
        if has_any_grad:
            break
    
    if has_any_grad:
        print(f"   ✅ Model parameters have gradients - training should work!")
    else:
        print(f"   ❌ CRITICAL: NO gradients found on model parameters after backward()!")
        print(f"      Possible causes:")
        print(f"      - Loss computation broke the computation graph")
        print(f"      - Model parameters were frozen (requires_grad=False)")
        print(f"      - Backward wasn't called yet")
        print(f"      - Inputs were detached from graph")
    
    print(f"\n{'='*80}\n")


# ============================================================================
# USAGE EXAMPLE - Add to your fit_batch() or compute_objectives()
# ============================================================================
"""
In your PhnMonoSSLModel.fit_batch(), add this after loss.backward():

    if self.hparams.auto_mix_prec:
        # ... existing code ...
        self.scaler.scale(loss / self.hparams.gradient_accumulation).backward()
        
        # ADD THIS:
        if self.step % 100 == 0:  # Check every 100 steps
            from debug_gradient_check import diagnose_gradients
            diagnose_gradients(self, self.hparams.ctc_cost, loss, self.step)
        
        self.scaler.unscale_(self.pretrained_opt_class)
        # ... rest of code ...

Or to check just once at the start of training:

    if self.step == 1:
        from debug_gradient_check import diagnose_gradients
        diagnose_gradients(self, self.hparams.ctc_cost, loss, self.step)
"""
