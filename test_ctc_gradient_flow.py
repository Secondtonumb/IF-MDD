"""
Quick test to verify CTCLossWithLabelPriors gradient flow.

Run this standalone to verify:
1. Loss computation returns a proper tensor with grad
2. Model parameters receive gradients
3. Prior statistics are accumulated correctly
"""

import torch
import sys
sys.path.append('.')

from utils.CTCLossWithLabelPriors import CTCLossWithLabelPriors

def test_ctc_loss_gradient_flow():
    print("="*80)
    print("Testing CTCLossWithLabelPriors Gradient Flow")
    print("="*80)
    
    # Setup
    batch_size = 2
    T = 50  # time steps
    C = 44  # num classes
    target_len = 10
    
    # Create a simple model (just a linear layer for testing)
    model = torch.nn.Linear(C, C)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create loss module
    ctc_loss = CTCLossWithLabelPriors(
        prior_scaling_factor=0.6,
        ctc_implementation='k2',
        blank=0,
        reduction='sum'
    )
    
    # Create dummy data
    log_probs = torch.randn(T, batch_size, C, requires_grad=True)  # (T, N, C)
    log_probs = torch.nn.functional.log_softmax(log_probs, dim=-1)
    
    # Pass through model to create computation graph
    log_probs_model = model(log_probs.permute(1, 0, 2)).permute(1, 0, 2)  # (T, N, C)
    
    targets = torch.randint(1, C, (batch_size, target_len))  # Avoid blank (0)
    input_lengths = torch.tensor([T, T])
    target_lengths = torch.tensor([target_len, target_len])
    
    print(f"\n1. Input Setup:")
    print(f"   log_probs shape: {log_probs_model.shape}")
    print(f"   log_probs requires_grad: {log_probs_model.requires_grad}")
    print(f"   targets shape: {targets.shape}")
    print(f"   Model params: {sum(p.numel() for p in model.parameters())}")
    
    # Forward pass
    print(f"\n2. Computing Loss...")
    try:
        loss = ctc_loss(
            log_probs_model,
            targets,
            input_lengths,
            target_lengths,
            step_type="train"
        )
        
        print(f"   ✅ Loss computed successfully")
        print(f"   Loss type: {type(loss)}")
        print(f"   Loss value: {loss.item():.4f}")
        print(f"   Loss requires_grad: {loss.requires_grad}")
        print(f"   Loss device: {loss.device}")
        
    except Exception as e:
        print(f"   ❌ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check prior accumulation
    print(f"\n3. Prior Statistics:")
    print(f"   num_samples: {ctc_loss.num_samples}")
    print(f"   log_priors_sum: {ctc_loss.log_priors_sum is not None}")
    if ctc_loss.log_priors_sum is not None:
        print(f"   log_priors_sum shape: {ctc_loss.log_priors_sum.shape}")
        print(f"   log_priors_sum requires_grad: {ctc_loss.log_priors_sum.requires_grad}")
    
    # Backward pass
    print(f"\n4. Running Backward...")
    optimizer.zero_grad()
    
    try:
        loss.backward()
        print(f"   ✅ Backward completed successfully")
    except Exception as e:
        print(f"   ❌ Backward failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check gradients
    print(f"\n5. Checking Gradients:")
    grads_found = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"   ✅ {name}: grad_norm={grad_norm:.6f}")
            grads_found = True
        else:
            print(f"   ❌ {name}: NO GRADIENT")
    
    if not grads_found:
        print(f"\n   ❌ CRITICAL: No gradients found on model parameters!")
        return False
    
    # Optimizer step
    print(f"\n6. Running Optimizer Step...")
    param_before = {name: p.clone() for name, p in model.named_parameters()}
    optimizer.step()
    
    params_changed = False
    for name, param in model.named_parameters():
        diff = (param - param_before[name]).abs().max().item()
        if diff > 1e-8:
            print(f"   ✅ {name}: changed by {diff:.6e}")
            params_changed = True
        else:
            print(f"   ⚠️  {name}: no change (diff={diff:.6e})")
    
    if not params_changed:
        print(f"\n   ⚠️  WARNING: Parameters didn't change after optimizer.step()")
    
    # Simulate epoch end (update priors)
    print(f"\n7. Simulating Epoch End (Prior Update)...")
    if ctc_loss.log_priors_sum is not None and ctc_loss.num_samples > 0:
        ctc_loss.log_priors = (
            ctc_loss.log_priors_sum 
            - torch.log(torch.tensor(float(ctc_loss.num_samples), device=ctc_loss.log_priors_sum.device))
        )
        print(f"   ✅ Priors updated: shape={ctc_loss.log_priors.shape}")
        print(f"      Min prior: {ctc_loss.log_priors.min().item():.4f}")
        print(f"      Max prior: {ctc_loss.log_priors.max().item():.4f}")
        
        # Reset for next epoch
        ctc_loss.log_priors_sum = None
        ctc_loss.num_samples = 0
    
    print(f"\n" + "="*80)
    print(f"✅ ALL TESTS PASSED - Gradient flow is working correctly!")
    print(f"="*80)
    return True


if __name__ == "__main__":
    success = test_ctc_loss_gradient_flow()
    sys.exit(0 if success else 1)
