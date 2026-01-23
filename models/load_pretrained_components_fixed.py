"""
Fixed load_pretrained_components method with proper key mapping
Replace the original method in Trans_IFMDD_ConPCO_ver2.py with this implementation
"""

def load_pretrained_components(self, checkpoint_path, components_to_load=None, freeze_loaded=True):
    """
    Load specific components from a pretrained model checkpoint with proper key mapping
    
    Args:
        checkpoint_path (str): Path to the checkpoint directory or file
        components_to_load (list): List of components to load. 
                                 Options: ['ssl', 'encoder', 'ctc_head', 'decoder', 'enc', 'enc_projection']
                                 If None, loads ['ssl'] by default
        freeze_loaded (bool): Whether to freeze the loaded components
    
    Note:
        This function handles key mapping between checkpoint format (numeric indices)
        and target model format (named modules):
        - Checkpoint [0] → modules.enc (VanillaNN + LayerNorm)
        - Checkpoint [1] → modules.TransASR.encoder (Conformer)
        - Checkpoint [2] → modules.ctc_lin (CTC output layer)
        - Checkpoint [3] → lm_weight (OTTC alpha prediction)
    """
    if components_to_load is None:
        components_to_load = ['ssl']  # Default: load SSL only
    
    print(f"\n{'='*80}")
    print(f"🔄 Loading pretrained components from: {checkpoint_path}")
    print(f"   Components to load: {components_to_load}")
    print(f"{'='*80}")
    
    import os
    import torch
    
    # Load SSL model separately (uses Pretrainer as before)
    if 'ssl' in components_to_load:
        from speechbrain.utils.parameter_transfer import Pretrainer
        
        ssl_pretrainer = Pretrainer(
            collect_in=self.hparams.pretrained_model_path,
            loadables={"perceived_ssl": self.modules.perceived_ssl},
            paths={"perceived_ssl": "perceived_ssl.ckpt"},
        )
        ssl_pretrainer.collect_files(default_source=self.hparams.pretrained_model_path)
        ssl_pretrainer.load_collected()
        print("✅ SSL model loaded successfully")
    
    # Load acoustic model components with key mapping
    acoustic_components = [c for c in components_to_load if c != 'ssl']
    if acoustic_components:
        model_ckpt_path = os.path.join(checkpoint_path, "model.ckpt")
        if not os.path.exists(model_ckpt_path):
            print(f"⚠️  Warning: {model_ckpt_path} not found, skipping acoustic components")
        else:
            print(f"\n📦 Loading acoustic components from: {model_ckpt_path}")
            
            # Load checkpoint
            ckpt_state = torch.load(model_ckpt_path, map_location=self.device)
            model_state_dict = self.modules.state_dict()
            
            # Define key mapping rules
            # Checkpoint format: 0.*, 1.*, 2.*, 3.*
            # Target format: enc.*, TransASR.encoder.*, ctc_lin.*, etc.
            mapping = {
                '0.': 'enc.',                      # VanillaNN + LayerNorm
                '1.': 'TransASR.encoder.',         # Conformer encoder
                '2.': 'ctc_lin.',                  # CTC output layer
                '3.': 'lm_weight.',                # OTTC alpha (if exists)
            }
            
            loaded_keys = []
            skipped_keys = []
            shape_mismatch = []
            
            for ckpt_key, ckpt_param in ckpt_state.items():
                # Apply mapping to get target model key
                model_key = None
                for ckpt_prefix, model_prefix in mapping.items():
                    if ckpt_key.startswith(ckpt_prefix):
                        model_key = ckpt_key.replace(ckpt_prefix, model_prefix, 1)
                        break
                
                if model_key is None:
                    skipped_keys.append(ckpt_key)
                    continue
                
                # Check if we should load this component
                should_load = False
                if 'enc' in components_to_load and model_key.startswith('enc.'):
                    should_load = True
                elif 'encoder' in components_to_load and model_key.startswith('TransASR.encoder.'):
                    should_load = True
                elif 'ctc_head' in components_to_load and model_key.startswith('ctc_lin.'):
                    should_load = True
                
                if not should_load:
                    continue
                
                # Load parameter if key exists and shape matches
                if model_key in model_state_dict:
                    if model_state_dict[model_key].shape == ckpt_param.shape:
                        model_state_dict[model_key] = ckpt_param.to(self.device)
                        loaded_keys.append((ckpt_key, model_key))
                    else:
                        shape_mismatch.append((ckpt_key, ckpt_param.shape, model_key, model_state_dict[model_key].shape))
                else:
                    skipped_keys.append(f"{ckpt_key} → {model_key} (not found in model)")
            
            # Load the updated state dict
            missing_keys, unexpected_keys = self.modules.load_state_dict(model_state_dict, strict=False)
            
            # Report loading status
            print(f"\n{'='*80}")
            print(f"📊 Loading Summary:")
            print(f"{'='*80}")
            print(f"✅ Successfully loaded: {len(loaded_keys)} parameters")
            
            if loaded_keys:
                print(f"\n📝 Loaded parameters (showing first 10):")
                for i, (ckpt_k, model_k) in enumerate(loaded_keys[:10]):
                    print(f"   {i+1:2d}. {ckpt_k} → {model_k}")
                if len(loaded_keys) > 10:
                    print(f"   ... and {len(loaded_keys) - 10} more")
            
            if shape_mismatch:
                print(f"\n⚠️  Shape mismatch ({len(shape_mismatch)} parameters):")
                for ckpt_k, ckpt_shape, model_k, model_shape in shape_mismatch[:5]:
                    print(f"   {ckpt_k} {ckpt_shape} ≠ {model_k} {model_shape}")
            
            if skipped_keys and len(skipped_keys) <= 10:
                print(f"\n⏭️  Skipped keys: {len(skipped_keys)}")
                for sk in skipped_keys[:5]:
                    print(f"   - {sk}")
            
            print(f"{'='*80}\n")
    
    # Freeze loaded components if requested
    if freeze_loaded:
        for component in components_to_load:
            if component == 'ssl':
                for param in self.modules.perceived_ssl.parameters():
                    param.requires_grad = False
                self.ssl_frozen = True
                print("   🔒 SSL model frozen")
                
            elif component == 'encoder':
                for param in self.modules.TransASR.encoder.parameters():
                    param.requires_grad = False
                if hasattr(self.modules.TransASR, 'custom_src_module'):
                    for param in self.modules.TransASR.custom_src_module.parameters():
                        param.requires_grad = False
                self.encoder_frozen = True
                print("   🔒 TransASR.encoder frozen")
                
            elif component == 'enc':
                if hasattr(self.modules, 'enc'):
                    for param in self.modules.enc.parameters():
                        param.requires_grad = False
                    print("   🔒 Encoder projection (enc) frozen")
                    
            elif component == 'ctc_head':
                for param in self.modules.ctc_lin.parameters():
                    param.requires_grad = False
                print("   🔒 CTC head frozen")
    
    print(f"{'='*80}")
    print("✅ Component loading completed!")
    print(f"{'='*80}\n")
