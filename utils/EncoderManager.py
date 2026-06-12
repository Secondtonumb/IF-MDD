# ============================================================================
# Unified Encoder Manager
# ============================================================================
from torch import nn

class EncoderManager(nn.Module):
    """
    Unified encoder manager that handles different encoder types:
    - None: Direct pass-through
    - Linear: Simple linear projection
    - Conformer: Conformer encoder with relative positional encoding
    - Zipformer: Zipformer encoder
    - RVQ: Residual Vector Quantization
    
    Usage:
        encoder = EncoderManager(
            encoder_type='conformer',  # or 'linear', 'zipformer', 'rvq', None
            modules=self.modules,
            hparams=self.hparams,
            device=self.device
        )
        output = encoder(features, wav_lens)
    """
    
    def __init__(self, encoder_type, modules, hparams, device, **kwargs):
        super().__init__()
        self.encoder_type = encoder_type
        self.modules = modules
        self.hparams = hparams
        self.device = device
        
        # Store references to relevant modules
        self.enc = getattr(modules, 'enc', None)
        # Alternative SSL_proj which works equal as enc
        self.ssl_proj = getattr(modules, 'ssl_proj', None)
        self.conformer_encoder = getattr(modules, 'ConformerEncoder', None)
        self.zipformer_encoder = getattr(modules, 'ZipformerEncoder', None)
        self.rvq = getattr(modules, 'RVQ', None)
        
        self.kwargs = kwargs

    def forward(self, features, wav_lens=None):
        """
        Args:
            features: [B, T, D] SSL features
            wav_lens: [B] relative lengths (optional)
            
        Returns:
            encoded: [B, T, D] encoded features
            extras: dict with additional outputs (commitment_loss, codebook_loss, etc.)
        """
        extras = {}
        x = features
        # Step 1: Linear projection (enc)
        
        if self.enc is not None:
            x = self.enc(x)
        
        # Step 1 Option: SSL projection
        if self.ssl_proj is not None:
            x = self.ssl_proj(x)

        # Step 2: Apply specific encoder type
        # import pdb; pdb.set_trace()
        if self.encoder_type == 'conformer' and self.conformer_encoder is not None:
            if self.conformer_encoder.attention_type == "RelPosEncXL":
                from speechbrain.nnet.attention import RelPosEncXL
                try:
                    pos_emb = RelPosEncXL(emb_dim=self.hparams.dnn_neurons)(x).to(self.device)
                except Exception as e:
                    pos_emb = RelPosEncXL(emb_dim=self.kwargs.get('dnn_neurons', 384))(x).to(self.device)
                x, _ = self.conformer_encoder(x, pos_embs=pos_emb)
            else:
                x, _ = self.conformer_encoder(x)
            
        elif self.encoder_type == 'zipformer' and self.zipformer_encoder is not None:
            x = self.zipformer_encoder(x.permute(1, 0, 2))  # [T, B, D]
            x = x.permute(1, 0, 2)  # [B, T, D]
        
        # Step 3: Optional RVQ
        if self.encoder_type == 'rvq' and self.rvq is not None:
            x = x.transpose(1, 2)  # [B, T, D] -> [B, D, T]
            discrete_embeddings, codes, latents, commitment_loss, codebook_loss = self.rvq(x)
            x = discrete_embeddings.transpose(1, 2)  # [B, D, T] -> [B, T, D]
            extras['commitment_loss'] = commitment_loss
            extras['codebook_loss'] = codebook_loss
            extras['discrete_embeddings'] = discrete_embeddings
            extras['codes'] = codes
            
        return x, extras


class AudioEncoder(nn.Module):
    """
    A simple audio encoder that applies a series of convolutional layers
    to extract features from raw audio input.
    """
    def __init__(self, modules, **kwargs):
        super().__init__()
        self.kwargs = kwargs

        self.perceived_ssl = getattr(modules, 'perceived_ssl', None)
        self.encoder_manager = self.kwargs.get('EncoderManager', None)
        
        
    
    def forward(self, audio_waveform):
        """
        Args:
            audio_waveform: [B, 1, T] raw audio waveform
            
        Returns:
            encoded_features: [B, D, T'] encoded features
        """
        # Step 1: Extract SSL features
        if self.perceived_ssl is not None:
            ssl_features = self.perceived_ssl(audio_waveform)  # [B, T, D]
        else:
            raise ValueError("perceived_ssl module is not defined.")
        
        # Step 2: Pass through EncoderManager
        encoded_features, extras = self.encoder_manager(ssl_features)  # [B, T, D]
        
        
        return encoded_features, extras