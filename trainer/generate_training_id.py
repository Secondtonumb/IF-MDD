"""
MDD (Mispronunciation Detection and Diagnosis) System - Training ID Generator

Author: Haopeng (Kevin) Geng
Institution: University of Tokyo
Year: 2025

This code is provided for non-commercial use only.
For commercial use, please contact the author.

This module generates unique training IDs based on hyperparameters.
"""

import hyperpyyaml

def generate_training_id(perceived_ssl_model_id, canonical_ssl_model_id, feature_fusion, prefix):
    """
    Generate a unique training ID based on key hyperparameters.
    """
    if perceived_ssl_model_id is not None:
        perceived_ssl_model = perceived_ssl_model_id.split("/")[-1]
    else:
        perceived_ssl_model = None
    if canonical_ssl_model_id is not None:  
        canonical_ssl_model = canonical_ssl_model_id.split("/")[-1]
    else:
        canonical_ssl_model = None
    # get stem of the model name
    feature_fusion = feature_fusion
    # if feature_fusion is mono, assume only one ssl model is used
    if feature_fusion == "mono":
        if perceived_ssl_model is None and canonical_ssl_model is not None:
            feature_fusion = "canonical_only"
        elif perceived_ssl_model is not None and canonical_ssl_model is None:
            feature_fusion = "perceived_only"
        elif perceived_ssl_model is not None and canonical_ssl_model is not None:
            # warning, should assign one ssl encoder only, default use perceived_ssl
            feature_fusion = "mono"
            print("Warning: should assign one ssl encoder only, default use perceived_ssl")
        elif perceived_ssl_model is None and canonical_ssl_model is None:
            return None
    
    if prefix == "":
        prefix = None

    training_id = None
    # omit null values
    if prefix is None and perceived_ssl_model != None and canonical_ssl_model != None:
        training_id = f"{perceived_ssl_model}_{canonical_ssl_model}_{feature_fusion}/"
    elif prefix is None and perceived_ssl_model != None:
        training_id = f"{perceived_ssl_model}_{feature_fusion}/"
    elif prefix is None and canonical_ssl_model != None:
        training_id = f"{canonical_ssl_model}_{feature_fusion}/"
    elif prefix != None and perceived_ssl_model != None and canonical_ssl_model != None:
        training_id = f"{prefix}_{perceived_ssl_model}_{canonical_ssl_model}_{feature_fusion}/"
    elif prefix != None and perceived_ssl_model != None:
        training_id = f"{prefix}_{perceived_ssl_model}_{feature_fusion}/"
    elif prefix != None and canonical_ssl_model != None:
        training_id = f"{prefix}_{canonical_ssl_model}_{feature_fusion}/"
    else:
        raise ValueError("No SSL model selected")
    print(training_id)
    return training_id

def get_pretrained_model_id(hparams, model_name):
    x = getattr(hparams, model_name, "Null")
    print(x)
    return x