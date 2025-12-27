"""Hook management utilities for steering and ablation."""


def _is_decoded_token(seq_len):
    """Check if we're processing decoded tokens (not prompt tokens).
    During prefill: seq_len == input_len (all positions are prompt)
    During generation: seq_len == 1 (only newest token, which is decoded)
    """
    return seq_len == 1


def clear_all_hooks(model):
    """Clear all hooks from model."""
    model.model.embed_tokens._forward_hooks.clear()
    if hasattr(model.model, 'norm'):
        model.model.norm._forward_pre_hooks.clear()
        model.model.norm._forward_hooks.clear()
    
    for layer in model.model.layers:
        layer._forward_pre_hooks.clear()
        layer._forward_hooks.clear()
        layer.self_attn.q_proj._forward_hooks.clear()
        layer.self_attn.k_proj._forward_hooks.clear()
        layer.self_attn.v_proj._forward_hooks.clear()
        layer.self_attn.o_proj._forward_pre_hooks.clear()
        layer.self_attn.o_proj._forward_hooks.clear()
        layer.mlp._forward_hooks.clear()
        layer.mlp._forward_pre_hooks.clear()


def register_steering(model, location_type, layer_idx, vector, alpha=1.0):
    """Register steering hook. Vector is added to ALL positions during generation.
    
    Args:
        model: Model instance
        location_type: "residual", "mlp", "attention", "q_layer"
        layer_idx: Layer index (0-indexed)
        vector: Steering vector tensor
        alpha: Scaling factor for vector
    
    Returns:
        Hook handle
    """
    clear_all_hooks(model)
    scaled_vector = vector * alpha if alpha != 0 else None
    num_layers = len(model.model.layers)
    
    if scaled_vector is None:
        return None
    
    if location_type == "residual":
        if layer_idx == 0:
            def hook_embed(module, args, output):
                out = output.clone()
                seq_len = out.shape[1]
                if _is_decoded_token(seq_len):
                    out[:, :, :] += scaled_vector.to(out.device)
                return out
            return model.model.embed_tokens.register_forward_hook(hook_embed)
        elif layer_idx == num_layers:
            def hook_norm(module, inputs):
                hidden_states = inputs[0].clone()
                seq_len = hidden_states.shape[1]
                if _is_decoded_token(seq_len):
                    hidden_states[:, :, :] += scaled_vector.to(hidden_states.device)
                return (hidden_states,) + inputs[1:]
            return model.model.norm.register_forward_pre_hook(hook_norm)
        else:
            def hook(module, inputs):
                hidden_states = inputs[0].clone()
                seq_len = hidden_states.shape[1]
                if _is_decoded_token(seq_len):
                    hidden_states[:, :, :] += scaled_vector.to(hidden_states.device)
                return (hidden_states,) + inputs[1:]
            return model.model.layers[layer_idx].register_forward_pre_hook(hook)
    
    elif location_type == "mlp":
        def hook(module, args, output):
            out = output.clone()
            seq_len = out.shape[1]
            if _is_decoded_token(seq_len):
                out[:, :, :] += scaled_vector.to(out.device)
            return out
        return model.model.layers[layer_idx].mlp.register_forward_hook(hook)
    
    elif location_type == "attention":
        def hook(module, args, output):
            out = output.clone()
            seq_len = out.shape[1]
            if _is_decoded_token(seq_len):
                out[:, :, :] += scaled_vector.to(out.device)
            return out
        return model.model.layers[layer_idx].self_attn.o_proj.register_forward_hook(hook)
    
    elif location_type == "q_layer":
        def hook(module, args, output):
            out = output.clone()
            seq_len = out.shape[1]
            if _is_decoded_token(seq_len):
                out[:, :, :] += scaled_vector.to(out.device)
            return out
        return model.model.layers[layer_idx].self_attn.q_proj.register_forward_hook(hook)
    
    else:
        raise ValueError(f"Unknown location_type: {location_type}")


def register_ablation(model, layer_idx, head_idx, ablation_mode, head_dim=None):
    """Register ablation hook for specified layer/head/mode.
    
    Args:
        model: Model instance
        layer_idx: Layer index (0-indexed)
        head_idx: Head index (None for layer-level ablations)
        ablation_mode: "q", "o_proj", "q_layer", "o_proj_layer", "mlp", "residual", "none"
        head_dim: Head dimension (inferred from model config if None)
    
    Returns:
        Hook handle or None
    """
    clear_all_hooks(model)
    
    if ablation_mode == "none":
        return None
    
    if head_dim is None:
        head_dim = model.config.hidden_size // model.config.num_attention_heads
    
    layer = model.model.layers[layer_idx]
    
    if ablation_mode == "q":
        if head_idx is None:
            raise ValueError("head_idx required for single-head Q ablation")
        def hook(module, args, output):
            out = output.clone()
            s = head_idx * head_dim
            e = (head_idx + 1) * head_dim
            seq_len = out.shape[1]
            if _is_decoded_token(seq_len):
                out[:, :, s:e] = 0
            return out
        return layer.self_attn.q_proj.register_forward_hook(hook)
    
    elif ablation_mode == "o_proj":
        if head_idx is None:
            raise ValueError("head_idx required for single-head o_proj ablation")
        def pre_hook(module, inputs):
            x = inputs[0].clone()
            s = head_idx * head_dim
            e = (head_idx + 1) * head_dim
            seq_len = x.shape[1]
            if _is_decoded_token(seq_len):
                x[:, :, s:e] = 0
            return (x,) + inputs[1:]
        return layer.self_attn.o_proj.register_forward_pre_hook(pre_hook)
    
    elif ablation_mode == "q_layer":
        def hook(module, args, output):
            out = output.clone()
            seq_len = out.shape[1]
            if _is_decoded_token(seq_len):
                out[:, :, :] = 0
            return out
        return layer.self_attn.q_proj.register_forward_hook(hook)
    
    elif ablation_mode == "o_proj_layer":
        def hook(module, inputs):
            inp = inputs[0].clone()
            seq_len = inp.shape[1]
            if _is_decoded_token(seq_len):
                inp[:, :, :] = 0
            return (inp,) + inputs[1:]
        return layer.self_attn.o_proj.register_forward_pre_hook(hook)
    
    elif ablation_mode == "mlp":
        def hook(module, args, output):
            out = output.clone()
            seq_len = out.shape[1]
            if _is_decoded_token(seq_len):
                out[:, :, :] = 0
            return out
        return layer.mlp.register_forward_hook(hook)
    
    elif ablation_mode == "residual":
        def hook(module, inputs):
            hidden_states = inputs[0].clone()
            seq_len = hidden_states.shape[1]
            if _is_decoded_token(seq_len):
                hidden_states[:, :, :] = 0
            return (hidden_states,) + inputs[1:]
        return layer.register_forward_pre_hook(hook)
    
    else:
        raise ValueError(f"Unknown ablation mode: {ablation_mode}")

