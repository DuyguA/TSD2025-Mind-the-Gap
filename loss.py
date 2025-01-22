import torch
import torch.nn.functional as F

def custom_loss(predictions, targets, mask, shift_penalty=1.0):
    """
    Custom loss function for Whisper fine-tuning with token shifting.

    Args:
        predictions (Tensor): Predicted logits (batch_size, seq_len, vocab_size).
        targets (Tensor): Ground truth token indices (batch_size, seq_len).
        mask (Tensor): Binary mask indicating valid positions for mid chunk.
        shift_penalty (float): Penalty weight for premature/delayed outputs.

    Returns:
        Tensor: Loss value.
    """
    # Calculate cross-entropy loss for all tokens
    ce_loss = F.cross_entropy(predictions.view(-1, predictions.size(-1)), targets.view(-1), reduction='none')
    ce_loss = ce_loss.view(targets.size())  # Reshape to (batch_size, seq_len)

    # Apply the mask: only include valid tokens for the mid chunk
    masked_loss = ce_loss * mask

    # Calculate shift penalties (optional)
    # Penalize premature or delayed outputs by applying an additional penalty
    shift_loss = shift_penalty * (1 - mask) * ce_loss

    # Combine the masked loss and shift penalty
    total_loss = masked_loss.sum() + shift_loss.sum()
    return total_loss / mask.sum()


import jax
import jax.numpy as jnp

def flax_custom_loss(logits, targets, mask, shift_penalty=1.0):
    """
    Flax-compatible custom loss function for boundary-aware token shifting.

    Args:
        logits (jnp.ndarray): Predicted logits (batch_size, seq_len, vocab_size).
        targets (jnp.ndarray): Ground truth token indices (batch_size, seq_len).
        mask (jnp.ndarray): Binary mask (batch_size, seq_len) indicating valid tokens for the mid chunk.
        shift_penalty (float): Penalty weight for premature/delayed outputs.

    Returns:
        jnp.ndarray: Scalar loss value.
    """
    # Compute log-softmax for numerical stability
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    # Gather the log-probabilities of the target tokens
    target_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)

    # Calculate the negative log-likelihood loss
    nll_loss = -target_log_probs

    # Mask the loss to only consider valid tokens in the mid chunk
    masked_loss = nll_loss * mask

    # Calculate shift penalties for tokens outside the valid mask
    shift_loss = shift_penalty * nll_loss * (1.0 - mask)

    # Combine masked loss and shift penalties
    total_loss = masked_loss.sum() + shift_loss.sum()

    # Normalize by the number of valid tokens in the mask
    normalizer = mask.sum()
    return total_loss / normalizer
