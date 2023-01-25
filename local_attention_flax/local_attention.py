from typing import (Any, Callable, Optional, Tuple, Sequence)
PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import flax.linen as nn
from flax.linen.dtypes import promote_dtype

from flax.linen.initializers import zeros
from flax.linen.linear import default_kernel_init
from flax.linen.linear import DenseGeneral
from flax.linen.linear import PrecisionLike
from flax.linen.module import compact
from flax.linen.module import merge_param
from flax.linen.module import Module
from flax.linen.attention import combine_masks

import math
import functools

@functools.partial(jax.jit, static_argnames=('window_size_left', 'window_size_right', 'dropout_rate'))
def local_row(i: Array,
              q: Array, 
              k_padded: Array, 
              v_padded: Array, 
              mask: Optional[Array],
              rel_pos_emb: Optional[Array], # (window_size_left + window_size_right, hidden)
              window_size_left: int=128, 
              window_size_right: int=128, 
              dropout_rate: float=0.0, 
              dropout_rng: Optional[PRNGKey]=None) -> Array:
  """ Computes local attention for `stride` rows of the attention matrix."""

  q_slice = jnp.expand_dims(q, axis=-2)  # (..., batch, 1, hidden)
  k_slice = jax.lax.dynamic_slice_in_dim(k_padded, i, window_size_left + window_size_right, axis=-1) # (..., batch, hidden, window_size_left + window_size_right)

  attn_logits = jnp.matmul(
    q_slice,
    k_slice) / np.sqrt(q.shape[-1]) # (..., batch, 1, window_size_left + window_size_right)
  
  # Add relative position embeddings like https://arxiv.org/pdf/1809.04281.pdf sec 3.4, but without the skewing
  # Skewing is not needed because we only have 1 query row
  if rel_pos_emb is not None:
    attn_logits += jnp.matmul(q_slice, rel_pos_emb.T) # (..., batch, 1, window_size_left + window_size_right)

  
  mask_q = jnp.expand_dims(mask, axis=-2)
  del mask
  mask_qk = jax.lax.dynamic_slice_in_dim(mask_q, i, window_size_left + window_size_right, axis=-1)
  del mask_q
  attn_logits = jnp.where(~mask_qk, -1e9, attn_logits)
  del mask_qk

  attn_weights = jax.nn.softmax(attn_logits, axis=-1)

  if dropout_rate > 0.0:
    keep_prob = 1.0 - dropout_rate
    keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape) # type: ignore
    multiplier = (keep.astype(q.dtype) /
                  jnp.asarray(keep_prob, dtype=q.dtype))
    attn_weights = attn_weights * multiplier

  value = jnp.matmul(attn_weights, jax.lax.dynamic_slice_in_dim(v_padded, i, window_size_left + window_size_right, axis=-2))
  return value

@functools.partial(jax.jit, static_argnames=('window_size_left', 'window_size_right', 'dropout_rate'))
def local_attention(q: Array, 
                    k: Array, 
                    v: Array, 
                    mask: Optional[Array]=None, 
                    rel_pos_emb: Optional[Array]=None,
                    window_size_left: int=128,
                    window_size_right: int=128,
                    dropout_rate: float=0.0, 
                    dropout_rng: Optional[PRNGKey]=None) -> Array:
  """ Computes local attention, optionally windowed with a stride.

  Args:
    q: query, shape `(..., batch, seq_len, hidden)`
    k: key, shape `(..., batch, seq_len, hidden)`
    v: value, shape `(..., batch, seq_len, hidden)`
    mask: mask, shape `(..., batch, seq_len_k)` or `(..., batch, seq_len_q, seq_len_k)` or None
      Elements with value False are masked out.
    rel_pos_emb: relative positional embedding, shape `(window_size_left + window_size_right, hidden)`
    window_size_left: window size to the left of each token
    window_size_right: window size to the right of each token, set to zero for causal attention
    dropout_rate: dropout rate
    dropout_rng: dropout rng key
  
  Returns:
    output, shape `(..., batch, seq_len, hidden)`
  """

  assert q.shape == k.shape == v.shape, 'q,k,v shapes do not match'
  assert q.ndim >= 2, 'q, k, v need shape of at least seq_len, hidden' # (batch, seq_len, hidden)
  assert mask is None or mask.shape == q.shape[:-1] or mask.shape == q.shape, 'mask must be broadcastable to q'
  assert window_size_right >= 0, 'window_size_right must be non-negative'
  assert window_size_left >= 0, 'window_size_left must be non-negative'
  assert dropout_rate == 0 or dropout_rng is not None, 'dropout_rng must be provided if dropout_rate > 0'
  assert dropout_rate <= 1 and dropout_rate >= 0, 'dropout_rate must be between 0 and 1'
  assert rel_pos_emb is None or rel_pos_emb.shape == (window_size_left + window_size_right, q.shape[-1]), 'relative positional embedding must have shape (window_size_left + window_size_right, hidden)'
  
  seq_len = q.shape[-2]
  d_k = q.shape[-1]
  k_padded = jnp.pad(k, [(0, 0)] * (k.ndim - 2) + [(window_size_left, window_size_right)] + [(0, 0)], mode='constant', constant_values=0)
  k_padded = jnp.swapaxes(k_padded, -1, -2) # (..., batch, hidden, seq_len_padded)
  v_padded = jnp.pad(v, [(0, 0)] * (v.ndim - 2) + [(window_size_left, window_size_right)] + [(0, 0)], mode='constant', constant_values=0)
  if mask is None:
    mask = jnp.ones(seq_len, dtype=jnp.bool_)
  mask = jnp.pad(mask, [(0, 0)] * (mask.ndim - 1) + [(window_size_left, window_size_right)], mode='constant', constant_values=False)

  if dropout_rate > 0:
    dropout_rng = random.split(dropout_rng, seq_len)

  i = jnp.arange(0, seq_len)
  res = jax.vmap(local_row, 
                 in_axes=(0, -2, None, None, -2 if mask.shape == k_padded.shape else None, None, None, None, None, None if dropout_rate == 0 else 0), 
                 out_axes=(-2))(i, q, k_padded, v_padded, mask, rel_pos_emb, window_size_left, window_size_right, dropout_rate, dropout_rng)
  return jnp.reshape(res, res.shape[:-3] + (seq_len, d_k)) # (..., batch, 1, seq_len, hidden) -> (..., batch, seq_len, hidden)

class MultiHeadLocalAttention(Module):
  """Multi-head local attention. Each token can only attend to a fixed
     number of tokens to the left and right of it. This scales linearly along
     the sequence length, but only makes sense when the window size is
     significantly smaller than the sequence length.

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      dtype: the dtype of the computation
        (default: infer from inputs and params)
      param_dtype: the dtype passed to parameter initializers (default: float32)
      qkv_features: dimension of the key, query, and value latent space.
      out_features: dimension of the last projection
      dropout_rate: dropout rate
      kernel_init: initializer for the kernel of the Dense layers.
      bias_init: initializer for the bias of the Dense layers.
      use_bias: bool: whether pointwise QKVO dense transforms use bias.
      decode: whether to prepare and use an autoregressive cache.
      window_size_left: number of tokens to the left of the current token to attend to.
      window_size_right: number of tokens to the right of the current token to attend to.
      precision: numerical precision of the computation see `jax.lax.Precision`
      use_rel_pos_emb: whether to use relative positional embeddings like in music transformer
  """
  num_heads: int
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  dropout_rate: float = 0.
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
  use_bias: bool = True
  decode: bool = False
  window_size_left: int = 128
  window_size_right: int = 128
  precision: Optional[PrecisionLike] = None
  use_rel_pos_emb: bool = False

  @compact
  def __call__(self,
               inputs_q: Array,
               inputs_kv: Array,
               mask: Optional[Array] = None,
               prng_key: Optional[PRNGKey] = None):
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    Args:
      inputs_q: input queries of shape
        `[batch_sizes..., length, features]`.
      inputs_kv: key/value inputs of shape
        `[batch_sizes..., length, features]`.
      mask: attention mask of shape
        `[batch_sizes..., num_heads, query_length, key/value_length]`.
        Attention weights are masked out if their corresponding mask value
        is `False`. If `None`, no mask is applied.
      prng_key: PRNG key for dropout. Only pass if you need deterministic
        dropout.

    Returns:
      output of shape `[batch_sizes..., length, features]`.
    """
    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    dtype = self.dtype or inputs_q.dtype
    assert qkv_features % self.num_heads == 0, (
        'Memory dimension must be divisible by number of heads.')
    head_dim = qkv_features // self.num_heads

    dense = functools.partial(DenseGeneral,
                              axis=-1,
                              dtype=dtype,
                              param_dtype=dtype,
                              features=(self.num_heads, head_dim),
                              kernel_init=self.kernel_init,
                              bias_init=self.bias_init,
                              use_bias=self.use_bias,
                              precision=self.precision)
    # project inputs_q to multi-headed q/k/v
    # dimensions are then [batch..., length, n_heads, n_features_per_head]
    query, key, value = (dense(name='query')(inputs_q),
                         dense(name='key')(inputs_kv),
                         dense(name='value')(inputs_kv))

    # Relative positional embeddings
    if self.use_rel_pos_emb:
      rel_pos_emb = nn.Embed(num_embeddings=self.window_size_left+self.window_size_right, 
                             features=head_dim, 
                             name='rel_pos_emb',
                             param_dtype=dtype)
    else:
      rel_pos_emb = None

    # During fast autoregressive decoding, we feed one position at a time,
    # and cache the keys and values step by step.
    if self.decode:
      # detect if we're initializing by absence of existing cache data.
      is_initialized = self.has_variable('cache', 'cached_key')
      cached_key = self.variable('cache', 'cached_key',
                                 jnp.zeros, key.shape, key.dtype)
      cached_value = self.variable('cache', 'cached_value',
                                   jnp.zeros, value.shape, value.dtype)
      cache_index = self.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, dtype=jnp.int32))
      if is_initialized:
        *batch_dims, max_length, num_heads, depth_per_head = (
            cached_key.value.shape)
        # shape check of cached keys against query input
        expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
        if expected_shape != query.shape:
          raise ValueError('Autoregressive cache shape error, '
                           'expected query shape %s instead got %s.' %
                           (expected_shape, query.shape))
        # update key, value caches with our new 1d spatial slices
        cur_index = cache_index.value
        indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
        key = lax.dynamic_update_slice(cached_key.value, key, indices)
        value = lax.dynamic_update_slice(cached_value.value, value, indices)
        cached_key.value = key
        cached_value.value = value
        cache_index.value = cache_index.value + 1
        # causal mask for cached decoder self-attention:
        # our single query position should only attend to those key
        # positions that have already been generated and cached,
        # not the remaining zero elements.
        mask = combine_masks(
            mask,
            jnp.broadcast_to(jnp.arange(max_length) <= cur_index,
                             tuple(batch_dims) + (1, 1, max_length)))

    if self.dropout_rate > 0. and prng_key is None:  # Require `deterministic` only if using dropout.
        prng_key = self.make_rng('dropout')

    # apply attention
    x = local_attention(
        jnp.swapaxes(query, -3, -2), # [batch..., n_heads, length, n_features_per_head]
        jnp.swapaxes(key, -3, -2), # [batch..., n_heads, length, n_features_per_head]
        jnp.swapaxes(value, -3, -2), # [batch..., n_heads, length, n_features_per_head]
        mask=jnp.swapaxes(mask, -3, -2) if mask is not None else None,
        rel_pos_emb=rel_pos_emb.embedding,
        dropout_rng=prng_key,
        dropout_rate=self.dropout_rate,
        window_size_left=self.window_size_left,
        window_size_right=self.window_size_right)

    x = jnp.swapaxes(x, -3, -2)

    # back to the original inputs dimensions
    out = DenseGeneral(features=features,
                       axis=(-2, -1),
                       kernel_init=self.kernel_init,
                       bias_init=self.bias_init,
                       use_bias=self.use_bias,
                       dtype=dtype,
                       param_dtype=dtype,
                       precision=self.precision,
                       name='out')(x)
    return out



class LocalSelfAttention(MultiHeadLocalAttention):
  """Self-attention special case of multi-head dot-product attention."""

  @compact
  def __call__(self, inputs_q: Array, mask: Optional[Array] = None, # type: ignore
               prng_key: Optional[PRNGKey] = None) -> Array:
    """Applies multi-head dot product self-attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    Args:
      inputs_q: input queries of shape
        `[batch_sizes..., length, features]`.
      mask: attention mask of shape
        `[batch_sizes..., num_heads, query_length, key/value_length]`.
        Attention weights are masked out if their corresponding mask value
        is `False`.
      prng_key: PRNG key for dropout. Only pass if you need deterministic
        dropout.

    Returns:
      output of shape `[batch_sizes..., length, features]`.
    """
    return super().__call__(inputs_q, inputs_q, mask,
                            prng_key=prng_key)