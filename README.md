# Flax local attention

This is an implementation of the local attention mechanism from longformers for language modelling. Instead of using windowed attention like the implementation by [lucidrains](https://github.com/lucidrains/local-attention-flax), this repository leverages jax `vmap` functionality to iterate over every query and make sure every query sees exactly window_size keys. I wrote this as a challenge to learn jax, hence I am grateful for feedback where I could have implemented something in a different way.

# Install

```
pip install 'local_attention_flax @ git+https://github.com/blacksph3re/local_attention_flax'
```

# Usage

You can use the `local_attention` function which implements the raw attention calculation based on given `q`, `k`, `v`:

```
batch_size = 32
num_heads = 2
seq_len = 2048
hidden_dim = 64

key = jax.random.PRNGKey(1)

q = jax.random.normal(key, (batch_size, num_heads, seq_len, hidden_dim))
k = jax.random.normal(key, (batch_size, num_heads, seq_len, hidden_dim))
v = jax.random.normal(key, (batch_size, num_heads, seq_len, hidden_dim))

x = local_attention(q, k, v, mask=None, rel_pos_emb=None, window_size_left=128, window_size_right=128, dropout_rate=0.1, dropout_rng = key)
```

Alternatively, you can use the wrapper flax modules that include Dense layers for self- or crossattention. `LocalSelfAttention` is comparable to [flax SelfAttention](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.SelfAttention.html) and `MultiHeadLocalAttention` is comparable to [flax MultiHeadDotProductAttention](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.MultiHeadDotProductAttention.html). I changed the mechanics dealing with random state, and allow the user to optionally pass a prng key which will run the dropout computation deterministically with that key. Flax decided to instead pass a deterministic flag, which allows for only one deterministic result.

```
batch_size = 32
seq_len = 2048
hidden_dim = 64

key = jax.random.PRNGKey(1)

att = LocalSelfAttention(qkv_features=hidden_dim, out_features=hidden_dim, num_heads=2, window_size_left=128, window_size_right=128, dropout_rate=0.1)

x = jax.random.normal(key, (batch_size, seq_len, hidden_dim))

params = att.init(key, x, prng_key=key)
x = att.apply(params, x, mask=None, prng_key=key)
```