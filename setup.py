from setuptools import setup, find_packages

setup(
  name = 'local_attention_flax',
  packages = find_packages(),
  version = '0.0.1',
  license='MIT',
  description = 'The local attention mechanism from Longformer, implemented in flax and without the need for windowing',
  long_description_content_type = 'text/markdown',
  author = 'Nico Westerbeck',
  url = 'https://github.com/blacksph3re/local_attention_flax',
  keywords = [
    'transformers',
    'attention',
    'artificial intelligence',
    'local attention',
    'linear attention'
  ],
  install_requires=[
    'jax',
    'flax',
    'numpy'
  ]
)
