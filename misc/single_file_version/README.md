
# README

- Author: Arthur Lui
- Email: luiarthur@gmail.com

# Demo
The demo (`demo.py`) was used to help me understand the internals of the PPL.
It might be useful to you too.

# Source code
The source is very short -- fewer than 150 lines total.
The core part of the PPL is fewer than 100 lines.

This version of the PPL doesn't have an inference algorithm, it only has the
`trace` and `condition` functionality. With these, you can write some generic
inference algorithms, like random walk Metropolis, to do inference.

```
ppl
├── __init__.py
├── core.py (<100 lines)
└── distributions.py (~30 lines)
```
