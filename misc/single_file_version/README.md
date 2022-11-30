# README

- Author: Arthur Lui
- Email: luiarthur@gmail.com

## Demo
The demo (`demo.py`) was used to help me understand the internals of the PPL.
It might be useful to you too.

## Source code
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

## Notes
This version implements a global effect handler stack, `_stack` instead of
storing the `stack` for each `AbstractModel`. The reason is that having a
global stack is good enough in python. The concern would be that having a
global stack will cause parallel writes to the same stack when models are being
fit in parallel. This does not happen, though, because python will make a copy
of `_stack` when parallelized via something like `multiprocessing`. This is,
however, not true in other languages, like Julia and JavaScript. I have
therefore kept the more translatable version in the root of the repo. But,
having a global stack is preferred in python because it is sufficient and
simpler.
