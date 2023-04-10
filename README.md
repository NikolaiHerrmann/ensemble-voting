# Comparison of Voting Rules for Ensemble Methods

Files [vorace.py](vorace.py) and [varace_agent.py](vorace_agent.py) have been adapted from the original [VORACE](https://github.com/aloreggia/vorace) repository paper.

### Run Instructions 

Install the Python [requirements.py](requirements.py). Version number for [corankco](https://github.com/pierreandrieu/corankco) and `numpy` are important.

Then run the notebooks.

### Debug

If the following error occurs:

```AttributeError: module 'collections' has no attribute 'Iterable'```

Proceed to change line 15 in `corankco/algorithms/bioconsert.py` (find pip installation location) from `collections.Iterable` to `collections.abc.Iterable`