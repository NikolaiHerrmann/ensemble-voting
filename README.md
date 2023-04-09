# Comparison of Voting Rules for Ensemble Methods


### Debug

If the following error occurs:

```AttributeError: module 'collections' has no attribute 'Iterable'```

Proceed to change line 15 in `corankco/algorithms/bioconsert.py` from `collections.Iterable` to `collections.abc.Iterable`