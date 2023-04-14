# Comparison of Voting Rules for Ensemble Methods

### Structure

- Files `vorace.py` and `varace_agent.py`(vorace_agent.py) have been adapted from the original [VORACE](https://github.com/aloreggia/vorace) repository. See the actual [paper](https://link.springer.com/article/10.1007/s10458-021-09504-y).

- `simulations.ipynb` runs our experiments

- `plots.py` shows results from simulations

- `summary_plots.py` summarizes results from original paper

- `stv.py` implements stv rule integrated with `vorace.py`

- `data/` contains datasets (.csv) and our results (.json)

### Run Instructions 

Install the Python [requirements.txt](requirements.txt). Version number for [`corankco`](https://github.com/pierreandrieu/corankco) and `numpy` are important.

Then run the notebook [simulations.ipynb](simulations.ipynb). Run `plots.py` for visualizations.

### Debug

If the following error occurs:

```AttributeError: module 'collections' has no attribute 'Iterable'```

Proceed to change line 15 in `corankco/algorithms/bioconsert.py` (find pip installation location) from `collections.Iterable` to `collections.abc.Iterable`