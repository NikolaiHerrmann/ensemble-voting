
import numpy as np


def re_order(votes):
    sorted_votes = np.flip(np.argsort(votes))
    prev = -1
    tmp = ()
    pref = ()

    for class_ in sorted_votes:
        for rank, val in enumerate(votes):
            if np.isclose(votes[class_], val):

                if rank == prev:
                    tmp += (class_,)
                else:
                    if tmp:
                        pref += (tmp,)
                    tmp = (class_,)
                prev = rank

                break
    pref += (tmp,)

    return pref


def collect_votes(ballots):
    all_votes = {}

    for x in ballots:
        pref = re_order(x)
        if pref in all_votes:
            all_votes[pref] += 1
        else:
            all_votes[pref] = 1

    return all_votes


def calculate_plurality(votes, alternatives):
	"""
	Determine the plurality scores of all alternatives
	Returns dict containing alternatives with their plurality scores
	"""
	scores = {i: 0 for i in alternatives}

	for ballot, n in votes.items():
		num_ties = len(ballot[0])
		for alt in ballot[0]:
			scores[alt] += n * (1 / num_ties) # divide incase of tie

	return scores


def find_least(scores):
	"""
    Find the alternative with the lowest plurality scores
    Returns string representing the name of alternatives with the least plurality scores
    """
	minimum = min(scores.values())
	keys = [k for k, v in scores.items() if v == minimum]
	return keys


def remove_alt(votes, alt):
	"""
	Filter out an alternative from the votes dictionary
	"""
	new_votes = {}

	for pref, n_votes in votes.items():
		new_pref = ()
		for x in pref:
			item = tuple(i for i in x if i != alt)
			if item:
				new_pref += (item,)
		if new_pref:
			if new_pref in new_votes:
				new_votes[new_pref] += n_votes
			else:
				new_votes[new_pref] = n_votes

	return new_votes


def stv_election(votes, n):
	"""
	Run a standard STV election
	"""
	a = [i for i in range(n)]
	a_cpy = a.copy()

	while len(a_cpy) != 1:

		scores = calculate_plurality(votes, a_cpy)
		alts = find_least(scores)

		for alt in alts:
			votes = remove_alt(votes, alt)
			a_cpy.remove(alt)
		if len(a_cpy) == 0:
			return alts
	
	return a_cpy


def stv_rule(preferences):
    n_samples, n_models, n_classes = preferences.shape
    scores = np.zeros((n_samples, n_classes), dtype="i2")
    
    for i, sample in enumerate(preferences):
        votes = collect_votes(sample)
        winner = stv_election(votes, n_classes)
        for j in winner:
            scores[i][j] = 1

    return scores
