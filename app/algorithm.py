import logging
import numpy as np
from numpy.linalg import norm
from typing import List
from scipy.spatial.distance import pdist, cdist
from collections import namedtuple
Segmentation = namedtuple('Segmentation',
                          'total splits gains min_gain optimal duration_penalty penalty')


def split_greedy(docmat, penalty=None, max_splits=None):
    """
    Iteratively segment a document into segments being greedy about the
    next choice. This gives very accurate results on crafted documents, i.e.
    artificial concatenations of random documents.

    `penalty` is the minimum quantity a split has to improve the score to be
    made. If not given `total` is not computed.
    `max_splits` is a limit on the number of splits.
    Either `penalty` or `max_splits` have to be given.

    Whenever the iteration reaches the while block the following holds:
    `cuts` == splits + [L] where splits are the segment start indices
    `segscore` maps all segment start indices to segment vector lengths
    `score_l[i]` is the cumulated vector length from the cut left of i to i
    `score_r[i]` is the cumulated vector length from i to the cut right of i
    `score_out[i]` is the sum of all segscores not including the segment at i
    `scores[i]` is the sum of all segment vector lengths if we split at i

    These quantities are repaired after determining a next split from `scores`.

    Returns `total`, `splits`, `gains` where
    - `total` is the score diminished by len(splits) * penalty to make it
      continuous in the input. It is comparable to the output of split_optimal.
    - `splits` is the list of splits
    - `gains` is a list of uplift each split contributes vs. leaving it out

    Note: The splitting strategy suggests all resulting splits will have gain at
    least `penalty`. This is not the case as new splits can decrease the gain
    of others. This can be repaired by blocking positions where a split would
    decreas the gain of an existing one to less than `penalty` but is not
    impemented here.
    """
    L, dim = docmat.shape

    assert max_splits is not None or (penalty is not None and penalty > 0)

    # norm(cumvecs[j] - cumvecs[i]) == norm(w_i + ... + w_{j-1})
    cumvecs = np.cumsum(np.vstack((np.zeros((1, dim)), docmat)), axis=0)

    # cut[0] seg[0] cut[1] seg[1] ... seg[L-1] cut[L]
    cuts = [0, L]
    segscore = dict()
    segscore[0] = norm(cumvecs[L, :] - cumvecs[0, :], ord=2)
    segscore[L] = 0  # corner case, always 0
    score_l = norm(cumvecs[:L, :] - cumvecs[0, :], axis=1, ord=2)
    score_r = norm(cumvecs[L, :] - cumvecs[:L, :], axis=1, ord=2)
    score_out = np.zeros(L)
    score_out[0] = -np.inf  # forbidden split position
    score = score_out + score_l + score_r

    min_gain = np.inf
    while True:
        split = np.argmax(score)

        if score[split] == -np.inf:
            break

        cut_l = max([c for c in cuts if c < split])
        cut_r = min([c for c in cuts if split < c])
        split_gain = score_l[split] + score_r[split] - segscore[cut_l]
        if penalty is not None:
            if split_gain < penalty:
                break

        min_gain = min(min_gain, split_gain)

        segscore[cut_l] = score_l[split]
        segscore[split] = score_r[split]

        cuts.append(split)
        cuts = sorted(cuts)

        if max_splits is not None:
            if len(cuts) >= max_splits + 2:
                break

        # differential changes to score arrays
        score_l[split:cut_r] = norm(
            cumvecs[split:cut_r, :] - cumvecs[split, :],
            axis=1, ord=2
        )
        score_r[cut_l:split] = norm(
            cumvecs[split, :] - cumvecs[cut_l:split, :],
            axis=1, ord=2
        )

        # adding following constant not necessary, only for score semantics
        score_out += split_gain
        score_out[cut_l:split] += segscore[split] - split_gain
        score_out[split:cut_r] += segscore[cut_l] - split_gain
        score_out[split] = -np.inf

        # update score
        score = score_out + score_l + score_r

    cuts = sorted(cuts)
    splits = cuts[1:-1]
    if penalty is None:
        total = None
    else:
        total = sum(
            norm(cumvecs[left, :] - cumvecs[right, :], ord=2)
            for left, right in zip(cuts[:-1], cuts[1:])) - len(splits) * penalty
    gains = []
    for beg, cen, end in zip(cuts[:-2], cuts[1:-1], cuts[2:]):
        no_split_score = norm(cumvecs[end, :] - cumvecs[beg, :], ord=2)
        gains.append(segscore[beg] + segscore[cen] - no_split_score)

    return Segmentation(total, splits, gains, min_gain=min_gain, optimal=None, penalty=None, duration_penalty=None)


def semantic_split(
    docmat,
    penalty: float,
    seg_limit: int = None,
    K: int = 10,
    smooth: int = 0,
    dump_results: bool = False
):

    """
    SEMANTIC SEGMENTATION OF TEXT USING DEEPLEARNING
    http://147.213.75.17/ojs/index.php/cai/article/view/2022_1_78/1137

    Determine the configuration of splits with the highest score, given that
    splitting has a cost of `penalty`. `seg_limit` is a limitation on the length
    of a segment that saves memory and computation, but gives poor results
    should there be no split withing the range.

    K = how many segments
    smooth = average window size

    These quantities are repaired after determining a next split from `scores`.

    Returns `total`, `splits`, `gains` where
    - `total` is the score diminished by len(splits) * penalty to make it
      continuous in the input. It is comparable to the output of split_optimal.
    - `splits` is the list of splits
    - `gains` is a list of uplift each split contributes vs. leaving it out

    Note: The splitting strategy suggests all resulting splits will have gain at
    least `penalty`. This is not the case as new splits can decrease the gain
    of others. This can be repaired by blocking positions where a split would
    decreas the gain of an existing one to less than `penalty` but is not
    impemented here.
    """

    def running_mean(x, N):
        x[x == -np.inf] = min(np.min(x[x != -np.inf]), 0)
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    L, dim = docmat.shape
    lim = L if seg_limit is None else seg_limit
    assert lim > 0
    assert penalty > 0
    acc = np.full((L, lim), -np.inf, dtype=np.float32)
    colmax = np.full((L,), -np.inf, dtype=np.float32)

    distance_function = 'euclidean'
    for i in range(L):
        for j in range(max(i - lim, 0), i):
            for k in range(i, min(i + lim, L)):
                intra = 0
                intra += np.mean(pdist(docmat[j:i, :], distance_function))
                intra += np.mean(pdist(docmat[i:k, :], distance_function))
                inter = 0
                inter += np.mean(cdist(docmat[j:i, :], docmat[i:k, :], distance_function))
                colmax[i] = max(colmax[i], intra - inter)

    if smooth:
        colmax = running_mean(colmax, smooth)

    total = np.sum(colmax)
    splits = sorted(np.argsort(colmax)[::-1][:K])
    gains = [colmax[i] for i in splits]
    optimal = True
    acc[acc == -np.inf] = 0

    if dump_results:
        import pickle
        with open("tmp.pkl", "wb") as f:
            pickle.dump({
                'splits': splits,
                'gains': gains,
                'acc': acc,
                'optimal': optimal,
                'total': total,
                'docmat': docmat,
                'colmax': colmax
            }, f)

    return Segmentation(total, splits, gains, min_gain=None, optimal=optimal)


def split_optimal(
    docmat,
    penalty: float,
    duration_penalty: List[float],
    seg_limit=None,
    intra_weight: float = 1.0,
    dump_results: bool = False,
    **kwargs
):
    """
    Determine the configuration of splits with the highest score, given that
    splitting has a cost of `penalty`. `seg_limit` is a limitation on the length
    of a segment that saves memory and computation, but gives poor results
    should there be no split withing the range.
    The algorithm is built upon the idea that there is a accumulated score
    matrix containing the maximal score of creating a segment (i, j), containing
    all words [w_i, ..., w_j] at position i, j. The matrix `acc` is indexed to
    contain the first `seg_limit` elements of each row of the score matrix.
    `colmax` contains the column maxima of the score matrix.
    `ptr` is a backtracking pointer to determine the splits made while
    forward accumulating the highest score in the score matrix.
    """
    L, dim = docmat.shape
    lim = L if seg_limit is None else seg_limit
    intra_weight = max(min(intra_weight, 0), 1.0)
    assert lim > 0
    assert penalty > 0

    S = np.full((L, lim), 0, dtype=np.float32)
    D = np.full((L, lim), 0, dtype=np.float32)
    P = np.zeros(L, dtype=np.int32)

    acc = np.full((L, lim), -np.inf, dtype=np.float32)
    acc2 = np.full((L, lim), -np.inf, dtype=np.float32)
    colmax = np.full((L,), -np.inf, dtype=np.float32)
    colmax2 = np.full((L,), -np.inf, dtype=np.float32)
    ptr = np.zeros(L, dtype=np.int32)

    for i in range(L):
        score_so_far = colmax[i - 1] if i > 0 else 0.

        ctxvecs = np.cumsum(docmat[i:i + lim, :], axis=0)
        winsz = ctxvecs.shape[0]
        score = norm(ctxvecs, axis=1, ord=2)

        intra_score = []
        for j in range(i, min(i + lim, L)):
            intra_score.append(np.mean(cdist(docmat[ptr[i]:i, :], docmat[i:j + 1, :], 'euclidean')))
        intra_score = np.array(intra_score, dtype="float32")
        intra_score[np.isnan(intra_score)] = 0.0
        intra_score *= intra_weight
        intra_score -= np.mean(intra_score)

        acc[i, :winsz] = score_so_far - penalty + score - intra_score - duration_penalty[i]
        deltas = np.where(acc[i, :winsz] > colmax[i:i + lim])[0]
        js = i + deltas
        colmax[js] = acc[i, deltas]
        ptr[js] = i
        S[i, :winsz] = score_so_far
        D[i, :winsz] = score
        acc2[i, :winsz] = score_so_far + score - intra_score + np.mean(intra_score)
        colmax2[js] = acc2[i, deltas]

    path = [ptr[-1]]
    while path[0] != 0:
        path.insert(0, ptr[path[0] - 1])

    splits = path
    gains = get_gains(docmat, path[1:])
    optimal = all(np.diff([0] + splits + [L]) < lim)
    total = colmax[-1] + penalty
    P = ptr * -penalty

    if dump_results:
        import pickle
        with open("tmp.pkl", "wb") as f:
            pickle.dump({
                'splits': splits,
                'gains': gains,
                'acc': acc2,
                'optimal': optimal,
                'total': total,
                'S': S,
                'D': D,
                'ptr': ptr,
                'P': P,
                'docmat': docmat,
                'colmax': colmax2
            }, f)

    logging.info(splits)
    return Segmentation(total, splits, gains, min_gain=None, optimal=optimal, duration_penalty=duration_penalty, penalty=penalty)


def split_exact(
    docmat,
    penalty: float,
    duration_penalty: List[float],
    exact_step: int,
    seg_limit=None,
    intra_weight: float = 1.0,
    **kwargs
):
    """
    Determine the configuration of splits with the highest score, given that
    splitting has a cost of `penalty`. `seg_limit` is a limitation on the length
    of a segment that saves memory and computation, but gives poor results
    should there be no split withing the range.
    The algorithm is built upon the idea that there is a accumulated score
    matrix containing the maximal score of creating a segment (i, j), containing
    all words [w_i, ..., w_j] at position i, j. The matrix `acc` is indexed to
    contain the first `seg_limit` elements of each row of the score matrix.
    `colmax` contains the column maxima of the score matrix.
    `ptr` is a backtracking pointer to determine the splits made while
    forward accumulating the highest score in the score matrix.

    DP formula
    * Score function σ(i, j) → a score for segment (i, j)
    * Total score S(n, k) → accumulative score for first n element with k segment
    * recursive
        * S(n, 0) = 0, σ(0, 0) = 0
        * S(n, 1) = σ(0, n)
        * S(n, k) = \max_{k-1}^{n} [ S(l, k-1) + σ(l, n) ]
    """
    logging.debug('start split exact')

    L, dim = docmat.shape
    lim = L if seg_limit is None else seg_limit
    intra_weight = max(min(intra_weight, 0), 1.0)
    total = 0
    gains = []
    optimal = True
    assert lim > 0
    assert penalty > 0
    
    if exact_step == 1:
        splits = [0]
    elif exact_step == L:
        splits = [i for i in range(L)]
    else:

        C = np.full((L, exact_step + 1), 0, dtype=int)
        S = np.full((L, exact_step + 1), 0, dtype=np.float32)

        ctxvecs = np.cumsum(docmat, axis=0)
        score = norm(ctxvecs, axis=1, ord=2)
        S[:, 1] = score

        for k in range(2, exact_step + 1):
            for n in range(k, L):
                ctxvecs = np.cumsum(docmat[:n][::-1], axis=0)
                score = norm(ctxvecs, axis=1, ord=2)[::-1]
                for l in range(n, max(k-1, n - lim), -1):
                    val = S[l, k - 1] + score[l - 1]
                    if val > S[n, k]:
                        S[n, k] = val
                        C[n, k] = l - 1

        # backtrack
        splits = []
        gains = []
        idx = L - 1
        for i in range(exact_step):
            gains.append(S[idx, exact_step - i] - S[C[idx, exact_step - i], exact_step - i - 1])
            splits.append(C[idx, exact_step - i])
            idx = C[idx, exact_step - i]

        splits = splits[::-1]
        gains = gains[::-1]
        total = S[L - 1, exact_step]

    logging.info(f"splits: {splits}")
    return Segmentation(total, splits, gains, min_gain=None, optimal=optimal, duration_penalty=duration_penalty, penalty=penalty)


def get_total(docmat, splits, penalty):
    """
    Compute the total score of a split configuration with given penalty.
    """
    L, dim = docmat.shape
    cuts = [0] + list(splits) + [L]
    cumvecs = np.cumsum(np.vstack((np.zeros((1, dim)), docmat)), axis=0)
    return sum(
        norm(cumvecs[left, :] - cumvecs[right, :], ord=2)
        for left, right in zip(cuts[:-1], cuts[1:])) - len(splits) * penalty


def get_gains(docmat, splits, width=None):
    """
    Calculate gains of the splits towards the left and right neighbouring
    split.
    If `width` is given, calculate gains of the splits towards a centered window
    of length 2 * `width`.
    """
    gains = []
    L = docmat.shape[0]
    for beg, cen, end in zip([0] + splits[:-1], splits, splits[1:] + [L]):
        if width is not None and width > 0:
            beg, end = max(cen - width, 0), min(cen + width, L)

        slice_l, slice_r, slice_t = [
            slice(beg, cen),  # left context
            slice(cen, end),  # right context
            slice(beg, end)
        ]  # total context

        left = norm(docmat[slice_l, :].sum(axis=0), ord=2)
        right = norm(docmat[slice_r, :].sum(axis=0), ord=2)
        total = norm(docmat[slice_t, :].sum(axis=0), ord=2)
        gains.append(left + right - total)
    return gains
