import numpy as np
import random
from algorithm import split_greedy


def get_segments(text_particles, segmentation):
    """
    Reorganize text particles by aggregating them to arrays described by the
    provided `segmentation`.
    """
    segmented_text = []
    L = len(text_particles)
    for beg, end in zip([0] + segmentation.splits, segmentation.splits + [L]):
        segmented_text.append(text_particles[beg:end])
    return segmented_text


def get_duration_penalty(request_data):
    start_time = [data["startTime"] for data in request_data['input']]
    end_time = [data["endTime"] for data in request_data['input']]
    duration = [0.0]
    for s, e in zip(start_time[1:], end_time[:-1]):
        duration.append((s - e) / 20)  # 20 seconds -> penalty 1.0
    duration = np.array(duration) - np.mean(duration)
    return duration.tolist()


def get_penalty(docmats, segment_len):
    """
    Determine penalty for segments having length `segment_len` on average.
    This is achieved by stochastically rounding the expected number
    of splits per document `max_splits` and taking the minimal split_gain that
    occurs in split_greedy given `max_splits`.
    """
    penalties = []
    for docmat in docmats:
        avg_n_seg = docmat.shape[0] / segment_len
        random.seed(87)
        max_splits = int(avg_n_seg) + (random.random() < avg_n_seg % 1) - 1
        if max_splits >= 1:
            seg = split_greedy(docmat, max_splits=max_splits)
            if seg.min_gain < np.inf:
                penalties.append(seg.min_gain)
    if len(penalties) > 0:
        return np.mean(penalties)
    else:
        return np.finfo(np.float32).max
    raise ValueError('All documents too short for given segment_len.')


def P_k(splits_ref, splits_hyp, N):
    """
    Metric to evaluate reference splits against hypothesised splits.
    Lower is better.
    `N` is the text length.
    """
    k = round(N / (len(splits_ref) + 1) / 2 - 1)
    ref = np.array(splits_ref, dtype=np.int32)
    hyp = np.array(splits_hyp, dtype=np.int32)

    def is_split_between(splits, left, right):
        return np.sometrue(np.logical_and(splits - left > 0, splits - right < 0))

    acc = 0
    for i in range(N - k):
        acc += is_split_between(ref, i, i + k) != is_split_between(
            hyp, i, i + k)

    return acc / (N - k)
