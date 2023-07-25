import os
import tools
import faiss
import requests
import logging
import numpy as np
import algorithm as algo
from typing import List
from copy import deepcopy
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import google_translate_api, supported_lang, match_step_english_number_string, timeit
from embedding import get_embedding
from summarization import summarize


@timeit
def preprocess(request_data: dict, p_lang: str):
    logging.info(f"{request_data['lang']} not supported, translated to {p_lang}")
    logging.info(f"Translating {len(request_data['input'])} sentences")

    def trans(data: dict):
        s = data['sentence']
        data['ori_sentence'] = s
        data['sentence'] = google_translate_api(s, p_lang)
        return data
    with ThreadPoolExecutor(max_workers=16) as executor:
        request_data['input'] = list(executor.map(trans, request_data['input']))

    request_data['lang'] = p_lang


@timeit
def postprocess(request_data: dict, result: dict, lang: str):
    logging.info(f'Translate english result to {lang}')
    logging.info(f"segmented_text (en):\n{result['segmented_text']}")
    logging.info(f"summaries (en):\n{result['summaries']}")
    logging.debug(f"video tagging (en):\n{result['video_tagging']}")

    # segmentation result
    segmented_text = result['segmented_text']
    cnt = 0
    for i, segment in enumerate(segmented_text):
        for j in range(len(segment)):
            ori_s = request_data['input'][cnt]['ori_sentence']
            result['segmented_text'][i][j] = ori_s
            cnt += 1

    # summaries
    for i, s in enumerate(result['summaries']):
        result['summaries'][i] = google_translate_api(s, lang)

    # video tagging
    for i, s in enumerate(result['video_tagging']):
        result['video_tagging'][i] = google_translate_api(s, lang)

    return result


@timeit
def segmentation_entry(request_data: dict):
    result = {}
    lang = request_data['lang']
    if lang not in supported_lang:
        preprocess(request_data, 'en')

    if request_data.get('isSlides', False):
        result = slide_segmentation(request_data)
    else:
        result = segmentation(request_data)
        result['message'] = 'success'

    if lang not in supported_lang:
        result = postprocess(request_data, result, lang)

    return result


def get_poster_image(image_embeddings):
    # cover poster image recommendation
    if image_embeddings is None:
        return 0
    else:
        index = faiss.IndexFlatL2(image_embeddings.shape[-1])
        index.add(image_embeddings)
        distance, index = index.search(np.array([np.mean(image_embeddings, axis=0)]), 1)
        return int(index[0][0])


@timeit
def segmentation(request_data: dict):
    sentenced_text = [data['sentence'] for data in request_data['input']]
    if _run_segmentation(request_data):
        sentenced_embeddings, image_embeddings, final_embeddings = get_embedding(
            request_data['workflowId'],
            request_data,
            sentenced_text
        )
    else:
        sentenced_embeddings, image_embeddings, final_embeddings = None, None, None

    poster_image_index = get_poster_image(image_embeddings)
    segmentation_results = get_segment_results(
        request_data,
        final_embeddings,
        sentenced_text,
        sentenced_embeddings,
        poster_image_index,
        verbose=True
    )

    # get candidates
    if request_data.get('params', {}).get('genCandidates', False):
        if _run_segmentation(request_data):
            refined_segmented_text = segmentation_results['segmented_text']
            adjust_rate = segmentation_results['adjust_rate']
            candidates = {
                len(refined_segmented_text): deepcopy(segmentation_results),
            }
            logging.info(f"Generate Candidates. Auto {len(refined_segmented_text)} steps.")
            request_data['params'] = request_data.get('params', {})
            request_data['params']['exact_step'] = len(refined_segmented_text)
            precompute = request_data.get('params', {}).get('precompute', 3)
            candidates = find_less_more(
                candidates,
                request_data,
                final_embeddings,
                sentenced_text,
                sentenced_embeddings,
                poster_image_index,
                adjust_rate,
                precompute=precompute
            )
            segmentation_results['candidates'] = candidates

    return segmentation_results


def get_segment_results(
    request_data: dict,
    final_embeddings: list,
    sentenced_text: List[str],
    sentenced_embeddings: list,
    poster_image_index: int,
    adjust_rate: float = 1.5,
    verbose: bool = False
):
    exact_step = request_data.get('params', {}).get('exact_step')
    logging.debug(f"Segment with exact={exact_step}")
    if _run_segmentation(request_data):
        refined_segmented_text, segmentation, adjust_rate = _segmentation(
            request_data=request_data,
            final_embeddings=final_embeddings,
            sentenced_text=sentenced_text,
            adjust_rate=adjust_rate,
            verbose=verbose
        )
    else:
        refined_segmented_text = [sentenced_text]

    logging.debug("Summarize")
    summaries = summarize(
        request_data,
        [refined_segmented_text, sentenced_text, sentenced_embeddings]
    )

    segmentation_results = {
        'segmented_text': refined_segmented_text,
        'poster_image_index': poster_image_index,
        'summaries': summaries,
        'video_tagging': [],
        'salient_sentence': [i[0] for i in refined_segmented_text if i],
        'workflowId': request_data['workflowId'],
        'adjust_rate': adjust_rate
    }

    return segmentation_results


def find_less_more(
    candidates: dict,
    request_data: dict,
    final_embeddings: list,
    sentenced_text: List[str],
    sentenced_embeddings: list,
    poster_image_index: int,
    adjust_rate: float,
    precompute: int,
):
    logging.info(f"Find less and more steps for {precompute} precomputed steps")

    def find_less(req_data: dict, less_adjust_rate: float, steps: int = 100):
        exact = req_data['params'].get('exact_step')
        logging.info(f"Find less with exact={exact}.")
        for _ in range(steps):
            segmentation_results = get_segment_results(
                req_data,
                final_embeddings,
                sentenced_text,
                sentenced_embeddings,
                poster_image_index,
                less_adjust_rate,
            )
            refined_segmented_text = segmentation_results['segmented_text']
            less_adjust_rate = segmentation_results['adjust_rate']
            if len(refined_segmented_text) in candidates:
                # if exact step is not specified, search less result by lower the penalty
                less_adjust_rate *= 1.01
                logging.info(f"less_adjust_rate: {less_adjust_rate}")
            else:
                candidates[len(refined_segmented_text)] = segmentation_results
                logging.info(f"Generate Candidates with exact={exact}. Less {len(refined_segmented_text)} steps.")
                return

    def find_more(req_data: dict, more_adjust_rate: float, steps: int = 100):
        exact = req_data['params'].get('exact_step')
        logging.info(f"Find More with exact={exact}.")
        for _ in range(steps):
            segmentation_results = get_segment_results(
                req_data,
                final_embeddings,
                sentenced_text,
                sentenced_embeddings,
                poster_image_index,
                more_adjust_rate,
            )
            refined_segmented_text = segmentation_results['segmented_text']
            more_adjust_rate = segmentation_results['adjust_rate']
            if len(refined_segmented_text) in candidates:
                # if exact step is not specified, search more result by increase the penalty
                less_adjust_rate *= 0.99
                logging.info(f"more_adjust_rate: {more_adjust_rate}")
            else:
                candidates[len(refined_segmented_text)] = segmentation_results
                logging.info(f"Generate Candidates with exact={exact}. More {len(refined_segmented_text)} steps.")
                return

    futures = []
    submitted = set()
    auto_step = request_data['params']['exact_step']

    with ThreadPoolExecutor() as executor:

        for dial in range(precompute):
            exact_step = max(1, auto_step - dial - 1)
            if exact_step in submitted:
                continue
            else:
                submitted.add(exact_step)

            req_data = deepcopy(request_data)
            req_data['params']['exact_step'] = exact_step
            task = executor.submit(find_less, req_data, adjust_rate)
            futures.append(task)

        for dial in range(precompute):
            exact_step = min(len(sentenced_text), auto_step + dial + 1)
            if exact_step in submitted:
                continue
            else:
                submitted.add(exact_step)

            req_data = deepcopy(request_data)
            req_data['params']['exact_step'] = exact_step
            task = executor.submit(find_more, req_data, adjust_rate)
            futures.append(task)

        for future in as_completed(futures):
            continue

    return candidates


def _run_segmentation(request_data: dict):
    return 'segment' not in request_data or request_data['segment'] is True


def _segmentation(
    request_data: dict,
    final_embeddings: list,
    sentenced_text: List[str],
    adjust_rate: float = 1.0,
    verbose: bool = False
):
    # larger adjust rate leads to less segments
    # smaller adjust rate leads to more segments

    n_segments = request_data.get('params', {}).get('exact_step', 10)
    max_segment_number = request_data.get('max_segment_number', len(final_embeddings))
    min_segment_number = request_data.get('min_segment_number', 1)

    trials = 20
    for _ in range(trials):  # trial on conflict with hard limits
        segment_len = min(max(len(final_embeddings) / n_segments, min_segment_number), max_segment_number)
        penalty = tools.get_penalty([final_embeddings], segment_len) * adjust_rate
        logging.debug(f"segment_len: {segment_len}, penalty: {penalty}")
        duration_penalty = tools.get_duration_penalty(request_data)

        if request_data.get('params', {}).get('exact_step'):
            segmentation = algo.split_exact(
                final_embeddings,
                penalty,
                duration_penalty,
                seg_limit=20,
                **request_data.get('params', {})
            )
        else:
            segmentation = algo.split_optimal(
                final_embeddings,
                penalty,
                duration_penalty,
                seg_limit=20,
                **request_data.get('params', {})
            )
            adjust = check_hard_limits(segmentation, request_data, max_segment_number)
            if adjust:
                adjust_rate *= adjust

        segmented_text = tools.get_segments(sentenced_text, segmentation)
        break

    if verbose:
        logging.info(f'length of segmented text: {len(segmented_text)-1}')
    postprocess_segment(segmented_text)
    return segmented_text[1:], segmentation, adjust_rate


def postprocess_segment(segmented_text: List[List[str]]):
    # detect [[step one, step two], [first step, second step]]
    output = []
    for i in segmented_text:
        tmp = []
        for j in i:
            if match_step_english_number_string(j.lower()) and tmp:
                output.append(tmp)
                tmp = []
            tmp.append(j)
        output.append(tmp)
    return output


def check_hard_limits(segmentation, request_data: dict, max_segment_number: int):

    # segment length > 50
    splits = segmentation.splits
    seg_len = [i - j for i, j in zip(splits[1:], splits[:-1])]
    if any([i > 50 for i in seg_len]):
        logging.debug('single segment exceed maximum length 50')
        return 0.95

    # segment duration > 10 minutes
    start_time = [i['startTime'] for i in request_data['input']]
    seg_duration = [i - j for i, j in zip(start_time[1:], start_time[:-1])]
    if any([i > 20 * 60 for i in seg_duration]):
        logging.debug('single segment exceed maximum 10 minutes')
        return 0.95

    if len(segmentation.splits) > max_segment_number:
        logging.debug(f'segment number {len(segmentation.splits)} exceed maximum length {max_segment_number}')
        return 1.05

    return 0


def slide_segmentation(json_request: dict):
    slides_url = os.getenv('slides_address')
    data = {
        'token': json_request.get('token'),
        'lang': json_request.get('lang'),
        'workflowId': json_request.get('workflowId'),
        'input': json_request.get('input')
    }
    if 'pdfUrl' in json_request:
        data['pdf'] = json_request.get('pdfUrl')

    logging.info(f"slide segmentation request made to {slides_url} with data {data}")
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.2,
        status_forcelist=[500, 502, 503, 504]
    )

    session.mount('https://', HTTPAdapter(max_retries=retries))
    slides_request = session.post(slides_url, json=data, timeout=300)
    response = slides_request.json()
    return response
