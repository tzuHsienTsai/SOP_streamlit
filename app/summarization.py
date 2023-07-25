import os
import logging
import asyncio
from typing import List
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from aiohttp_retry import RetryClient, ExponentialRetry
from utils import google_translate_api, get_or_create_eventloop, timeit, flan_max_len, token
from old_summarization import stage1, stage2, stage3, stage4, get_segmentation_ret
from CloudRunService import CloudRunService, FLANSentenceGenerator


HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "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")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
flan_t5_url = os.getenv('flan_t5_address', 'https://flan-sentence-generator-3oqy6iorxq-uc.a.run.app')
flan_t5_sentence_generator = FLANSentenceGenerator(url=flan_t5_url)
flan_t5_gke_url = os.getenv('flan_t5_gke_address', 'http://34.149.230.147/flan-t5')
gke_flan_t5_sentence_generator = FLANSentenceGenerator(url=flan_t5_gke_url)


@timeit
def summarize(request_data: dict, fallback_data: list):
    logging.debug('Get summarization')
    refined_segmented_text = fallback_data[0]

    if request_data.get('params', {}).get('summarize') != "flan":
        try:
            return openai_summarize(request_data, refined_segmented_text)
        except Exception as e:
            logging.error(f"get openai summary error:{e}. Fallback to flan summary")
            # logging.exception("")

    try:
        return flan_summarize(request_data, refined_segmented_text)
    except Exception as e:
        logging.error(f"get flan summary error:{e}. Fallback to old summary")
        # logging.exception("")

    try:
        logging.info("Summarization fallback to old salient sentence summary")
        return fallback_summarize(request_data, fallback_data)
    except Exception as e:
        logging.error(f"fallback summary error:{e}")
        # logging.exception("")

    # return empty string if above both failed
    return ['' for i in range(len(refined_segmented_text))]


def openai_summarize(request_data: dict, refined_segmented_text: List[str]):
    max_len = 10000
    input_text = [' '.join(text_list)[:max_len] for text_list in refined_segmented_text]
    summary = get_summary(input_text)
    if request_data["lang"] != "en":
        with ThreadPoolExecutor() as executor:
            args = [(s, request_data["lang"]) for s in summary]
            summary = list(executor.map(lambda p: google_translate_api(*p), args))
    return summary


def flan_summarize(request_data: dict, refined_segmented_text: List[str]):
    max_len = 10000
    input_text = [' '.join(text_list)[:max_len] for text_list in refined_segmented_text]
    summary = batch_get_summary(request_data, input_text)
    if request_data["lang"] != "en":
        with ThreadPoolExecutor() as executor:
            args = [(s, request_data["lang"]) for s in summary]
            summary = list(executor.map(lambda p: google_translate_api(*p), args))
    return summary


def fallback_summarize(request_data: dict, fallback_data: list):
    staged_results = []
    staged_results.append(stage1(request_data, *fallback_data))
    staged_results.append(stage2(request_data, staged_results))
    staged_results.append(stage3(request_data, staged_results))
    staged_results.append(stage4(request_data, staged_results))
    return get_segmentation_ret(request_data, staged_results)['summaries']


async def fetch_openai(session, text: str) -> str:
    headers = {
        "Authorization": f'Bearer {OPENAI_API_KEY}',
        'Content-Type': 'application/json'
    }
    prompt = f"{text}\n\nSummarize the title of the above paragraph in maximum three words:"
    for _ in range(3):
        async with session.post(
            'https://api.openai.com/v1/completions',
            json={
                "model": "text-davinci-002",
                "prompt": prompt,
                "temperature": 0,
                "max_tokens": 10,
                "top_p": 1,
                "best_of": 5,
                "frequency_penalty": 2,
                "presence_penalty": 0
            },
            headers=headers,
            timeout=10
        ) as response:
            response_json = await response.json()
            try:
                summary = response_json['choices'][0]['text'].strip()
                return summary
            except Exception:
                logging.error(f"{response_json}")
    raise Exception("fetch openai failed")


async def fetch_flan(session, text: str) -> str:
    headers = {
        'Content-Type': 'application/json'
    }
    prompt = f"{text}\n\nSummarize the title of the above paragraph in maximum three words:"
    for _ in range(3):
        async with session.post(
            flan_t5_sentence_generator.url,
            # gke_flan_t5_sentence_generator.url,
            json={
                'token': '3lPDYZWupFO9tCUU2c5VUTiY4r6ciOvL',
                'input': [text + prompt],
            },
            headers=headers,
        ) as response:
            response_json = await response.json()
            try:
                summary = response_json['sentences'][0].strip()
                return summary
            except Exception:
                logging.error(f"{response_json}")
    raise Exception("fetch flan failed")


async def fetch_huggingface(session, text: str, model: str = "bigscience/bloom") -> str:
    headers = {
        "Authorization": f'Bearer {HUGGINGFACE_API_KEY}',
        'Content-Type': 'application/json'
    }
    prompt = f"{text} \n To sum up, the best topic for the above article is: "
    for _ in range(3):
        async with session.post(
            f"https://api-inference.huggingface.co/models/{model}",
            json={
                'inputs': prompt,
                'temperature': 0,
                'repetition_penalty': 100,
                'diversity_penalty': 0.9,
                'top_k': 1,
                'wait_for_model': True,
                'num_return_sequences': 1,
                'return_full_text': False,
            },
            headers=headers,
        ) as response:
            response_json = await response.json()
            try:
                summary = response_json[0]['generated_text'][len(prompt):].strip()
                return summary
            except Exception:
                logging.error(f"{response_json}")
    raise Exception("fetch huggingface failed")


def get_summary(input_text_list: List[List[str]]) -> list:

    async def fetch(input_text_list: List[List[str]], fetch_fn):
        retry_options = ExponentialRetry(
            attempts=5,
            factor=3.0
        )
        retry_client = RetryClient(
            raise_for_status=False,
            retry_options=retry_options,
        )
        async with retry_client as session:
            tasks = [fetch_fn(session, text) for text in input_text_list]
            results = await asyncio.gather(*tasks)
            return results

    loop = get_or_create_eventloop()
    results = loop.run_until_complete(fetch(input_text_list, fetch_fn=fetch_openai))
    results = postprocess_summary(results)
    return results


def batch_get_summary(request_data: dict, input_text_list: List[List[str]]) -> list:

    if False:  # os.getenv('GPU') == "True":
        batch_data = gke_flan_t5_sentence_generator.make_data(
            request_data=request_data,
            input_data=input_text_list,
            max_len=flan_max_len,
        )
        tasks = [
            CloudRunService.internal_post(
                gke_flan_t5_sentence_generator.url,
                data,
                i,
                len(batch_data),
                gke_flan_t5_sentence_generator.retry_limit,
            ) for i, data in enumerate(batch_data)
        ]
    else:
        batch_data = gke_flan_t5_sentence_generator.make_data(
            request_data=request_data,
            input_data=input_text_list,
            max_len=flan_max_len // 4,
        )
        tasks = [
            CloudRunService.internal_post(
                flan_t5_sentence_generator.url,
                data,
                i,
                len(batch_data),
                flan_t5_sentence_generator.retry_limit,
            ) for i, data in enumerate(batch_data)
        ]

    # device = 'GPU' if os.getenv('GPU') == 'True' else 'CPU'
    device = 'CPU'
    logging.info(f"get flan summary ({device})")
    loop = get_or_create_eventloop()
    results = loop.run_until_complete(asyncio.gather(*(tasks)))
    results = [j for i in results for j in i.json()['sentences']]
    results = postprocess_summary(results)
    return results


def postprocess_summary(results: List[str]):
    counter = Counter()
    for e, i in enumerate(results):
        if i in counter:
            cnt = counter[i]
            results[e] = f"{i} {cnt+1}"
        counter[i] += 1
    return results
