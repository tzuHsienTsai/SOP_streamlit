import os
import json
import logging
import asyncio
import numpy as np
from typing import List
from cache import lru_cache_on_first_argument
from gsutil import upload_blob, download_blob, get_blob_name
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from CloudRunService import CloudRunService
from CloudRunService import UniversalSentenceEncoder
from CloudRunService import TemporalEmbeddingsGen
from CloudRunService import EntitiesandTags
from CloudRunService import MT5SentenceEncoder
from CloudRunService import VitEmbeddingsGen
from utils import mt5_max_len, vit_max_len, get_or_create_eventloop, timeit


u_s_e_url = os.getenv('u_s_e_address', 'https://universal-sentence-encoder-3oqy6iorxq-uc.a.run.app')
temporal_embeddings_url = os.getenv('temporal_embeddings_gen_address', 'https://temporal-embeddings-gen-3oqy6iorxq-uc.a.run.app')
mt5_url = os.getenv('mt5_address', 'https://mt5-sentence-encoder-3oqy6iorxq-uc.a.run.app')
vit_url = os.getenv('vit_address', 'https://vit-embeddings-gen-3oqy6iorxq-uc.a.run.app')
mt5_gke_url = os.getenv('mt5_gke_address', 'http://34.149.230.147/mt5')
vit_gke_url = os.getenv('vit_gke_address', 'http://34.149.230.147/vit')
entities_and_tags_url = os.getenv('entities_and_tags_gen_address', 'https://entities-and-tags-3oqy6iorxq-uc.a.run.app')

universal_sentence_encoder = UniversalSentenceEncoder(url=u_s_e_url)
temporal_embeddings_gen = TemporalEmbeddingsGen(url=temporal_embeddings_url)
mt5_sentence_encoder = MT5SentenceEncoder(url=mt5_url)
vit_embeddings_gen = VitEmbeddingsGen(url=vit_url)
gke_mt5_sentence_encoder = MT5SentenceEncoder(url=mt5_gke_url)
gke_vit_embeddings_gen = VitEmbeddingsGen(url=vit_gke_url)
entities_and_tags = EntitiesandTags(url=entities_and_tags_url)


def _get_embeddings(result):
    res = json.loads(result)
    key1 = 'sentences_embeddings'
    key2 = 'u_s_e_embeddings'
    return normalize(res[key1]) if key1 in res else normalize(res[key2])


@lru_cache_on_first_argument()
def get_embedding_cache(blob_name: str):
    try:
        logging.info("Get embedding")
        obj = download_blob(blob_name)
        logging.info("Embedding in cache.")
        return obj["sentenced_embeddings"], obj["image_embeddings"], obj["final_embeddings"]
    except:
        return None


def store_embedding_cache(
    blob_name: str,
    request_data: dict,
    sentenced_embeddings,
    image_embeddings,
    final_embeddings
):
    # cache only when candidate is required
    if request_data.get('params', {}).get('genCandidates', False):
        logging.info("Cache embeddings")
        obj = {
            'sentenced_embeddings': sentenced_embeddings,
            'image_embeddings': image_embeddings,
            'final_embeddings': final_embeddings,
        }
        upload_blob(obj, blob_name)


#@lru_cache_on_first_argument()
@timeit
def get_embedding(workflowId: str, request_data: dict, sentenced_text: List[str]):
    device = 'GPU' if os.getenv('GPU') else 'CPU'
    logging.info(f"Get embedding ({device})")
    blob_name = get_blob_name(workflowId)
    cache = None

    if request_data.get('cache', True):
        cache = get_embedding_cache(blob_name)

    if cache is not None:
        return cache
    else:
        logging.info("Start inference")

        # request data to mt5
        se_split_data = mt5_sentence_encoder.make_data(
            postfix='_sentences',
            request_data=request_data,
            input_data=sentenced_text,
            max_len=mt5_max_len,
        )

        # request data to vit
        input_image_list = [{
            'image': data['image']
        } for data in request_data['input']]
        te_split_data = vit_embeddings_gen.make_data(
            postfix='',
            request_data=request_data,
            input_data=input_image_list,
            max_len=vit_max_len,
        )

        # concurrent process
        se_results, te_results = _parallel_requests(se_split_data, te_split_data)

        # recieve mt5
        sentenced_embeddings = np.vstack([
            normalize(res.json()['sentences_embeddings']) for res in se_results
        ]).astype('float32')

        # recieve vit
        vit_embeddings = np.vstack([
            res.json()['embeddings_result'] for res in te_results
        ]).astype('float32')
        # reduce embeddings
        image_embeddings = reduce_embedding_dim(request_data, vit_embeddings)

        if len(request_data['input']) > 1:
            final_embeddings = np.hstack([
                sentenced_embeddings,
                normalize(image_embeddings),
            ])
            final_embeddings = normalize(final_embeddings)
        else:
            final_embeddings = normalize(sentenced_embeddings)

        store_embedding_cache(
            blob_name,
            request_data,
            sentenced_embeddings,
            image_embeddings,
            final_embeddings,
        )
        return sentenced_embeddings, image_embeddings, final_embeddings


@timeit
def reduce_embedding_dim(request_data: dict, embeddings):
    logging.info("Dimension reduction")
    reduction_time_threshold = 40 * 60  # 40 min * 60 sec
    try:
        video_time = request_data['input'][-1]['startTime'] - request_data['input'][0]['endTime']
    except Exception:
        video_time = reduction_time_threshold

    if video_time <= reduction_time_threshold:
        tsne = TSNE(n_components=100, method='exact', random_state=0)
        reduce_embeddings = tsne.fit_transform(embeddings)
    else:
        pca = PCA(n_components=0.025, random_state=0)
        reduce_embeddings = pca.fit_transform(embeddings)

    return reduce_embeddings


async def _entrypoint(data1, data2):

    N1, N2 = len(data1), len(data2)
    logging.debug(
        "Sending parallel POST requests to sentence encoder and image embeddings ..."
    )

    if os.getenv('GPU') == "True":
        mt5_task = [
            CloudRunService.internal_post(
                gke_mt5_sentence_encoder.url,
                data,
                i,
                N1,
                gke_mt5_sentence_encoder.retry_limit,
            ) for i, data in enumerate(data1)
        ]
        vit_task = [
            CloudRunService.internal_post(
                gke_vit_embeddings_gen.url,
                data,
                i,
                N2,
                gke_vit_embeddings_gen.retry_limit,
            ) for i, data in enumerate(data2)
        ]
        tasks = mt5_task + vit_task
    else:
        tasks = [
            CloudRunService.internal_post(
                mt5_sentence_encoder.url,
                data,
                i,
                N1,
                mt5_sentence_encoder.retry_limit,
            ) for i, data in enumerate(data1)
        ] + [
            CloudRunService.internal_post(
                vit_embeddings_gen.url,
                data,
                i,
                N2,
                vit_embeddings_gen.retry_limit,
            ) for i, data in enumerate(data2)
        ]

    results = await asyncio.gather(*(tasks))

    # task1 = CloudRunService.batch_internal_post(
    #     mt5_sentence_encoder.url,
    #     data1,
    #     mt5_sentence_encoder.retry_limit,
    # )
    # task2 = CloudRunService.batch_internal_post(
    #     vit_embeddings_gen.url,
    #     data2,
    #     vit_embeddings_gen.retry_limit,
    # )
    # results = await asyncio.gather(task1, task2)

    return results


def _parallel_requests(data1, data2):
    loop = get_or_create_eventloop()
    results = loop.run_until_complete(_entrypoint(data1, data2))
    CloudRunService.check_result('Parallel Requests', results)
    return results[:len(data1)], results[len(data1):]
