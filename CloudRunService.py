import time
import old_utils
import logging
import asyncio
import requests
import traceback
from functools import partial
from aiohttp_retry import RetryClient, ExponentialRetry


class CloudRunService:

    total_response_size = 0

    def __init__(self, service_name, url, retry_limit=5):
        self.service_name = service_name
        self.url = url
        self.retry_limit = retry_limit
        self.batch_request = 8
        # self.result = None

    @staticmethod
    async def batch_internal_post(url, data_list, retry_limit):

        async def fetch(client, data: dict):
            headers = {'Content-Type': 'application/json'}
            async with client.post(
                url,
                json=data,
                headers=headers,
            ) as response:
                res = await response.text()
                if response.status == 200:
                    return res
                else:
                    err_msg = f"POST request to {url} failed with status {response.status}: {res}"
                    logging.error(err_msg)
                    raise Exception(err_msg)

        retry_options = ExponentialRetry(attempts=retry_limit)
        retry_client = RetryClient(
            raise_for_status=False,
            retry_options=retry_options
        )
        async with retry_client as client:
            tasks = [fetch(client, data) for data in data_list]
            results = await asyncio.gather(*tasks)
            return results

    @staticmethod
    async def internal_post(url, data, i, N, retry_limit):
        st = time.time()
        async with asyncio.Semaphore(8):
            second_interval = 10
            loop = old_utils.get_or_create_eventloop()

            idx = f'{i + 1}/{N}'
            logging.debug(f"Sending POST request No.{idx} to '{url}'...")
            for retry_count in range(retry_limit):
                try:
                    ts = time.time()
                    res = await loop.run_in_executor(None, partial(requests.post, url=url, json=data))
                    t = time.time() - ts
                    logging.debug(f"POST request {idx} to {url} {retry_count + 1} time after {t:.2f} secs")
                    if res.status_code == 200:
                        logging.debug(f"POST request {idx} to {url} succeeded after {retry_count + 1} attempt(s)")
                        return res
                    else:
                        logging.debug(f"POST request {idx} to {url} "
                                      f"{retry_count + 1} time failed, "
                                      f"status code: {res.status_code} "
                                      f"retry after {second_interval} seconds")
                except:
                    pass
                    # logging.error(traceback.format_exc())
                finally:
                    await asyncio.sleep(second_interval)

            logging.error(f"POST request {idx} to {url} failed after {retry_limit} attempts")
            logging.error(f'error response: {res.text}')
            raise Exception(f"POST request to {url} failed ")


    async def fetch(self, request_data):
        N = len(request_data)
        # logging.debug(f"Sending POST requests to {self.service_name}...")
        ts = time.time()
        try:
            tasks = [
                self.internal_post(self.url, data, i, N, self.retry_limit)
                for i, data in enumerate(request_data)
            ]
            results = await asyncio.gather(*tasks)
            t = time.time() - ts
            logging.debug(f"{self.service_name} returned after {t:.2f} secs")
        except Exception as e:
            t = time.time() - ts
            logging.error(f"Parallel requests to {self.service_name} "
                          f"error after {t:.2f} secs: {str(e)}")
            raise Exception(f"Parallel requests to {self.service_name} "
                            f"error after {t:.2f} secs: {str(e)}")
        return results
    
    async def resolve_result(self, tasks: list):
        results = await asyncio.gather(*tasks)
        results = [j for i in results for j in i]
        return results

    def get_result(self, data, batch=None):
        batch = self.batch_request if batch is None else batch
        loop = utils.get_or_create_eventloop()
        try:
            tasks = []
            results = []
            N = len(data)
            logging.debug(f"Sending {N} POST requests to {self.service_name}...")
            for start in range(0, N, batch):
                end = min(start + batch, N)
                # results += loop.run_until_complete(self.fetch(data[start: end]))
                tasks.append(loop.create_task(self.fetch(data[start: end])))
            results = loop.run_until_complete(self.resolve_result(tasks))
            self.check_result(self.service_name, results)
            logging.debug(f"Service {self.service_name} response success")
            return results
        except Exception as e:
            logging.error(f"Service {self.service_name} response error: {str(e)}")
            raise Exception(f'Service {self.service_name} response error: {str(e)}')

    @staticmethod
    def check_result(service_name, results):
        for res in results:
            CloudRunService.total_response_size += len(res.text)
            if res.status_code != 200:
                raise Exception(f'Service {service_name} error: {res.text}')


class UniversalSentenceEncoder(CloudRunService):

    def __init__(self, url):
        # Assumed url have been checked
        service_name = 'Universal_Sentence_Encoder'
        super(UniversalSentenceEncoder, self).__init__(service_name, url)

    def make_data(self, postfix, request_data, input_data, max_len):
        split_data = []
        for begin in range(0, len(input_data), max_len):
            end = min(begin + max_len, len(input_data))
            split_data.append({
                "requestId": request_data['workflowId'] + postfix,
                "token": request_data['token'],
                "input": input_data[begin:end],
            })
        return split_data


class MT5SentenceEncoder(CloudRunService):

    def __init__(self, url):
        # Assumed url have been checked
        service_name = 'MT5_Sentence_Encoder'
        super(MT5SentenceEncoder, self).__init__(service_name, url)

    def make_data(self, postfix, request_data, input_data, max_len):
        split_data = []
        for begin in range(0, len(input_data), max_len):
            end = min(begin + max_len, len(input_data))
            split_data.append({
                "requestId": request_data['workflowId'] + postfix,
                "token": request_data['token'],
                "input": input_data[begin:end],
            })
        return split_data


class FLANSentenceGenerator(CloudRunService):

    def __init__(self, url):
        # Assumed url have been checked
        service_name = 'FLAN_Sentence_Generator'
        super(FLANSentenceGenerator, self).__init__(service_name, url)

    def make_data(self, request_data: dict, input_data: list, max_len: int):
        split_data = []
        for begin in range(0, len(input_data), max_len):
            end = min(begin + max_len, len(input_data))
            split_data.append({
                "token": request_data['token'],
                "input": input_data[begin: end],
            })
        return split_data


class TemporalEmbeddingsGen(CloudRunService):

    def __init__(self, url):
        service_name = 'Temporal_Embeddings_Gen'
        super(TemporalEmbeddingsGen, self).__init__(service_name, url)

    def make_data(self, postfix, request_data, input_data, max_len):
        split_data = []
        for begin in range(0, len(input_data), max_len):
            end = min(begin + max_len, len(input_data))
            split_data.append({
                "workflowId": request_data['workflowId'] + postfix,
                "token": request_data['token'],
                "lang": request_data['lang'],
                "input": input_data[begin:end],
            })
        return split_data


class VitEmbeddingsGen(CloudRunService):

    def __init__(self, url):
        service_name = 'Vit_Embeddings_Gen'
        super(VitEmbeddingsGen, self).__init__(service_name, url)

    def make_data(self, postfix, request_data, input_data, max_len):
        split_data = []
        for begin in range(0, len(input_data), max_len):
            end = min(begin + max_len, len(input_data))
            split_data.append({
                "workflowId": request_data['workflowId'] + postfix,
                "token": request_data['token'],
                "lang": request_data['lang'],
                "input": input_data[begin:end],
            })
        return split_data


class EntitiesandTags(CloudRunService):

    def __init__(self, url):
        service_name = 'Entities_and_Tags'
        super(EntitiesandTags, self).__init__(service_name, url)

    def make_data(self, postfix, request_data, input_data, max_len):
        split_data = []
        for i, text in enumerate(input_data['refined_segment']):
            for begin in range(0, len(text), max_len):
                end = min(begin + max_len, len(text))
                split_data.append({
                    "requestId": request_data['workflowId'] + postfix,
                    "token": request_data['token'],
                    "lang": request_data['lang'],
                    "salient_sentence": input_data['salient_sentence'][i],
                    "input": text[begin:end],
                })
        return split_data
