import os
import time
import html
import json
import asyncio
import logging
import requests
from google.cloud import translate_v2 as translate


env = os.getenv('env', 'dev')
cloud_env = os.getenv('cloud_environment', 'GCP')
project_id = os.getenv('project_id', 'deephow-dev')

#config_path = os.getenv('configuration_file_path', 'config.json')
#config_json = json.load(open(config_path, 'r'))
#config = config_json[env][cloud_env]
#token = config["token"]
#callback_token = config["callback_token"]

params_path = os.getenv('params_file_path', 'params.json')
params = json.load(open(params_path, 'r'))
supported_lang = params['supported_lang']
mt5_max_len = params['max_len'][cloud_env]['MT5_max_len']
vit_max_len = params['max_len'][cloud_env]['VIT_max_len']
flan_max_len = params['max_len'][cloud_env]['FLAN_max_len']
stage1_use_max_len = params['max_len'][cloud_env]['stage1_USE_max_len']
stage1_teg_max_len = params['max_len'][cloud_env]['stage1_TEG_max_len']
stage2_et_max_len = params['max_len'][cloud_env]['stage2_ET_max_len']
stage3_use_max_len = params['max_len'][cloud_env]['stage3_USE_max_len']


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        res = method(*args, **kw)
        t = time.time() - ts
        logging.info(f' * Execution time of {method.__name__}: {t:.2f} secs')
        return res

    return timed


def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return asyncio.get_event_loop()


def validate_input_schema(request_data: dict):
    logging.info('Checking request data...')
    check_workflowId(request_data)
    check_token(request_data)
    check_data(request_data)
    check_params(request_data)


def check_params(request_data):
    if 'params' in request_data:
        if 'intra_weight' in request_data['params']:
            if not isinstance(request_data['params']['intra_weight'], (int, float)):
                raise Exception('intra weight type error.')
        if 'genCandidates' in request_data['params']:
            if not isinstance(request_data['params']['genCandidates'], bool):
                raise Exception('genCandidates type error.')
        if 'exact_step' in request_data['params']:
            if not isinstance(request_data['params']['exact_step'], (int, float)):
                raise Exception('exact_step type error.')
            if request_data['params']['exact_step'] < 1:
                raise Exception('exact_step has a minimum step = 1')
            if request_data['params']['exact_step'] > len(request_data['input']):
                raise Exception(f"exact_step has a maximum step < {len(request_data['input'])}")
        if 'summarize' in request_data['params']:
            if not isinstance(request_data['params']['summarize'], (str)):
                raise Exception('summarize type error.')
            if request_data['params']['summarize'] != 'flan':
                raise Exception('summarize use flan ?')


def check_workflowId(request_data):
    if 'workflowId' not in request_data or request_data['workflowId'] == "":
        raise Exception('Workflow Id not provided.')


def check_token(request_data):
    if 'token' not in request_data:
        raise Exception('No token provided')
    elif request_data['token'] != token:
        wrong_token = request_data['token']
        raise Exception(f'Wrong token: {wrong_token}')


def check_data(request_data):
    if 'input' not in request_data or len(request_data['input']) == 0:
        raise Exception('No input provided')
    elif 'lang' not in request_data or len(request_data['lang']) == 0:
        raise Exception('No lang provided')
    else:
        for i, input_data in enumerate(request_data['input']):
            if 'sentence' not in input_data or input_data['sentence'] == "":
                raise Exception(f'Sentence not provided in input[{i}]')
            elif type(input_data['sentence']) != str:
                wrong_type = type(input_data['sentence'])
                raise Exception(f'Wrong sentence type: {wrong_type}. The sentence type should be str')
            elif 'image' not in input_data or input_data['image'] == "":
                raise Exception(f'Image path not provided in input[{i}]')
            elif type(input_data['image']) != str:
                wrong_type = type(input_data['image'])
                raise Exception(f'Wrong image path type: {wrong_type}. The image path type should be str')


def google_translate_api(text: str, target: str):
    translate_client = translate.Client()
    result = translate_client.translate(text, target_language=target)
    ret = html.unescape(result['translatedText'])
    if target == 'en' and '.' in ret:
        ret_split = []
        for s in ret.split('.'):
            if s != '':
                ret_split.append(s + '.')
        ret = ' '.join(ret_split)
    return ret


def batch_translate_text(
    project_id="YOUR_PROJECT_ID",
    timeout=180,
):
    """Translates a batch of texts on GCS and stores the result in a GCS location."""

    input_uri="gs://deephow-dev.appspot.com/translation/input.txt",
    output_uri="gs://deephow-dev.appspot.com/translation/result.txt",

    client = translate.TranslationServiceClient()

    location = "us-central1"
    # Supported file types: https://cloud.google.com/translate/docs/supported-formats
    gcs_source = {"input_uri": input_uri}

    input_configs_element = {
        "gcs_source": gcs_source,
        "mime_type": "text/plain",  # Can be "text/plain" or "text/html".
    }
    gcs_destination = {"output_uri_prefix": output_uri}
    output_config = {"gcs_destination": gcs_destination}
    parent = f"projects/{project_id}/locations/{location}"

    # Supported language codes: https://cloud.google.com/translate/docs/languages
    operation = client.batch_translate_text(
        request={
            "parent": parent,
            "source_language_code": "en",
            "target_language_codes": ["ja"],  # Up to 10 language codes here.
            "input_configs": [input_configs_element],
            "output_config": output_config,
        }
    )

    print("Waiting for operation to complete...")
    response = operation.result(timeout)


def callback(data: dict, call_times: int = 3):
    callback_url = os.getenv('callback_url')
    try:
        for retry in range(call_times):
            callback_request = {
                'data': data,
                'token': callback_token,
            }
            logging.debug(f'callback to steps server ...{retry}')
            response = requests.post(callback_url, json=callback_request)
            if response.status_code != 200:
                logging.error(f'callback error: {response.text}')
            return
    except Exception:
        logging.exception("")


def match_step_english_number_string(string: str) -> str:
    pattern = """(?x)           # free-spacing mode
    (?(DEFINE)
      # Within this DEFINE block, we'll define many subroutines
      # They build on each other like lego until we can define
      # a "big number"

      (?<one_to_9>
      # The basic regex:
      # one|two|three|four|five|six|seven|eight|nine
      # We'll use an optimized version:
      # Option 1: four|eight|(?:fiv|(?:ni|o)n)e|t(?:wo|hree)|
      #                                          s(?:ix|even)
      # Option 2:
      (?:f(?:ive|our)|s(?:even|ix)|t(?:hree|wo)|(?:ni|o)ne|eight)
      ) # end one_to_9 definition

      (?<ten_to_19>
      # The basic regex:
      # ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|
      #                                              eighteen|nineteen
      # We'll use an optimized version:
      # Option 1: twelve|(?:(?:elev|t)e|(?:fif|eigh|nine|(?:thi|fou)r|
      #                                             s(?:ix|even))tee)n
      # Option 2:
      (?:(?:(?:s(?:even|ix)|f(?:our|if)|nine)te|e(?:ighte|lev))en|
                                              t(?:(?:hirte)?en|welve))
      ) # end ten_to_19 definition

      (?<two_digit_prefix>
      # The basic regex:
      # twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety
      # We'll use an optimized version:
      # Option 1: (?:fif|six|eigh|nine|(?:tw|sev)en|(?:thi|fo)r)ty
      # Option 2:
      (?:s(?:even|ix)|t(?:hir|wen)|f(?:if|or)|eigh|nine)ty
      ) # end two_digit_prefix definition

      (?<one_to_99>
      (?&two_digit_prefix)(?:[- ](?&one_to_9))?|(?&ten_to_19)|
                                                  (?&one_to_9)
      ) # end one_to_99 definition

      (?<one_to_999>
      (?&one_to_9)[ ]hundred(?:[ ](?:and[ ])?(?&one_to_99))?|
                                                (?&one_to_99)
      ) # end one_to_999 definition

      (?<one_to_999_999>
      (?&one_to_999)[ ]thousand(?:[ ](?&one_to_999))?|
                                        (?&one_to_999)
      ) # end one_to_999_999 definition

      (?<one_to_999_999_999>
      (?&one_to_999)[ ]million(?:[ ](?&one_to_999_999))?|
                                       (?&one_to_999_999)
      ) # end one_to_999_999_999 definition

      (?<one_to_999_999_999_999>
      (?&one_to_999)[ ]billion(?:[ ](?&one_to_999_999_999))?|
                                       (?&one_to_999_999_999)
      ) # end one_to_999_999_999_999 definition

      (?<one_to_999_999_999_999_999>
      (?&one_to_999)[ ]trillion(?:[ ](?&one_to_999_999_999_999))?|
                                        (?&one_to_999_999_999_999)
      ) # end one_to_999_999_999_999_999 definition

      (?<bignumber>
      zero|(?&one_to_999_999_999_999_999)
      ) # end bignumber definition

      (?<step>
      step
      ) # end bignumber definitionszdd[3

      (?<zero_to_9>
      (?&one_to_9)|zero
      ) # end zero to 9 definition

      (?<decimals>
      point(?:[ ](?&zero_to_9))+
      ) # end decimals definition

    ) # End DEFINE


    ####### The Regex Matching Starts Here ########
    ^?(?&step)[ ](?&bignumber)(?:[ ](?&decimals))*

    """

    import regex
    pattern = regex.compile(pattern)
    return (pattern.match(string) is not None)
