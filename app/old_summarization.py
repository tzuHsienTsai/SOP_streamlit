import json
import faiss
import string
import numpy as np
import stopwordsiso as sw
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import OrderedDict

from embedding import universal_sentence_encoder, _get_embeddings, entities_and_tags
from utils import stage1_use_max_len, stage2_et_max_len, stage3_use_max_len, timeit


@timeit
def stage1(request_data, refined_segmented_text, sentenced_text, sentenced_embeddings):

    # send request to universal sentence encoder
    segmented_text = [
        ' '.join(text) for text in refined_segmented_text + [sentenced_text]
    ]
    split_data = universal_sentence_encoder.make_data(
        postfix='_segments',
        request_data=request_data,
        input_data=segmented_text,
        max_len=stage1_use_max_len,
    )
    results = universal_sentence_encoder.get_result(split_data)
    output_embeddings = np.vstack([
        _get_embeddings(r.text) for r in results
    ]).astype('float32')

    # get salient sentences
    segment_embeddings = output_embeddings[:-1]
    whole_text_embedding = output_embeddings[-1]
    salient_sentences = []
    start = 0
    for i in range(len(segment_embeddings)):
        index = faiss.IndexFlatL2(segment_embeddings.shape[-1])
        segmented_text_len = len(refined_segmented_text[i])
        index.add(sentenced_embeddings[start:start + segmented_text_len])
        distance, index = index.search(np.array([segment_embeddings[i]]), 1)
        salient_sentences.append(sentenced_text[start + index[0][0]])
        start += segmented_text_len
    
    return {
        'refined_segmented_text': refined_segmented_text,
        'salient_sentences': salient_sentences,
        'segment_embeddings': segment_embeddings,
        'whole_text_embedding': whole_text_embedding,
    }


@timeit
def stage2(request_data, staged_results):
    # extract results from previous stages
    stage1_result = staged_results[0]
    refined_segmented_text = stage1_result['refined_segmented_text']
    salient_sentences = stage1_result['salient_sentences']

    # send request to entities and tags generator
    input_data = {
        'salient_sentence': salient_sentences,
        'refined_segment': refined_segmented_text,
    }
    split_data = entities_and_tags.make_data(
        postfix='_entities_and_tags',
        request_data=request_data,
        input_data=input_data,
        max_len=stage2_et_max_len,
    )
    results = entities_and_tags.get_result(split_data)

    # get grams list and salient POS list
    grams_list = []
    salient_pos_list = []
    length = [
        (len(t) - 1) // stage2_et_max_len + 1 for t in refined_segmented_text
    ]
    cum_sum = [0] + list(np.cumsum(length)[:-1])
    for i, cum_len in enumerate(cum_sum):
        if length[i] == 1:
            text = json.loads(results[cum_len].text)
            grams_list.append(text['grams_list'])
            salient_pos_list += text['salient_pos_list']
        else:

            def get_text(j):
                return json.loads(results[cum_len + j].text)

            g_list = [get_text(j)['grams_list'] for j in range(length[i])]
            grams_list.append([item for items in g_list for item in items])
            acc = 0
            for j in range(length[i]):
                pos_list = get_text(j)['salient_pos_list']
                if pos_list != []:
                    salient_pos_list += [pos + acc for pos in pos_list]
                    break
                acc += len(g_list[j])

    whole_grams = [item for items in grams_list for item in items]
    return {
        'whole_grams': whole_grams,
        'grams_list': grams_list,
        'salient_pos_list': salient_pos_list,
    }


@timeit
def stage3(request_data, staged_results):
    # extract results from previous stages
    stage1_result = staged_results[0]
    stage2_result = staged_results[1]
    segment_embeddings = stage1_result['segment_embeddings']
    whole_text_embedding = stage1_result['whole_text_embedding']
    whole_grams = stage2_result['whole_grams']
    grams_list = stage2_result['grams_list']
    salient_pos_list = stage2_result['salient_pos_list']
    lang = request_data['lang']
    salient_grams = []

    # send request to universal sentence encoder
    delimiter = '' if lang == 'zh' else ' '
    input_whole_grams = [
        delimiter.join(item[0] for item in grams) for grams in whole_grams
    ]
    split_data = universal_sentence_encoder.make_data(
        postfix='_grams',
        request_data=request_data,
        input_data=input_whole_grams,
        max_len=stage3_use_max_len,
    )
    results = universal_sentence_encoder.get_result(split_data)
    grams_embedding = np.vstack([
        _get_embeddings(r.text) for r in results
    ]).astype('float32')

    # get salient grams
    start = 0
    for i, grams in enumerate(grams_list):
        if grams != []:
            salient_index = faiss.IndexFlatL2(segment_embeddings.shape[-1])
            begin = start + salient_pos_list[2 * i]
            end = start + salient_pos_list[2 * i + 1]
            salient_index.add(grams_embedding[begin:end])
            distance, index = salient_index.search(np.array([segment_embeddings[i]]), 1)
            begin = salient_pos_list[2 * i]
            end = salient_pos_list[2 * i + 1]

            if lang == 'zh':
                summary = [g[0] for g in grams[begin:end][index[0][0]]]
            else:
                summary = ' '.join(g[0] for g in grams[begin:end][index[0][0]])
            salient_grams.append(summary)
            start += len(grams)
        else:
            salient_grams.append('')

    # get whole text grams
    index = faiss.IndexFlatL2(segment_embeddings.shape[-1])
    index.add(grams_embedding)
    distance, index = index.search(np.array([whole_text_embedding]), 100)
    thres = {'en': 1.5, 'pt': 1, 'es': 1, 'de': 1, 'zh': 1, 'fr': 1, 'nl': 1}
    whole_text_grams = [
        whole_grams[index[0][i]] for i in range(len(index[0])) if distance[0][i] < thres[lang]
    ]
    n = {'en': 30, 'pt': 20, 'es': 20, 'de': 20, 'zh': 20, 'fr': 20, 'nl': 20}
    if len(whole_text_grams) == 0:
        whole_text_grams = [whole_grams[index[0][i]] for i in range(n[lang])]

    return {
        'salient_grams': salient_grams,
        'whole_text_grams': whole_text_grams,
    }


@timeit
def stage4(request_data, staged_results):
    # extract results from previous stages
    stage3_result = staged_results[2]
    salient_grams = stage3_result['salient_grams']
    whole_text_grams = stage3_result['whole_text_grams']
    lang = request_data['lang']

    # define stopwords
    chinese_stopwords = _load_chinese_stopwords()
    lang_map = {
        'pt': 'portuguese',
        'es': 'spanish',
        'de': 'german',
        'fr': 'french',
        'nl': 'dutch',
    }
    if lang == 'en':
        removed_stopwords = set(['and'])
        stop_words = set(stopwords.words('english') + list(string.punctuation) + ["'m", "'s"]) - removed_stopwords
    elif lang == 'zh':
        stop_words = set(list(sw.stopwords(["zh"])) + chinese_stopwords)
    else:
        stop_words = set(stopwords.words(lang_map[lang]))

    # remove stopwords in salient grams
    if lang == 'zh':
        for i in range(len(salient_grams)):
            salient_grams[i] = ''.join(
                [w for w in salient_grams[i] if w not in stop_words])
    else:
        for i in range(len(salient_grams)):
            word_tokens = word_tokenize(salient_grams[i])
            salient_grams[i] = ' '.join(
                [w for w in word_tokens if not w.lower() in stop_words])

    # define tags
    if lang == "pt":
        tags = set(['NOUN', 'ADJ'])
    elif lang == 'zh':
        tags = set(['n', 'v', 'r'])
    else:
        tags = set(['<ADJ>', '<NOUN>', '<PROPN>'])

    # remove stopwords in whole text grams
    if lang == 'zh':
        for i in range(len(whole_text_grams)):
            tokens = []
            for word, pos in whole_text_grams[i]:
                if sum([pos.startswith(item) for item in tags]):
                    tokens.append(word)
            whole_text_grams[i] = ''.join(
                [w for w in tokens if w not in stop_words])
    else:
        for i in range(len(whole_text_grams)):
            tokens = []
            for word, pos in whole_text_grams[i]:
                if pos in tags:
                    tokens.append(word)
            whole_text_grams[i] = ' '.join(
                [w for w in tokens if not w.lower() in stop_words])
    whole_text_grams = list(
        filter(
            None,
            list(OrderedDict.fromkeys(whole_text_grams)),
        ))

    if lang != 'zh':
        # get stemmer
        if lang == 'en':
            stemmer = PorterStemmer()
        elif lang in ['pt', 'es', 'de', 'fr', 'nl']:
            stemmer = SnowballStemmer(lang_map[lang])
        stemmed = [
            [stemmer.stem(w) for w in kw.split(" ")] for kw in whole_text_grams
        ]

        # remove duplicated grams in whole text grams
        for j, grams_a in enumerate(whole_text_grams):
            if grams_a != '':
                for k in range(j + 1, len(whole_text_grams)):
                    grams_b = whole_text_grams[k]
                    if grams_b != '':
                        conditions = [
                            grams_a in grams_b,
                            grams_b in grams_a,
                            all(x in iter(' '.join(stemmed[j]))
                                for x in ' '.join(stemmed[k])),
                            all(x in iter(' '.join(stemmed[k]))
                                for x in ' '.join(stemmed[j])),
                        ]
                        if any(conditions):
                            whole_text_grams[k] = ''
        whole_text_grams = list(filter(None, whole_text_grams))

    return salient_grams, whole_text_grams


def get_segmentation_ret(request_data, staged_results):
    stage1_result = staged_results[0]
    salient_grams, whole_text_grams = staged_results[3]

    return {
        "salient_sentence": stage1_result['salient_sentences'],
        "segmented_text": stage1_result['refined_segmented_text'],
        "summaries": salient_grams,
        "video_tagging": whole_text_grams,
        "workflowId": request_data["workflowId"],
    }


def _load_chinese_stopwords():
    return [
        item.rstrip() for item in open(
            'chinese_stopwords.txt', 'r', encoding='utf-8'
        ).readlines()
    ]
