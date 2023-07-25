import asyncio
import numpy as np
from typing import List
from sklearn.preprocessing import normalize
from CloudRunService import MT5SentenceEncoder, CloudRunService


mt5_url = 'https://mt5-sentence-encoder-3oqy6iorxq-uc.a.run.app'
mt5_sentence_encoder = MT5SentenceEncoder(url=mt5_url)



def segmentation(input_data: dict):
    sentenced_text = [sentence for sentence in input_data['input']]
    sentenced_embeddings = get_embedding(input_data, sentenced_text)
    segments, detail = segment(sentenced_text, sentenced_embeddings)
    return segments


async def run(se_split_data):
    tasks = [
        CloudRunService.internal_post(
            mt5_sentence_encoder.url,
            data,
            i,
            len(se_split_data),
            mt5_sentence_encoder.retry_limit,
        ) for i, data in enumerate(se_split_data)
    ]
    se_results = await asyncio.gather(*(tasks))
    return se_results

    
def get_embedding(input_data: dict, sentenced_text: List[str]):
    
    se_split_data = mt5_sentence_encoder.make_data(
        postfix='_sentences',
        request_data=input_data,
        input_data=sentenced_text,
        max_len=128,
    )
    
    loop = asyncio.get_event_loop()
    se_results = loop.run_until_complete(run(se_split_data))
    
    sentenced_embeddings = np.vstack([
        normalize(res.json()['sentences_embeddings']) for res in se_results
    ]).astype('float32')

    return sentenced_embeddings


def segment(sentenced_text, final_embeddings):
    import tools
    import algorithm as algo
    
    adjust_rate = 1.0
    n_segments = 3
    max_segment_number = len(final_embeddings)
    min_segment_number = 1
    
    segment_len = min(max(len(final_embeddings) / n_segments, min_segment_number), max_segment_number)
    penalty = tools.get_penalty([final_embeddings], segment_len) * adjust_rate
    duration_penalty = [0] * len(final_embeddings)
    
    trials = 20
    for _ in range(trials):  # trial on conflict with hard limits
        segmentation = algo.split_optimal(
            final_embeddings,
            penalty,
            duration_penalty,
            seg_limit=20,
        )
        # adjust = check_hard_limits(segmentation, request_data, max_segment_number)
        # if adjust:
        #     adjust_rate *= adjust
            
        segmented_text = tools.get_segments(sentenced_text, segmentation)
        break

    return segmented_text[1:], segmentation


if __name__ == "__main__":
    print(\
	segmentation({
        'token': '3lPDYZWupFO9tCUU2c5VUTiY4r6ciOvL',
        'workflowId': '123',
        'lang': 'en',
		'input': ["OK, let's talk about the next topic. What we are going to talk about is called self supervised learning.", 'Before we talk about self supervised learning, we must introduce Sesame Street. Why? Because somehow self supervised learning models are named after characters from Sesame Street.', 'I especially wore a Sesame Street T-shirt today. Everyone could take a look at.', "The Sesame Street T-shirt. The classmates who can't see clearly or online classmates can also look at this picture. Here. They are the same.", 'This is me and these characters are from Sesame Street.', 'For these Sesame Street characters, what kind of models are they? We first take a look at their names first.', 'Before we actually understand what they do, we get to know their names first.', 'This Red monster is called Elmo.', "For self supervised learning, there is a model called embeddings from language modeling, which is the earliest self supervised learning model. It's abbreviation is called Elmo.", 'After Elmo, there is another animal called Bert. It is also the abbreviation for the most popular self supervised model today.', 'Bert is the abbreviation of bidirectional encoder representation from Transformers. Elmo and Bert are both Sesame Street characters.', "After having two Sesame Street characters, Bert's best friend is this one. Who is it? It's name is Ernie. In fact, after having Bert two different models immediately appeared.", 'They are both called Ernie the full model name of the first one is enhanced representation from knowledge integration. Its abbreviation is called Ernie.', "It's a little bit weird, they just name it after Ernie because they want it to be.", 'You might think this is ridiculous enough, but this animal is called big bird and there is a model called Big Bird whose full name is.', 'Transformers for longer sequences now they even gave up making up words they have gave up collecting characters from their name.', "They just call it big bird and that's it. So in those self supervised learning models, there are a bunch of characters from Sesame Street.", 'No one has touched Cookie monster yet. It is waiting for you to make up Cookie Monster.', 'When it comes to Bert, I have to mention the attacking giant.', 'In the following I will mention the plot of the attacking giant.']})\
	)
