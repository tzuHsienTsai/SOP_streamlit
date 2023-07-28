import openai
import tiktoken
import sys
from tqdm import tqdm
import os
import time
import streamlit as st
from Levenshtein import distance
from tenacity import retry, stop_after_attempt, wait_exponential
from utils import get_args, count_the_number_of_tokens, count_the_length_of_prompt, levenshtein_distance, reverse_string, merge_segments_gpt_version, read_file, is_mandarin
import nltk
import argparse
from nltk.tokenize import sent_tokenize, word_tokenize
from stqdm import stqdm
#nltk.download('punkt')

def find_the_longest_article_that_fits_the_len_limit_of_chatGPT(article:str, openai_model_type:str, acceptable_len_of_gpt_input:int) -> str:
	sentences_in_article = nltk.tokenize.sent_tokenize(article)
	tokenizer = tiktoken.encoding_for_model(openai_model_type)
	encoding_article = tokenizer.encode(article)
	
	article_with_valid_len = ""
	article_buffer = ""
	sentence_index = 0
	max_len_of_input = acceptable_len_of_gpt_input - count_the_length_of_prompt(openai_model_type) - 100 - 1024
	# The -100 in the last arthimetic operation is the buffer for double newline and correctness of original article in the response of chatGPT.

	for sentence in sentences_in_article:
		if len(tokenizer.encode(article_with_valid_len + sentence)) >= max_len_of_input:
			break
		else:
			# For choi dataset
			if len(article_with_valid_len) > 0 and article_with_valid_len[-1] != " ":
				article_with_valid_len += " "
			article_with_valid_len += sentence

	if len(article_with_valid_len) > 0:
		return article_with_valid_len
	else:
		return tokenizer.decode(tokenizer.encode(article)[:max_len_of_input])



@retry(wait=wait_exponential(multiplier=1, min=4, max=10))
def segment_short_article(article:str, openai_model_type:str, acceptable_len_of_gpt_input:int) -> str:
	prompt = read_file("./segment_prompt_v2.txt")
	prompt += article
	prompt += "\n```\n</Question 4>\n<Answer 4>\n"
	max_len_of_response = acceptable_len_of_gpt_input - 1\
							- count_the_number_of_tokens(prompt, openai_model_type) - 8

	prompt_start_time = time.time()
#	print("Request sent.")
	response = openai.ChatCompletion.create(
		model = openai_model_type, 
		temperature = 0.3,
		max_tokens = max_len_of_response,
		messages = [{"role": "user", "content": prompt}],
	)
#	print("Request return.")
#	print("Time in prompt = " + str(time.time() - prompt_start_time) + " secs.")

	return_of_chatGPT = ""
	for choice in response.choices:
		return_of_chatGPT += choice.message.content
	
	return return_of_chatGPT

def locate_segment_start(sentence_prefix:str, article:str) -> int:
	best_score = len(sentence_prefix)
	best_score_position = -1
	head = 0
	puncs = [".", "!", "?", "。", "！", "？"]
	for start_index_of_article in range(len(article) - 1):
		if start_index_of_article > 2 and article[start_index_of_article] in puncs:
			head = start_index_of_article + 1
			while head < len(article) and article[head] == " ":
				head += 1
		current_score = levenshtein_distance(sentence_prefix, article[start_index_of_article:start_index_of_article + len(sentence_prefix)])
		if current_score < best_score:
			best_score = current_score
			best_score_position = head
	return best_score_position



def extract_the_head_of_paragraphs(segmented_article:str) -> list:
	lines = segmented_article.split("\n")
	sentences_prefix = []
	for line in lines:
		if line.startswith("Topic"):
			prefix = line[line.find(":") + 2:]
			while len(prefix) > 0 and prefix[-1] in [".", "\n"]:
				prefix = prefix[:-1]
			sentences_prefix.append(prefix)
	return sentences_prefix

num_of_API_call = 0

def segment_article(article:str, openai_model_type:str, acceptable_len_of_gpt_input:int, prevent_long_segment:str, prevent_short_segment:str)->list:
	segmentable_article = find_the_longest_article_that_fits_the_len_limit_of_chatGPT(article, openai_model_type, acceptable_len_of_gpt_input)
	global num_of_API_call
	num_of_API_call += 1
	segmented_article = segment_short_article(segmentable_article, openai_model_type, acceptable_len_of_gpt_input)
	sentences_prefix = extract_the_head_of_paragraphs(segmented_article)
#	if prevent_long_segment == "NO":
#		print(segmented_article, file=sys.stderr)
#		print(sentences_prefix, file=sys.stderr)
#		print("==================", file=sys.stderr)

	last_head = 0
	current_head = 0
	valid_segments = []
	for idx, sentence_prefix in enumerate(sentences_prefix):
		if idx == 0:
			continue
		current_head = last_head + locate_segment_start(sentence_prefix, article[last_head:])
#		print(current_head, file=sys.stderr)
		segment = article[last_head:current_head].strip()
		if len(segment) > 0:
			valid_segments.append(segment)
		last_head = current_head

	if len(segmentable_article.strip()) == len(article.strip()):
		segment = article[last_head:len(segmentable_article)].strip()
		if len(segment) > 0:
			valid_segments.append(segment)
		
	
	if prevent_long_segment == "YES":
		segment_idx = 0
		while segment_idx < len(valid_segments):
			segment = valid_segments[segment_idx]
			if (len(word_tokenize(segment)) > 700 and is_mandarin(segment.strip()) == False) or (len(segment) > 860 and is_mandarin(segment.strip()) == True):
				split_once_more_segments = segment_article(segment,\
														openai_model_type,\
														acceptable_len_of_gpt_input,\
														"NO",\
														prevent_short_segment,\
														)
				num_of_API_call += 1
				for idx in range(len(split_once_more_segments)):
					while idx < len(split_once_more_segments)\
							and len(split_once_more_segments[idx]) == 0:
						split_once_more_segments.pop(idx)
				valid_segments = valid_segments[:segment_idx] + split_once_more_segments + valid_segments[segment_idx + 1:]
			else:
				segment_idx += 1

	if prevent_short_segment == "YES":
		idx = 0
		while idx < len(valid_segments) and len(valid_segments) > 1:
			if (len(word_tokenize(valid_segments[idx])) < 70 and len(word_tokenize(valid_segments[idx])) * 3 <= len(word_tokenize(segmentable_article)) and is_mandarin(valid_segments[idx].strip()) == False) or (len(valid_segments[idx]) < 86 and len(valid_segments[idx]) * 3 <= len(segmentable_article) and is_mandarin(valid_segments[idx].strip()) == True):
				if idx == 0:
					new_segment = valid_segments[idx] + " " + valid_segments[1]
					valid_segments = [new_segment] + valid_segments[2:]
				elif idx == len(valid_segments) - 1:
					new_segment = valid_segments[idx - 1] + " " + valid_segments[idx]
					valid_segments = valid_segments[:-2] + [new_segment]
				else:
					merge_result = merge_segments_gpt_version(valid_segments[idx],\
#					merge_result = merge_segments_text_split_version(valid_segments[idx],\
																valid_segments[idx - 1],\
																valid_segments[idx + 1],\
#																)
																openai_model_type)
					valid_segments = valid_segments[:idx - 1] + merge_result + valid_segments[idx + 2:]
					num_of_API_call += 1
			else:
				idx += 1

	global progress
#	global streamlit_progress
	if len(segmentable_article.strip()) == len(article.strip()):
		if prevent_long_segment == "YES":
			progress.update(len(segmentable_article.strip()))
			progress.set_description("Workflow segmentation")
#			streamlit_progress.update(len(segmentable_article.strip()))
#			streamlit_progress.set_description("Workflow segmentation")
		return valid_segments
	else:
		if prevent_long_segment == "YES":
			progress.update(len(article[:current_head].strip()))
			progress.set_description("Workflow segmentation")
#			streamlit_progress.update(len(article[:current_head].strip()))
#			streamlit_progress.set_description("Workflow segmentation")
		return valid_segments + segment_article(article[current_head:], openai_model_type, acceptable_len_of_gpt_input, prevent_long_segment, prevent_short_segment)


progress = None
#streamlit_progress = None

def segment_workflow(transcription:str, args) -> list:
	acceptable_len_of_gpt_input = 4096
	if args.gpt_model_type.endswith("16k"):
		acceptable_len_of_gpt_input = 16384
	elif args.gpt_model_type.endswith("4"):
		acceptable_len_of_gpt_input = 8192
	global progress
	global streamlit_progress
	progress = tqdm(total=len(transcription.strip()))
	progress.set_description("Workflow segmentation")
#	streamlit_progress = stqdm(total=len(transcription.strip()))
#	streamlit_progress.set_description("Workflow segmentation")
	return segment_article(transcription, args.gpt_model_type, acceptable_len_of_gpt_input, args.prevent_long_segment, args.prevent_short_segment)


if __name__ == "__main__":
	openai.api_key = "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"

	args = get_args()

# Choi dataset code
	arg1 = "4k"
	if args.gpt_model_type.endswith("16k"):
		arg1 = "16k"
	elif args.gpt_model_type.endswith("4"):
		arg1 = "gpt4"
	arg2 = args.prevent_long_segment
	arg3 = args.prevent_short_segment
	file_name = args.file_name
	file_path = "./../transcription/trans" + file_name + ".txt"
	transcription = ""
	with open(file_path, "r") as fp:
		line = fp.readline()
		while line:
			transcription += line
			line = fp.readline()
	start_segmentation_time = time.time()
	segmented_transcription = segment_workflow(transcription, args)
	for idx, segment in enumerate(segmented_transcription):
		print("Segment " + str(idx + 1) + ":")
		print(segment)
#	print("Total time segmentation takes = " + str(time.time() - start_segmentation_time) + " secs.")
#	print(segmented_transcript)
'''

	transcript_dir_path = args.dir_path
	file_names = os.listdir(transcript_dir_path)
#	file_names.reverse()

	for file_name in file_names:

		trans_buf = transcript_dir_path.replace("choiDataset", "choiResult" + arg1 + arg2 + arg3 + "tshort")
		print(trans_buf + "/" + file_name + " processing")
		if os.path.exists(trans_buf + "/" + file_name):
			continue

		transcript = ""
		with open(transcript_dir_path + "/" + file_name, "r") as fp:
			line = fp.readline()
			while line:
				if line.startswith("===") == False:
					transcript += line
				line = fp.readline()
		transcript = transcript.replace("\n", " ")
		transcript = transcript.replace("  ", " ")	

		segmented_transcript = segment_workflow(transcript, args)

		transcript_des_path = transcript_dir_path.replace("choiDataset", "choiResult" + arg1 + arg2 + arg3 + "tshort")
		with open(transcript_des_path + "/" + file_name, "w") as fp:
			for paragraph in segmented_transcript:
				fp.write(paragraph + "\n\n")

'''
