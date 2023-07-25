import openai
import tiktoken
import sys
import os
import time
from Levenshtein import distance
from tenacity import retry, stop_after_attempt, wait_exponential
from utils import get_args, count_the_number_of_tokens, count_the_length_of_prompt, levenshtein_distance, reverse_string, read_file
import nltk
import argparse
#nltk.download('punkt')

def find_the_longest_article_that_fits_the_len_limit_of_chatGPT(article:str, openai_model_type:str, acceptable_len_of_gpt_input:int) -> str:
	sentences_in_article = nltk.tokenize.sent_tokenize(article)
	tokenizer = tiktoken.encoding_for_model(openai_model_type)
	encoding_article = tokenizer.encode(article)
	
	article_with_valid_len = ""
	article_buffer = ""
	sentence_index = 0
	max_len_of_input = (acceptable_len_of_gpt_input - count_the_length_of_prompt(openai_model_type)) // 2 - 100
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



@retry
def segment_short_article(article:str, openai_model_type:str, acceptable_len_of_gpt_input:int) -> str:
	prompt = read_file("./segment_prompt.txt")
	prompt += article
	prompt += "\n```\n</Question-2>\n<Answer-2>\n"
	max_len_of_response = acceptable_len_of_gpt_input - 1\
							- count_the_number_of_tokens(prompt, openai_model_type) - 8

	prompt_start_time = time.time()
	response = openai.ChatCompletion.create(
		model = openai_model_type, 
		temperature = 0.3,
		max_tokens = max_len_of_response,
		messages = [{"role": "user", "content": prompt}]
	)
	print("Time in prompt = " + str(time.time() - prompt_start_time) + " secs.")

	return_of_chatGPT = ""
	for choice in response.choices:
		return_of_chatGPT += choice.message.content
	
	return return_of_chatGPT



def locate_paragraph_tail(paragraph:str, article:str) -> int:
	candidate = reverse_string(paragraph)
	universe = reverse_string(article)
	best_score = len(candidate)
	best_score_position = -1
	for start_index_of_article in range(len(universe)):
		if start_index_of_article < 2 or universe[start_index_of_article] != ".":
			continue
		current_score = levenshtein_distance(candidate, universe[start_index_of_article:start_index_of_article + len(candidate)])
		if current_score < best_score:
			best_score = current_score
			best_score_position = start_index_of_article
	return len(article) - 1 - best_score_position + 1



def extract_paragraphs(segmented_article:str) -> list:
	lines = segmented_article.split("\n")
	paragraphs = []
	for line in lines:
		if line.startswith("Topic"):
			paragraphs.append(line[line.find(":") + 2:])
	return paragraphs



def segment_article(article:str, openai_model_type:str, acceptable_len_of_gpt_input:int, prevent_long_segment_flag:str) -> list:
	segmentable_article = find_the_longest_article_that_fits_the_len_limit_of_chatGPT(article, openai_model_type, acceptable_len_of_gpt_input)
	segmented_article = segment_short_article(segmentable_article, openai_model_type, acceptable_len_of_gpt_input)
	paragraphs = extract_paragraphs(segmented_article)

	last_tail = 0
	current_tail = 0
	valid_segments = []
	for idx, paragraph in enumerate(paragraphs):
		if idx == len(paragraphs) - 1 and len(segmentable_article.strip()) == len(article.strip()):
			current_tail = len(article)
		else:
			current_tail = last_tail + locate_paragraph_tail(paragraph, article[last_tail:])
		segment = article[last_tail:current_tail].strip()
		if prevent_long_segment_flag == "ON" and len(nltk.word_tokenize(segment)) > 700:
			valid_segments = valid_segments + segment_article(segment, openai_model_type, acceptable_len_of_gpt_input, "OFF")
		else:
			valid_segments.append(article[last_tail:current_tail].strip())
		last_tail = current_tail

	if last_tail == len(article):
		return valid_segments
	else:
		return valid_segments + segment_article(article[last_tail:], openai_model_type, acceptable_len_of_gpt_input, prevent_long_segment_flag)



def segment_workflow(transcription:str, args) -> list:
	acceptable_len_of_gpt_input = 4096
	if args.gpt_model_type.endswith("16k"):
		acceptable_len_of_gpt_input = 16384
	elif args.gpt_model_type.endswith("4"):
		acceptable_len_of_gpt_input = 8192
	return segment_article(transcription, args.gpt_model_type, acceptable_len_of_gpt_input, args.prevent_long_segment_flag)





if __name__ == "__main__":
	openai.api_key = "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"

	args = get_args()
	arg1 = "4k"
	if args.gpt_model_type.endswith("16k"):
		arg1 = "16k"
	elif args.gpt_model_type.endswith("4"):
		arg1 = "gpt4"

	arg2 = args.prevent_long_segment_flag

	file_name = args.file_name
	file_path = "./../transcription/trans" + file_name + ".txt"
	transcription = ""
	with open(file_path, "r") as fp:
		line = fp.readline()
		while line:
			transcription += line
			line = fp.readline()
	start_segmentation_time = time.time()
	segmented_transcript = segment_workflow(transcription, args)
	print("Total time segmentation takes = " + str(time.time() - start_segmentation_time) + " secs.")
'''
# Choi dataset code
	transcript_dir_path = args.dir_path
	file_names = os.listdir(transcript_dir_path)
	file_names.reverse()

	for file_name in file_names:
		print(transcript_dir_path + "/" + file_name + " processing")

		trans_buf = transcript_dir_path.replace("choiDataset", "choiResult" + arg1 + arg2)
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

		transcript_des_path = transcript_dir_path.replace("choiDataset", "choiResult" + arg1 + arg2)
		with open(transcript_des_path + "/" + file_name, "w") as fp:
			for paragraph in segmented_transcript:
				fp.write(paragraph + "\n\n")

'''
