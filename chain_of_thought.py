import openai
import tiktoken
import sys
import os
from Levenshtein import distance
from tenacity import retry, stop_after_attempt, wait_exponential
import nltk
nltk.download('punkt')

def count_the_number_of_tokens(text:str, openai_model_type:str):
	tokenizer = tiktoken.encoding_for_model(openai_model_type)
	encoded_text = tokenizer.encode(text)
	return len(encoded_text)

def count_the_length_of_prompt(openai_model_type:str) -> int:
	prompt = load_prompt_of_workflow_segmentation()
	prompt += "\n```\n</Question-2>\n<Answer-2>\n"

	num_of_tokens_in_the_header_of_prompt = 8

	return count_the_number_of_tokens(prompt, openai_model_type) + num_of_tokens_in_the_header_of_prompt

def locate_the_longest_section_from_the_start_of_the_article_that_fits_the_char_limit_of_chatGPT(article:str, openai_model_type:str, acceptable_len_of_gpt_input:int) -> str:
	sentences_in_article = nltk.tokenize.sent_tokenize(article)
#	print(sentences_in_article)
	tokenizer = tiktoken.encoding_for_model(openai_model_type)
	encoding_article = tokenizer.encode(article)
	
	article_with_valid_len = ""
	article_buffer = ""
	sentence_index = 0
	max_len_of_input = (acceptable_len_of_gpt_input - count_the_length_of_prompt(openai_model_type)) // 2 - 100
#	print("Max length of input is: " + str(max_len_of_input))
#	print(article)
#	print(encoding_article)
	while sentence_index < len(sentences_in_article) and len(tokenizer.encode(article_buffer)) < max_len_of_input:
		article_with_valid_len = str(article_buffer)
		article_buffer += sentences_in_article[sentence_index]
		if article_buffer[-1] != " ":
			article_buffer += " "
		sentence_index += 1
	
	if sentence_index == len(sentences_in_article):
		article_with_valid_len = str(article_buffer)
#	print("=============\narticle with valid length is: " + str(article_with_valid_len))

	if len(article_with_valid_len) > 0:
		return article_with_valid_len
	else:
		return tokenizer.decode(tokenizer.encode(article)[:max_len_of_input])
	# The -100 in the last arthimetic operation is the buffer for double newline and correctness of original article in the response of chatGPT.

def load_prompt_of_workflow_segmentation() -> str:
	prompt_file_path = "./segment_prompt.txt"
	prompt = ""
	with open(prompt_file_path, "r") as fp:
		line = fp.readline()
		while line:
			prompt += line
			line = fp.readline()
	return prompt

@retry(wait=wait_exponential(multiplier=1, min=1, max=8))
def segment_short_article(article:str, openai_model_type:str, acceptable_len_of_gpt_input:int) -> str:
	prompt = load_prompt_of_workflow_segmentation()
	prompt += article
	prompt += "\n```\n</Question-2>\n<Answer-2>\n"
	max_len_of_response = acceptable_len_of_gpt_input - 1\
							- count_the_number_of_tokens(prompt, openai_model_type) - 8

#	print("Request sent.")
	response = openai.ChatCompletion.create(
		model = openai_model_type, 
		temperature = 0.3,
		max_tokens = max_len_of_response,
		messages = [{"role": "user", "content": prompt}]
	)
#	print("Request return.")

	return_of_chatGPT = ""
	for choice in response.choices:
		return_of_chatGPT += choice.message.content
	
	return return_of_chatGPT

def levenshtein_distance(string_a:str, string_b:str) -> int:
	return distance(string_a, string_b)	

def reverse_string(string:str) -> str:
	return string[::-1]

def find_the_end_of_paragraph_in_the_article(paragraph:str, article:str) -> int:
	candidate = reverse_string(paragraph)
	universe = reverse_string(article)
	best_score = len(candidate)
	best_score_position = -1
	for start_index_of_article in range(len(universe)):
		if start_index_of_article < 2 or universe[start_index_of_article] != ".":
			continue
		current_score = levenshtein_distance(candidate, article[start_index_of_article:start_index_of_article + len(candidate)])
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

def segment_article(article:str, openai_model_type:str, acceptable_len_of_gpt_input:int) -> str:
	segmentable_article = locate_the_longest_section_from_the_start_of_the_article_that_fits_the_char_limit_of_chatGPT(article, openai_model_type, acceptable_len_of_gpt_input)
	print("Remaining length: " + str(len(article)))
	print("Working on: "  + str(len(segmentable_article)))
	segmented_article = segment_short_article(segmentable_article, openai_model_type, acceptable_len_of_gpt_input)
	paragraphs = extract_paragraphs(segmented_article)

	last_end = 0
	current_end = 0
	valid_segments = ""
#	print(paragraphs)
	for paragraph in paragraphs:
		current_end = last_end + find_the_end_of_paragraph_in_the_article(paragraph, article[last_end:])
		valid_segments += (article[last_end:current_end].strip() + "\n\n")
		last_end = current_end

#	print(last_end)
	if len(segmentable_article.strip()) == len(article.strip()):
		valid_segments += article[last_end:] + "\n\n"
		return valid_segments
	else:
		return valid_segments + segment_article(article[last_end:], openai_model_type, acceptable_len_of_gpt_input)
	
def segment_workflow(transcription:str) -> str:
	openai_model_type = "gpt-3.5-turbo-16k"
	acceptable_len_of_gpt_input = 16384
	return segment_article(transcription, openai_model_type, acceptable_len_of_gpt_input)

if __name__ == "__main__":
	openai.api_key = "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"
	
	file_name = sys.argv[1]
	file_path = "./../transcription/trans" + file_name + ".txt"
	transcription = ""
	with open(file_path, "r") as fp:
		line = fp.readline()
		while line:
			transcription += line
			line = fp.readline()
	segmented_transcript = segment_workflow(transcription)
	print(segmented_transcript)

'''
# Choi dataset code
	transcript_dir_path = sys.argv[1]
	for file_name in os.listdir(transcript_dir_path):
		print(transcript_dir_path + "/" + file_name + " processing")

		trans_buf = transcript_dir_path.replace("choiDataset", "choiResult")
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

		segmented_transcript = segment_workflow(transcript)

		transcript_des_path = transcript_dir_path.replace("choiDataset", "choiResult")
		with open(transcript_des_path + "/" + file_name, "w") as fp:
			fp.write(segmented_transcript)
'''
