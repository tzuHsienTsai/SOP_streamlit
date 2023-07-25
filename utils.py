import openai
import time
import tiktoken
import sys
import os
from Levenshtein import distance
from tenacity import retry, stop_after_attempt, wait_exponential
import nltk
import argparse
from old_segmentation import segmentation
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize

def get_args():
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument("--gpt_model_type", default="gpt-3.5-turbo", type=str)
	parser.add_argument("--prevent_long_segment", default="NO", type=str)
	parser.add_argument("--prevent_short_segment", default="NO", type=str)
	parser.add_argument("--file_name", nargs='?', default="", type=str)
	parser.add_argument("--dir_path", nargs='?', default="1", type=str)
	return parser.parse_args()

def count_the_number_of_tokens(text:str, openai_model_type:str):
	tokenizer = tiktoken.encoding_for_model(openai_model_type)
	encoded_text = tokenizer.encode(text)
	return len(encoded_text)

def count_the_length_of_prompt(openai_model_type:str) -> int:
	prompt = read_file("./segment_prompt.txt")
	prompt += "\n```\n</Question-2>\n<Answer-2>\n"

	num_of_tokens_in_the_header_of_prompt = 8

	return count_the_number_of_tokens(prompt, openai_model_type) + num_of_tokens_in_the_header_of_prompt

def read_file(file_path:str) -> str:
	content = ""
	content_lines = []
	with open(file_path, "r") as fp:
		content_lines = fp.readlines()
	for line in content_lines:
		content += line
	return content

def levenshtein_distance(string_a:str, string_b:str) -> int:
	return distance(string_a, string_b)	

def reverse_string(string:str) -> str:
	return string[::-1]

def merge_segments_text_split_version(short_segment:str, previous_segment:str, next_segment:str) -> list:
	previous_segment_sentences = sent_tokenize(previous_segment)
	next_segment_sentences = sent_tokenize(next_segment)
	short_segment_sentences = sent_tokenize(short_segment)

	if len(previous_segment_sentences) == 0:
		return [previous_segment, short_segment + " " + next_segment]
	if len(next_segment_sentences) == 0:
		return [previous_segment + " " + short_segment, next_segment]

	segments = segmentation({
						'token': '3lPDYZWupFO9tCUU2c5VUTiY4r6ciOvL',
						'workflowId': '123',
						'lang': 'en',
						'input': [previous_segment_sentences[-1]] + short_segment_sentences + [next_segment_sentences[0]]
#						'input': previous_segment_sentences + short_segment_sentences + next_segment_sentences
						})

	new_previous_segment = previous_segment
	new_next_segment = next_segment
	if len(segments) == 1 or len(segments[0]) == len(segments[-1]):
		if len(previous_segment) > len(next_segment):
			new_next_segment = short_segment + " " + next_segment
		else:
			new_previous_segment = previous_segment + " " + short_segment
	else:
		if len(segments[0]) < len(segments[-1]):
			new_next_segment = short_segment + " " + next_segment
		else:
			new_previous_segment = previous_segment + " " + short_segment

	return [new_previous_segment, new_next_segment]
		
@retry(wait=wait_exponential(multiplier=1, min=1, max=5))
def merge_segments_gpt_version(short_segment:str, previous_segment:str, next_segment:str, openai_model_type) -> list:
	prompt = read_file("./segment_relation_prompt.txt")
	prompt += "\nSegment A: " + short_segment
	prompt += "\nSegment B: " + previous_segment
	prompt += "\nSegment C: " + next_segment
	prompt += "\nPlease determine whether segment A is more semantically related to segment B or segment C.\n</Question 4>\n<Answer 4>"
#	print("Request sent. (small segment)")
	prompt_start_time = time.time()
	response = openai.ChatCompletion.create(
		model = openai_model_type,
		temperature = 0.3,
		max_tokens = 1024,
		messages = [{"role": "user", "content": prompt}],
	)
#	print("Time in prompt (small segment) = " + str(time.time() - prompt_start_time) + " secs.")
#	print("Request return.")

	return_of_chatGPT = ""
	for choice in response.choices:
		return_of_chatGPT += choice.message.content
	
#	print(return_of_chatGPT)
	new_previous_segment = previous_segment
	new_next_segment = next_segment
	if return_of_chatGPT.find(" B.") >= 0:
		new_previous_segment = previous_segment + " " + short_segment
	elif return_of_chatGPT.find(" C.") >= 0:
		new_next_segment = short_segment + " " + next_segment
	else:
		if len(word_tokenize(previous_segment)) > len(word_tokenize(next_segment)):
			new_next_segment = short_segment + " " + next_segment
		else:
			new_previous_segment = previous_segment + " " + short_segment
			
	return [new_previous_segment, new_next_segment]

if __name__ == "__main__":
#	print(read_file("./segment_prompt.txt"))
	openai.api_key = "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"
	previous_segment = "Prime numbers are tricky things. We learn in school that they’re numbers with no factors other than 1 and themselves, and that mathematicians have known for thousands of years that an infinite number of them exist. Producing one on command doesn’t seem as if it should be difficult."
	short_segment = "Constructing arbitrarily large prime numbers is remarkably complicated. You basically have two computational options, both with drawbacks."
	next_segment = "I place the tour within the political, social and sporting conditions under apartheid. In 1971 South Africa was a racist, segregated and repressive society, based on white supremacy and privilege and black subjugation. Black people were denied proper sports facilities, coaching and opportunities to excel, could not belong to the same clubs as whites or compete with or against white players. Considered subjects, not citizens, they couldn’t represent South Africa in sport. Sport under apartheid was a killing field of ambitions and dreams."
#	merge_result = merge_segments_text_split_version(short_segment, previous_segment, next_segment)
	merge_result = merge_segments_gpt_version(short_segment, previous_segment, next_segment, "gpt-3.5-turbo")
#	print(merge_result)
