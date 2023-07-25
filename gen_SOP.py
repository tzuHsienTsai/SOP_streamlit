import openai
from workflow_segmentation_short_response import segment_workflow
import argparse
import re
import sys
import time
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
from utils import count_the_number_of_tokens, read_file

def get_args():
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument("--gpt_model_type", default="gpt-3.5-turbo", type=str)
	parser.add_argument("--prevent_long_segment", default="NO", type=str)
	parser.add_argument("--prevent_short_segment", default="NO", type=str)
	parser.add_argument("--file_path", default="", type=str)
	return parser.parse_args()

@retry(wait=wait_exponential(multiplier=1, min=4, max=10))
def summarize_article(article:str, openai_model_type:str)->str:
	prompt = read_file("./summarize_prompt.txt")
	if prompt[-1] != '\n':
		prompt += "\n"
	prompt += article
	if prompt[-1] != '\n':
		prompt += "\n"
	prompt += "```\n</Question 4>\n<Answer 4>"

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
	
	return return_of_chatGPT

@retry(wait=wait_exponential(multiplier=1, min=4, max=10))
def create_SOP_of_workflow(article:str, openai_model_type:str) -> list:
	prompt = read_file("./create_one_level_SOP_prompt.txt")
	if prompt[-1] != '\n':
		prompt += "\n"
	prompt += article
	if prompt[-1] != '\n':
		prompt += "\n"
	prompt += "```\n</Question 4>\n<Answer 4>"

	prompt_start_time = time.time()
	response = openai.ChatCompletion.create(
		model = openai_model_type,
		temperature = 0.3,
		max_tokens = 16384 - count_the_number_of_tokens(prompt, openai_model_type) - 8,
		messages = [{"role": "user", "content": prompt}],
	)
#	print("Time in prompt (small segment) = " + str(time.time() - prompt_start_time) + " secs.")
#	print("Request return.")

	return_of_chatGPT = ""
	for choice in response.choices:
		return_of_chatGPT += choice.message.content

	steps_name = return_of_chatGPT.split("\n")
	steps_name = [x[x.find(":") + 2:] for x in steps_name if x.startswith("Step")]

	return steps_name

def make_diagram(steps:list, openai_model_type:str) -> str:
	prompt = read_file("./diagram_prompt.txt")
	if prompt[-1] != "\n":
		prompt += "\n"
	prompt += "```\n"
	for step_idx, step in enumerate(steps):
		prompt += "Step " + str(step_idx) + ": " + step + "\n"
	prompt += "```\n</Question 4>\n<Answer 4>"
#	print(prompt)
#	exit()

	prompt_start_time = time.time()
	response = openai.ChatCompletion.create(
		model = openai_model_type,
		temperature = 0.3,
		max_tokens = 16384 - count_the_number_of_tokens(prompt, openai_model_type) - 8,
		messages = [{"role": "user", "content": prompt}],
	)
#	print("Time in prompt (small segment) = " + str(time.time() - prompt_start_time) + " secs.")
#	print("Request return.")

	return_of_chatGPT = ""
	for choice in response.choices:
		return_of_chatGPT += choice.message.content
#	print(return_of_chatGPT)

	return return_of_chatGPT

def extract_steps_name(diagram:str) -> list:
	names = []
	diagram_lines = diagram.split("\n")
	for line in diagram_lines:
		if line.find("[") >= 0 and line.find("]") and line.endswith("]") >= 0:
			head = line.find("[")
			tail = line.find("]")
			names.append((line[:head], line[head+1:tail]))
	return names

def extract_steps_relation(diagram:str) -> list:
	relation = []
	diagram_lines = diagram.split("\n")
	for line in diagram_lines:
		if line.find(" --> ") >= 0:
			mid = line.find(" --> ")
			relation.append((line[:mid], line[mid+5:]))
	return relation

def find_idx_from_name(step_id:str, id_name_twin:list) -> int:
#	print(name)
#	print(id_name_twin)
	for idx, id_name in enumerate(id_name_twin):
		if id_name[0] == step_id:
			return idx
	return -1

def prevent_same_name(items:list) -> list:
	output = []
	for item in items:
		if item not in output:
			output.append(item)
		else:
			idx = 1
			while item + "-" + str(idx) in output:
				idx += 1
			output.append(item + "-" + str(idx))
	return output
	
def create_SOP(args):
	transcription = read_file(args.file_path)
	segmented_transcription = segment_workflow(transcription, args)
#	for idx, segment in enumerate(segmented_transcription):
#		print("Segment " + str(idx + 1) + ":\n" + segment)
#	print("\n===================================\n")

	summarization = []
	steps = []
	SOP_progress = tqdm(total=len(segmented_transcription))
	SOP_progress.set_description("SOP creation")
	for segment in segmented_transcription:
		summarization.append(summarize_article(segment, args.gpt_model_type))
		steps.append(create_SOP_of_workflow(segment, args.gpt_model_type))
		SOP_progress.update(1)
		SOP_progress.set_description("SOP creation")

	summarization = prevent_same_name(summarization)
	for step_idx in range(len(steps)):
		steps[step_idx] = prevent_same_name(steps[step_idx])
	
	SOP = ""
	for idx in range(len(summarization)):
#		print("High level step " + str(idx + 1) + ": " + summarization[idx])
		SOP += "High level step " + str(idx + 1) + ": " + summarization[idx] + "\n"
		for substep_idx, substep in enumerate(steps[idx]):
#			print("  Substep " + str(idx + 1) + "-" + str(substep_idx + 1) + ": " + substep)
			SOP += "  Substep " + str(idx + 1) + "-" + str(substep_idx + 1) + ": " + substep + "\n"
	return SOP

def main():
	openai.api_key = "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"
	args = get_args()
	print(create_SOP(args))
	return

if __name__ == "__main__":
	main()
#	segmented_transcription = ["In May, a team of computer scientists showed that a kind of hybrid approach could also work. They published an algorithm that effectively combines the random and deterministic approaches to output a prime number of a specific length, with a high probability of delivering the same one even if the algorithm is run many times. The algorithm connects randomness and complexity in interesting ways, and it might also be useful for cryptography, where some encoding schemes rely on the construction of big primes."]
