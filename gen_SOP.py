import openai
from workflow_segmentation_short_response import segment_workflow
import argparse
import re
import sys
import time
import streamlit as st
from tqdm import tqdm
from stqdm import stqdm
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
	
def create_SOP(transcription, args):
	segmentation_start_time = time.time()
	segmented_transcription = segment_workflow(transcription, args)
	segmentation_take_time = time.time() - segmentation_start_time
#	for idx, segment in enumerate(segmented_transcription):
#		print("Segment " + str(idx + 1) + ":\n" + segment)
#	print("\n===================================\n")

	summarization = []
	steps = []
	SOP_progress = tqdm(total=len(segmented_transcription))
	SOP_progress_streamlit = stqdm(total=len(segmented_transcription))
	SOP_progress.set_description("SOP creation")
	SOP_progress_streamlit.set_description("SOP creation")
	for segment in segmented_transcription:
		summarization.append(summarize_article(segment, args.gpt_model_type))
		steps.append(create_SOP_of_workflow(segment, args.gpt_model_type))
		SOP_progress.update(1)
		SOP_progress.set_description("SOP creation")
		SOP_progress_streamlit.update(1)
		SOP_progress_streamlit.set_description("SOP creation")

	SOP_list = []
	for idx in range(len(summarization)):
		SOP = ""
#		print("High level step " + str(idx + 1) + ": " + summarization[idx])
		SOP += "High level step " + str(idx + 1) + ": " + summarization[idx] + "\n"
		for substep_idx, substep in enumerate(steps[idx]):
#			print("  Substep " + str(idx + 1) + "-" + str(substep_idx + 1) + ": " + substep)
			SOP += "  Substep " + str(idx + 1) + "-" + str(substep_idx + 1) + ": " + substep + "\n"
		SOP_list.append(SOP)
	st.write("Workflow segmentation execution time: " + str(segmentation_take_time))
	return segmented_transcription, SOP_list

def trans_preprocessing(transcription:str) -> str:
	new_transcription = transcription.replace("\n", " ")
	new_transcription = transcription.replace("  ", " ")
	return new_transcription


class arguments:
	def __init__(self, gpt_model_type, prevent_long_segment, prevent_short_segment):
		self.gpt_model_type = gpt_model_type
		self.prevent_long_segment = prevent_long_segment
		self.prevent_short_segment = prevent_short_segment


def streamlit_app():
	openai.api_key = "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"
	st.title("SOP creation Demo Site (English and Mandarin Version)")
	st.header("Transcription")
	transcription = st.text_area("Input:")
	transcription = trans_preprocessing(transcription)
#	st.write(transcription)
	start_running = st.button("Run")
	if start_running:
		args = arguments("gpt-3.5-turbo-16k", "YES", "YES")
		start_time = time.time()
		segments, SOPs = create_SOP(transcription, args)
		st.write("Overall execution time: " + str(time.time() - start_time) + " seconds.")
		st.divider()
		for idx in range(len(segments)):
			segment_col, step_col = st.columns([1, 1])
			with segment_col:
				st.write(segments[idx])
			with step_col:
				steps = SOPs[idx].split("\n")
				for step in steps:
					if step.startswith("H"):
						st.subheader(step)
					else:
						st.write(step)
			st.divider()
#		st.write(args.gpt_model_type)
#		st.write(args.prevent_long_segment)
#		st.write(args.prevent_short_segment)


def main():
	openai.api_key = "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"
	args = get_args()
	print(create_SOP(args))
	return

if __name__ == "__main__":
	streamlit_app()
#	main()
#	segmented_transcription = ["In May, a team of computer scientists showed that a kind of hybrid approach could also work. They published an algorithm that effectively combines the random and deterministic approaches to output a prime number of a specific length, with a high probability of delivering the same one even if the algorithm is run many times. The algorithm connects randomness and complexity in interesting ways, and it might also be useful for cryptography, where some encoding schemes rely on the construction of big primes."]
