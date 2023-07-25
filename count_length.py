import os
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import argparse
from nltk.tokenize import word_tokenize

def get_args():
	parser = argparse.ArgumentParser(description='Count_the_length_of_segment_in_choiDataset')
	parser.add_argument("--gpt_model_type", default="gpt-3.5-turbo", type=str)
	parser.add_argument("--prevent_long_segment", default="NO", type=str)
	parser.add_argument("--prevent_short_segment", default="NO", type=str)
	return parser.parse_args()

def decide_file_path(args):
	model_type_path = ""
	if args.gpt_model_type.endswith("16k"):
		model_type_path = "16k"
	elif args.gpt_model_type.endswith("4"):
		model_type_path = "gpt4"
	elif args.gpt_model_type.endswith("turbo"):
		model_type_path = "4k"

	prevent_long_segment_path = "NO"
	if args.prevent_long_segment == "YES":
		prevent_long_segment_path = "YES"

	prevent_short_segment_path = "NO"
	if args.prevent_short_segment == "YES":
		prevent_short_segment_path = "YES"

	return "./../choiResult" + model_type_path + prevent_long_segment_path + prevent_short_segment_path + "short" + "/"

def read_file(file_path:str)->str:
#	print(file_path)
	file_content = ""
	with open(file_path) as fp:
		line = fp.readline()
		while line:
			file_content += line
			line = fp.readline()
	return file_content

def count_the_length_of_each_segment(file_path:str) -> list:
	article = read_file(file_path)
	paragraphs = article.split("\n\n")
	return [len(word_tokenize(paragraph)) for paragraph in paragraphs] 
	
#	return [rd.randint(0, 5000 - 1) for _ in range(10)]

def main():
	args = get_args()
	dir_path_lv0 = decide_file_path(args)
#	print(dir_path_lv0)
	
	length_counter = [0] * 10000
#	print(length_counter)

	for file_name_lv1 in os.listdir(dir_path_lv0):
#		print(file_name_lv1)
		dir_path_lv1 = dir_path_lv0 + file_name_lv1 + "/"
		if os.path.isdir(dir_path_lv1) == False:
			continue
#		print(dir_path_lv1)
		
		for file_name_lv2 in os.listdir(dir_path_lv1):
			dir_path_lv2 = dir_path_lv1 + file_name_lv2 + "/"
			if os.path.isdir(dir_path_lv2) == False:
				continue
			for file_name_lv3 in os.listdir(dir_path_lv2):
				file_path = dir_path_lv2 + file_name_lv3
				if file_path.endswith(".ref") == False:
					continue
#				print(file_path)
				len_of_each_segment = count_the_length_of_each_segment(file_path)
				for length in len_of_each_segment:
					length_counter[length] += 1
	
	max_value = 0
	for idx in range(len(length_counter)):
		if length_counter[idx] > 0:
			max_value = idx
	
	x_axis_range = range(max_value + 100)
#	print(max_value)
	y_axis_value = length_counter[:max_value + 100]
	plt.bar(x_axis_range[1:], y_axis_value[1:])
#	plt.ylim(0, 10)
#	plt.xlim(0, 1000)
	plt.xlabel("The length of segments")
	plt.ylabel("The number of semgnets")
	file_name = args.gpt_model_type + args.prevent_long_segment + args.prevent_short_segment + '.png'
	plt.savefig("./../../" + file_name)
#	plt.savefig("test1.png")



	

if __name__ == "__main__":
	main()
