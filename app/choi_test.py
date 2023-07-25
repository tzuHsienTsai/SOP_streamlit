import os
import sys
import openai
import time
from old_segmentation import segmentation

def read_article(file_path:str) -> str:
	article = ""
	with open(file_path, "r") as fp:
		line = fp.readline()
		while line:
			if line.startswith("=====") == False:
				article += line
			line = fp.readline()
	return article

def extract_sentences(article:str) -> list:
	sentences = []
	head_idx = 0
	for article_idx in range(len(article)):
		if article[article_idx] == "."\
			and (article_idx > 0 and article[article_idx - 1] == " ")\
				and (article_idx < len(article) - 1 and article[article_idx + 1] == " "):
			sentences.append(article[head_idx:article_idx + 1].strip())
			head_idx = article_idx + 2
	return sentences

def generate_paragraphs(segments:list) -> list:
	paragraphs = []
	for segment in segments:
		paragraph = ""
		for sentence in segment:
			if len(paragraph) > 0:
				paragraph += " "
			paragraph += sentence
		paragraphs.append(paragraph)
	return paragraphs

def read_trans_file(file_path:str) -> str:
	article = ""
	with open(file_path, "r") as fp:
		line = fp.readline()
		while line:
			article += line
			line = fp.readline()

	return article

def extract_trans_sentences(article:str) -> list:
	sentences = []
	head_idx = 0
	for article_idx in range(len(article)):
		if article[article_idx] == "."\
			and (article_idx < len(article) - 1 and article[article_idx + 1] == " "):
			sentences.append(article[head_idx:article_idx + 1].strip())
			head_idx = article_idx + 2
	return sentences

if __name__ == "__main__":
	file_path = "/Users/caizixian/Desktop/DeepHow/transcription/transBike.txt"
	article = read_trans_file(file_path)
	article = article.replace("\n", " ")
	start_segmentation_time = time.time()
	sentences = extract_trans_sentences(article)
	segments = segmentation({
				'token': '3lPDYZWupFO9tCUU2c5VUTiY4r6ciOvL',
				'workflowId': '123',
				'lang': 'en',
				'input': sentences
				})
	print("Total time segmentation takes: " + str(time.time() - start_segmentation_time) + " secs.")
	exit()
	paragraphs = generate_paragraphs(segments)
	for paragraph in paragraphs:
		print("===========")
		print(paragraph)
'''
	for dir_name_lv1 in os.listdir(dir_path_lv0):
		dir_path_lv1 = dir_path_lv0 + dir_name_lv1 + "/"
		if os.path.isdir(dir_path_lv1) == False:
			continue
		for file_name in os.listdir(dir_path_lv1):
			file_path = dir_path_lv1 + file_name
			article = read_article(file_path)
			article = article.replace("\n", " ")
			sentences = extract_sentences(article)
			segments = segmentation({
						'token': '3lPDYZWupFO9tCUU2c5VUTiY4r6ciOvL',
						'workflowId': '123',
						'lang': 'en',
						'input': sentences
						})
			paragraphs = generate_paragraphs(segments)
			destination_file_path = file_path.replace("choiDataset", "choiResultOld")
			with open(destination_file_path, "w") as fp:
				for paragraph in paragraphs:
					fp.write(paragraph)
					fp.write("\n\n")
'''












