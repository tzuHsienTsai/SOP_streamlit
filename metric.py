import nltk
import os
import sys

def slice_article(article:str) -> list:
	paragraphs = []
	pos = article.find("\n\n")
	if pos == -1:
		article = article[11:]
		pos = article.find("==========")
		while pos != -1:
			paragraphs.append(article[:pos])
			article = article[pos + 11:]
			pos = article.find("==========")
	else:
		while pos != -1:
			paragraphs.append(article[:pos])
			article = article[pos + 2:]
			pos = article.find("\n\n")
	return paragraphs

def encode_the_article(article:str) -> list:
	paragraphs = slice_article(article)
	encode_paragraphs = []
	for idx, paragraph in enumerate(paragraphs):
		encode_paragraphs.append([idx] * len(nltk.word_tokenize(paragraph)))	
	encode_article = []
	for code in encode_paragraphs:
		encode_article = encode_article + code
	return encode_article

def read_file(file_path:str) -> str:
	article = ""
	with open(file_path, "r") as fp:
		line = fp.readline()
		while line:
			article += line
			line = fp.readline()
	return article

def pk(truth:list, hypo:list) -> float:
	window_size = int(len(truth) / (2 * (truth[-1] + 1)))
	pk_penalty = 0
	max_size = min(len(hypo), len(truth))
	for index in range(max_size - window_size):
		truth_diff = truth[index] - truth[index + window_size - 1]
		hypo_diff = hypo[index] - hypo[index + window_size - 1]
		if (truth_diff == 0 and hypo_diff != 0) or\
			(truth_diff != 0 and hypo_diff == 0):
			pk_penalty += 1
	return pk_penalty / (max_size - window_size)
	
def windowdiff(truth:list, hypo:list) -> float:
	window_size = int(len(truth) / (2 * (truth[-1] + 1)))
	windowdiff_penalty = 0
	max_size = min(len(truth), len(hypo))
	for index in range(max_size - window_size):
		truth_diff = truth[index + window_size - 1] - truth[index]
		hypo_diff = hypo[index + window_size - 1] - hypo[index]
		if truth_diff - hypo_diff != 0:
			windowdiff_penalty += 1
	return windowdiff_penalty / (max_size - window_size)

def find_interval(array:list):
	ret_interval = []
	for category in range(array[-1] + 1):
		head = 0
		tail = len(array)
		for idx in range(len(array) - 1):
			if array[idx] != category and array[idx + 1] == category:
				head = idx + 1
			if array[idx] == category and array[idx + 1] != category:
				tail = idx + 1
		ret_interval.append((head, tail))
	return ret_interval

def miou(truth:list, hypo:list) -> float:
	miou_score = 0
	truth_interval = find_interval(truth)
#	print(hypo)
	hypo_interval = find_interval(hypo)
	for interval in truth_interval:
		score = 0
		for cand in hypo_interval:
			if cand[1] < interval[0] or cand[0] > interval[1]:
				continue
			max_inter = min(interval[1], cand[1])
			min_inter = max(interval[0], cand[0])
			max_union = max(interval[1], cand[1])
			min_union = min(interval[0], cand[0])
			IOU = (max_inter - min_inter) / (max_union - min_union)
			score = max(score, IOU)
		miou_score += score
	return miou_score / len(truth_interval)

def find_significant(truth_interval, hypo_interval):
	significant = []
	for interval in truth_interval:
		best_idx = 0
		score = 0
		for cand_idx, cand in enumerate(hypo_interval):
			if cand[1] < interval[0] or cand[0] > interval[1]:
				continue
			max_inter = min(interval[1], cand[1])
			min_inter = max(interval[0], cand[0])
			max_union = max(interval[1], cand[1])
			min_union = min(interval[0], cand[0])
			IOU = (max_inter - min_inter) / (max_union - min_union)
			if IOU > score:
				score = IOU
				best_idx = cand_idx
		if best_idx not in significant:
			significant.append(best_idx)
	return significant

def eiou(truth:list, hypo:list) -> float:
	eiou_score = 0
	truth_interval = find_interval(truth)
	hypo_interval = find_interval(hypo)
	significant_truth = find_significant(hypo_interval, truth_interval)
	significant_hypo = find_significant(truth_interval, hypo_interval)
	for interval_idx in significant_hypo:
		interval = hypo_interval[interval_idx]
		score = 0
		for cand_idx, cand in enumerate(truth_interval):
			if cand[1] < interval[0] or cand[0] > interval[1]:
				continue
			max_inter = min(interval[1], cand[1])
			min_inter = max(interval[0], cand[0])
			max_union = max(interval[1], cand[1])
			min_union = min(interval[0], cand[0])
			IOU = (max_inter - min_inter) / (max_union - min_union)
			if IOU > score:
				score = IOU
		eiou_score += score * (interval[1] - interval[0]) / len(hypo)
	
	scale = len(truth_interval) / (len(truth_interval) + len(truth_interval) - len(significant_truth) + len(hypo_interval) - len(significant_hypo))
	return eiou_score * scale

def find_shortest_segment(hypo:str) -> float:
	shortest_len = len(hypo)
	last_tail = 0
	for pos in range(len(hypo)):
		if pos < len(hypo) - 1 and hypo[pos] == '\n' and hypo[pos + 1] == '\n':
			shortest_len = min(shortest_len, len(nltk.tokenize.word_tokenize(hypo[last_tail:pos])))
			last_tail = shortest_len + 2
	return shortest_len

if __name__ == "__main__":
	des_name = ""
	if len(sys.argv) > 1:
		des_name = sys.argv[1]
	pk_score = 0
	windowdiff_score = 0
	miou_score = 0
	eiou_score = 0
	shortest_segment_len = 0
	file_num = 0
	for lv1_dir_name in os.listdir("./../choiDataset/"):
		lv1_dir_path = "./../choiDataset/" + lv1_dir_name
		if os.path.isdir(lv1_dir_path) == False:
			continue
		lv1_dir_path += "/"
		for lv2_dir_name in os.listdir(lv1_dir_path):
			lv2_dir_path = lv1_dir_path + lv2_dir_name
			if os.path.isdir(lv2_dir_path) == False:
				continue
			lv2_dir_path += "/"
			for file_name in os.listdir(lv2_dir_path):
				file_path_true = lv2_dir_path + file_name
				file_path_hypo = file_path_true.replace("choiDataset", "choiResult" + des_name)
#				print(file_path_hypo)
				article_true = read_file(file_path_true)
				article_hypo = read_file(file_path_hypo)
				encode_true = encode_the_article(article_true)
				encode_hypo = encode_the_article(article_hypo)

				pk_score += pk(encode_true, encode_hypo)
				windowdiff_score += windowdiff(encode_true, encode_hypo)
				miou_score += miou(encode_true, encode_hypo)
				eiou_score += eiou(encode_true, encode_hypo)
				shortest_segment_len += find_shortest_segment(article_hypo)
				file_num += 1
	print("pk: " + str(pk_score))
	print("windowdiff: " + str(windowdiff_score))
	print("MIOU: " + str(miou_score))
	print("EIOU: " + str(eiou_score))
	print("Shortest segment: " + str(shortest_segment_len / file_num) + " words.")
	print("The number of files is " + str(file_num))
#				print(file_path_true)
#				print(file_path_hypo)

