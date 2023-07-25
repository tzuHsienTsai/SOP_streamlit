import tiktoken

if __name__ == "__main__":
	sentence = input("Please input a sentence.")
	tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
	print(len(tokenizer.encode(sentence)))

