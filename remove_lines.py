import os

def remove_newlines(file_path):
	with open(file_path, 'r') as file:
		lines = ""
		line = file.readline()
		while line:
			lines += line
			line = file.readline()

	# Remove '\n' characters from each line
	lines = lines.replace('\n', '')

	with open(file_path, 'w') as file:
		file.write(lines)

def add_space(file_path):
	with open(file_path, 'r') as file:
		content = file.read()

	modified_content = ""
	for i, char in enumerate(content):
		modified_content += char
		if char in [',', '.'] and i < len(content) - 1 and content[i + 1] != ' ':
			modified_content += ' '

	with open(file_path, 'w') as file:
		file.write(modified_content)

if __name__ == "__main__":
	for file in os.listdir("./../transcription/"):
		remove_newlines("./../transcription/" + file)
		add_space("./../transcription/" + file)
		
