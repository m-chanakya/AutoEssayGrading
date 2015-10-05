#! /usr/bin/python

import nltk
import sys

def tokenize(data, stops = []):
	if stops == []:
		stops = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']
	endings = ["'t", "'s", "'d", "'m", "'ll", "'ve", "'re"]
	startings = set([x.split("'")[0] for x in stops])
	tokenized = []
	incorrect = []

	for line in data:
		try:
			tokenized.extend(nltk.word_tokenize(line))
		except:
			incorrect.append(line)

	temp = []
	PUNCTUATION = [ch for ch in """(){}[]<>!?.:;,`'"@#$%^&*+-|=~/\\_"""]
	OTHER = ['``']
	NER = ['NUM', 'PERSON', 'ORGANISATION', 'LOCATION', 'MONTH', 'DATE', 'TIME', 'MONEY', 'PERCENT', 'CAPS', 'CITY', 'STATE', 'EMAIL', 'DR']
	PUNCTUATION.extend(OTHER)
	for line in incorrect:
		for word in line.split():
			try:
				temp.extend(nltk.word_tokenize(word))
			except:
				if word[-1] in PUNCTUATION:
					punc = word[-1]
					word = word[:-1]
					temp.append(punc)
				temp.append(word)

	tokenized.extend(temp)

	final_tokens = []
	i=0
	while i < len(tokenized):
		if i < len(tokenized) - 1 and tokenized[i+1] == "n't" or (tokenized[i] in startings and tokenized[i+1] in endings):
			new_token = tokenized[i]+tokenized[i+1]
			final_tokens.append(new_token)
			i += 2
		elif tokenized[i][0] == "'" and (len(tokenized[i]) >= 4 or (len(tokenized[i]) != 1 and tokenized[i].strip("'") not in endings)):
			apostrophe = "'"
			final_tokens.append(apostrophe)
			final_tokens.append(tokenized[i].strip("'"))
			i += 1
		elif tokenized[i][0] == '*':
			temp = tokenized[i].split('*')
			new_token = temp.pop()
			if new_token:
				final_tokens.append(new_token)
			final_tokens.append('*'*len(temp))
			i += 1
		else:
			final_tokens.append(tokenized[i])
			i += 1

	final_tokens = [x for x in final_tokens if x]
	puncs = [x for x in final_tokens if x in PUNCTUATION]
	final_tokens = [x for x in final_tokens if x not in PUNCTUATION]
	for ner in NER:
		final_tokens = [x for x in final_tokens if not x.startswith(ner)]

	return (final_tokens, puncs)

if __name__ == "__main__":
	f = open(sys.argv[1], 'r')
	f1 = open(sys.argv[2], 'r')
	stops = [x.strip('\n') for x in f1.readlines()]
	datas = f.readlines()
	for data in datas:
		tokenize([data], stops)
	f.close()
	f1.close()
