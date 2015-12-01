#! /usr/bin/python

import sys


TAG_SET = ['$$START', '$START', '$END', 'NN', 'NST', 'NNP', 'PRP', 'DEM', 'VM', 'VAUX', 'JJ', 'RB', 'PSP', 'RP', 'CC', 'WQ', 'QF', 'QC', 'QO','CL', 'INTF', 'INJ', 'NEG', 'UT', 'SYM', 'XC', 'RDP', 'ECH', 'UNK']
V = set()

transition = {}
emission = {}

bigram = {}
unigram = {}
wgram = {}

def find_emision(sentences):
	global emission
	for sentence in sentences:
		#Emission for first word
		trigram = ('$START', sentence[0][1], sentence[0][0])
		emission[trigram] = emission.get(trigram, 0) + 1
		
		i = 1
		while i < len(sentence):
			trigram = (sentence[i-1][1], sentence[i][1], sentence[i][0])
			emission[trigram] = emission.get(trigram, 0) + 1
			i += 1

def find_transition(sentences):
	global transition
	for sentence in sentences:
		
		#Transition for first word
		trigram = ('$$START', '$START', sentence[0][1])
		transition[trigram] = transition.get(trigram, 0) + 1

		#Transition for second word
		trigram = ('$START', sentence[0][1], sentence[1][1])
		transition[trigram] = transition.get(trigram, 0) + 1
		
		i = 2
		while i < len(sentence):
			trigram = (sentence[i-2][1], sentence[i-1][1], sentence[i][1])
			transition[trigram] = transition.get(trigram, 0) + 1
			i += 1

def ip_laplace(ngram, type):
	k = 0.1
	l1 = 0.1
	l2 = 0.9

	wn_0 = ngram[0]
	wn_1 = ngram[1]
	wn_2 = ngram[2]

	if type == 0:
		x1 = l1*1.0*(transition.get( (wn_0, wn_1, wn_2) , 0) + k)/(bigram.get( (wn_0, wn_1) , 0) + k*len(unigram.keys()) )
		x2 = l2*1.0*(bigram.get( (wn_1, wn_2) , 0) + k)/(unigram.get(wn_1, 0) + k*len(unigram.keys()) )
	else:
		x1 = l1*1.0*(emission.get( (wn_0, wn_1, wn_2) , 0) + k)/(bigram.get( (wn_0, wn_1) , 0) + k*len(V) )
		x2 = l2*1.0*(wgram.get( (wn_1, wn_2) , 0) + k)/(unigram.get(wn_1, 0) + k*len(V))
	return x1 + x2


def viterbi(sentence):
	dp = []
	tag_seq = []

	#INITIALIZE DATA STRUCT
	for i in xrange(len(sentence)):
		dp.append([])
		tag_seq.append([])
		for j in xrange(len(TAG_SET)):
			dp[i].append([])
			tag_seq[i].append([])
			for each in TAG_SET:
				dp[i][j].append(0)
				tag_seq[i][j].append(0)

	#INITIALIZE DP
	for i in xrange(len(TAG_SET)):
		tag_gram = ('$$START', '$START', TAG_SET[i])
		trans = ip_laplace(tag_gram, 0)
		word_gram = ('$START', TAG_SET[i], sentence[0][0])
		emit = ip_laplace(word_gram, 1)
		dp[0][1][i] = trans*emit

	#DP
	for p in xrange(1, len(sentence)):
		for k in xrange(len(TAG_SET)):
			for j in xrange(len(TAG_SET)):
				maximum = -1
				word_gram = (TAG_SET[j], TAG_SET[k], sentence[p][0])
				emit = ip_laplace(word_gram, 1)
				for i in xrange(len(TAG_SET)):
					tag_gram = (TAG_SET[i], TAG_SET[j], TAG_SET[k])
					trans = ip_laplace(tag_gram, 0)
					temp = dp[p-1][i][j] * trans
					if temp >= maximum:
						maximum = temp
						max_pos = i
				dp[p][j][k] = maximum * emit
				tag_seq[p][j][k] = max_pos

	#RECONSTRUCT TAG SEQUENCE
	final_tags = []
	maximum = -1
	for i in xrange(len(TAG_SET)):
		for j in xrange(len(TAG_SET)):
			temp = dp[-1][i][j]
			if temp > maximum:
				maximum = temp
				max_i = i
				max_j = j
	final_tags.append(TAG_SET[max_j])
	final_tags.append(TAG_SET[max_i])
	p = len(sentence) - 1
	while p >= 0:
		next = tag_seq[p][max_i][max_j]
		max_j = max_i
		max_i = next
		final_tags.append(TAG_SET[next])
		p -= 1
	final_tags.reverse()
	return final_tags

def parse(data):
	sentences = data.split('\n')
	sentences = [ ' '.join([each.strip(), '$$_$END']) for each in sentences if each.strip() != '' ]
	sentences = [ [every.split('_') for every in each.split(' ')] for each in sentences]
	return sentences

def preprocess(data):
	global V, bigram, unigram, wgram
	sentences = parse(data)
	for sentence in sentences:
		for word in sentence:
			V.add(word[0])

	for sentence in sentences:
		tag_gram = ('$$START', '$START')
		bigram[tag_gram] = bigram.get(tag_gram, 0) + 1

		tag_gram = ('$START', sentence[0][1])
		bigram[tag_gram] = bigram.get(tag_gram, 0) + 1

		i = 1
		while i < len(sentence):
			tag_gram = (sentence[i-1][1], sentence[i][1])
			bigram[tag_gram] = bigram.get(tag_gram, 0) + 1
			i += 1

	for sentence in sentences:
		unigram['$$START'] = unigram.get('$$START', 0) + 1
		unigram['$START'] = unigram.get('$START', 0) + 1
		for word in sentence:
			unigram[word[1]] = unigram.get(word[1], 0) + 1

	for sentence in sentences:
		for word in sentence:
			word_gram = (word[1], word[0])
			wgram[word_gram] = wgram.get(word_gram, 0) + 1

	return sentences

if __name__ == "__main__":
	if len(sys.argv) != 3:
		'Usage: python hmm.py train.data test.data'
		sys.exit(0)

	f = open(sys.argv[1], 'r')
	data = f.read()
	sentences = preprocess(data)
	
	find_transition(sentences)
	find_emision(sentences)

	f1 = open(sys.argv[2], 'r')
	test_data = f1.read()
	test_sentences = parse(test_data)


	sent_accuracy = 0
	correct_tokens = 0
	total_tokens = 0
	avg = 0
	for sent in test_sentences:
		token_accuracy = 0
		result = viterbi(sent)[2:]
		answer = [word[1] for word in sent]
		if result == answer:
			sent_accuracy += 1
		else:
			pass
			# print "OURS", result
			# print "REAL", answer
			# print
		total_tokens += len(result)
		for i in xrange(len(result)):
			if result[i] == answer[i]:
				correct_tokens += 1
				token_accuracy += 1
		avg += 1.0*token_accuracy/len(sent)

	R1 = 1.0*sent_accuracy/len(test_sentences)
	R2 = 1.0*correct_tokens/total_tokens
	R3 = 1.0*avg/len(test_sentences)

	print "Sentence accuracy : ", 1.0*sent_accuracy/len(test_sentences)
	print "Correct tags/sent accuracy : ", 1.0*avg/len(test_sentences)
	print "Tag accuracy : ", 1.0*correct_tokens/total_tokens
	