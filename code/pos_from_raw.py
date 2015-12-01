import sys
import nltk

reload(sys)
sys.setdefaultencoding('utf-8')

if __name__ == "__main__":
	if len(sys.argv)<2:
		print "Usage: python gen.py input_file output_file"
		sys.exit(0)

	ifp = open(sys.argv[1], 'r')
	ofp = open(sys.argv[2], 'w')
	data = ifp.read()
	all_data = nltk.tokenize.sent_tokenize(data)
	PUNCS = ['``', ',', "''", ')', '(', '.', ':', '--']
	for each in all_data:
		data = nltk.word_tokenize(each)
		data = nltk.pos_tag(data)
		data2 = []
		for x in xrange(len(data)):
			if data[x][1] in PUNCS:
				data2.append(data[x][0]+"_PUNC")
			else:
				data2.append(data[x][0]+"_"+data[x][1])
		ofp.write(" ".join(data2)+"\n")
	ofp.close()