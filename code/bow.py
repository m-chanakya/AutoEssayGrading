'''

Module to extract bag of words and 
generate term-document matrix.

'''

import textmining as txtm
import fileparse as docR


def make_tdm(docs_list):
	textMatrices = []

	for x in xrange(0, 9):
		textMatrix = txtm.TermDocumentMatrix()
		textMatrices.append(textMatrix)

	for doc in docs_list:
		doc_set = int(doc.essay_set)
		textMatrices[doc_set].add_doc(doc.essay)

	return textMatrices


if __name__ == "__main__":
	docs_list = docR.get_list()
	tdMatrices = make_tdm(docs_list)
	for tdm in tdMatrices:
		for row in tdm.rows(cutoff=1):			# Here, cutoff means the number of documents in which this word has to occur for it to be placed in the 'bag of words'.
			print row
