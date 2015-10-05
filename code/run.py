''' 

Main script that runs everything.

'''

import fileparse as docR
import speller

if __name__=="__main__":
	docs_list = docR.get_list()
	for doc in docs_list:
		print speller.num_spell_errors(doc)
