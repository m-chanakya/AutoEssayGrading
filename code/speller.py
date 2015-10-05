''' 

Spellchecker module that returns the number of 
misspelled words given an essay.

'''

import nltk
import enchant

Dict = enchant.Dict("en_US")

def num_spell_errors(essay):
	errors = 0
	tokens = nltk.tokenize(essay.essay)
	for each in tokens:
		if not Dict.check(each):
			errors += 1
	return errors
