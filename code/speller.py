''' 

Spellchecker module that returns the number of 
misspelled words given an essay.

'''

import enchant
import token_gen

Dict = enchant.Dict("en_US")

def num_spell_errors(essay):
	errors = 0
	tokens = token_gen.tokenize([essay.essay])
	for each in tokens:
		if len(each) >= 1:
			try:
				if not Dict.check(each.encode('utf8')):
					errors += 1
			except:
			  	errors += 1
	essay.misspellings = errors
