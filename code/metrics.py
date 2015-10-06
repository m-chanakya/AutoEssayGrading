import skll.metrics as met

def get_average_kappa(arr_act, arr_pred):
	"""
	Calculates the average quadratic kappa
	over the entire essay set
	"""

	assert(len(arr_act) == len(arr_pred))
	total = len(arr_act)
	kappa_val = 0

	for i in xrange(0, total):
		kappa_val += met.kappa([arr_act[i]], [arr_pred[i]], \
					'quadratic')
#		print arr_act[i], '-', arr_pred[i]

	kappa_val  = float(kappa_val) / float(total)

	return kappa_val
