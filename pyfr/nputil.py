import ctypes as ct
import functools as ft
import itertools as it
import re
from collections import defaultdict

import numpy as np


def block_diag(arrs):
	shapes = [a.shape for a in arrs]
	out = np.zeros(np.sum(shapes, axis=0), dtype=arrs[0].dtype)

	r, c = 0, 0
	for i, (rr, cc) in enumerate(shapes):
		out[r:r + rr, c:c + cc] = arrs[i]
		r += rr
		c += cc

	return out


def clean(origfn=None, tol=1e-10):
	def cleanfn(fn):
		@ft.wraps(fn)
		def newfn(*args, **kwargs):
			arr = fn(*args, **kwargs).copy()

			# Flush small elements to zero
			arr[np.abs(arr) < tol] = 0

			# Coalesce similar elements
			if arr.size > 1:
				amfl = np.abs(arr.flat)
				amix = np.argsort(amfl)

				i, ix = 0, amix[0]
				for j, jx in enumerate(amix[1:], start=1):
					if not np.isclose(amfl[jx], amfl[ix], rtol=tol,
									  atol=0.1*tol):
						if j - i > 1:
							amfl[amix[i:j]] = np.median(amfl[amix[i:j]])
						i, ix = j, jx

				if i != j:
					amfl[amix[i:]] = np.median(amfl[amix[i:]])

				# Fix up the signs and assign
				arr.flat = np.copysign(amfl, arr.flat)

			return arr
		return newfn

	return cleanfn(origfn) if origfn else cleanfn


_npeval_syms = {
	'__builtins__': {},
	'exp': np.exp, 'log': np.log,
	'sin': np.sin, 'asin': np.arcsin,
	'cos': np.cos, 'acos': np.arccos,
	'tan': np.tan, 'atan': np.arctan, 'atan2': np.arctan2,
	'abs': np.abs, 'pow': np.power, 'sqrt': np.sqrt,
	'tanh': np.tanh, 'pi': np.pi,
	'max': np.maximum, 'min': np.minimum
}


def npeval(expr, locals):
	# Disallow direct exponentiation
	if '^' in expr or '**' in expr:
		raise ValueError('Direct exponentiation is not supported; use pow')

	# Ensure the expression does not contain invalid characters
	if not re.match(r'[A-Za-z0-9_ \t\n\r.,+\-*/%()]+$', expr):
		raise ValueError('Invalid characters in expression')

	# Disallow access to object attributes
	objs = '|'.join(it.chain(_npeval_syms, locals))
	if re.search(rf'({objs}|\))\s*\.', expr):
		raise ValueError('Invalid expression')

	return eval(expr, _npeval_syms, locals)


def fuzzysort(arr, idx, dim=0, tol=1e-6):
	# Extract our dimension and argsort
	arrd = arr[dim]
	srtdidx = sorted(idx, key=arrd.__getitem__)

	if len(srtdidx) > 1:
		i, ix = 0, srtdidx[0]
		for j, jx in enumerate(srtdidx[1:], start=1):
			if arrd[jx] - arrd[ix] >= tol:
				if j - i > 1:
					srtdidx[i:j] = fuzzysort(arr, srtdidx[i:j], dim + 1, tol)
				i, ix = j, jx

		if i != j:
			srtdidx[i:] = fuzzysort(arr, srtdidx[i:], dim + 1, tol)

	return srtdidx


_ctype_map = {
	np.int32: 'int', np.uint32: 'unsigned int',
	np.int64: 'int64_t', np.uint64: 'uint64_t',
	np.float32: 'float', np.float64: 'double'
}


def npdtype_to_ctype(dtype):
	return _ctype_map[np.dtype(dtype).type]


_ctypestype_map = {
	np.int32: ct.c_int32, np.uint32: ct.c_uint32,
	np.int64: ct.c_int64, np.uint64: ct.c_uint64,
	np.float32: ct.c_float, np.float64: ct.c_double
}


def npdtype_to_ctypestype(dtype):
	# Special-case None which otherwise expands to np.float
	if dtype is None:
		return None

	return _ctypestype_map[np.dtype(dtype).type]


# Python3 program to implement greedy 
# algorithm for graph coloring 

def addEdge(adj, v, w):
	
	adj[v].append(w)
	
	# Note: the graph is undirected
	adj[w].append(v) 
	return adj

# Assigns colors (starting from 0) to all
# vertices and prints the assignment of colors
def greedyColoring(adj, V):
	
	result = [-1] * V

	# Assign the first color to first vertex
	result[0] = 0


	# A temporary array to store the available colors. 
	# True value of available[cr] would mean that the
	# color cr is assigned to one of its adjacent vertices
	available = [False] * V

	# Assign colors to remaining V-1 vertices
	for u in range(1, V):
		
		# Process all adjacent vertices and
		# flag their colors as unavailable
		for i in adj[u]:
			if (result[i] != -1):
				available[result[i]] = True

		# Find the first available color
		cr = 0
		while cr < V:
			if (available[cr] == False):
				break
			
			cr += 1
			
		# Assign the found color
		result[u] = cr 

		# Reset the values back to false 
		# for the next iteration
		for i in adj[u]:
			if (result[i] != -1):
				available[result[i]] = False

	# Print the result
	
	fn = defaultdict(list)
	for u in range(V):
		fn[result[u]].append(u)
	
	return fn

# Driver Code;if __name__ == '__main__':
	
#	g1 = [[] for i in range(5)]
#	g1 = addEdge(g1, 0, 1)
#	g1 = addEdge(g1, 0, 2)
#	g1 = addEdge(g1, 1, 2)
#	g1 = addEdge(g1, 1, 3)
#	g1 = addEdge(g1, 2, 3)
#	g1 = addEdge(g1, 3, 4)
#	print("Coloring of graph 1 ")
#	greedyColoring(g1, 5)

#	g2 = [[] for i in range(5)]
#	g2 = addEdge(g2, 0, 1)
#	g2 = addEdge(g2, 0, 2)
#	g2 = addEdge(g2, 1, 2)
#	g2 = addEdge(g2, 1, 4)
#	g2 = addEdge(g2, 2, 4)
#	g2 = addEdge(g2, 4, 3)
#	print("\nColoring of graph 2")
#	greedyColoring(g2, 5)

# This code is contributed by mohit kumar 29
#
