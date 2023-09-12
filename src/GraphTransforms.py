import sys
import networkx as nx
import numpy as np


zeroTolerance = 1e-9 # Values below this number are considered zero.
infiniteNumber = 1e12 # infinity equals this number


class WaveletMomentTransform:
	def __init__(self,
		numScales:int,
		maxMoment:int,
		adjacency_matrix:np.ndarray,
		central:bool):

		assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1], "adjacency matrix must be square"
		# assert (adjacency_matrix == adjacency_matrix.T).all(), "adjacency_matrix must be symmetric"


		self.numScales = numScales
		self.maxMoment = maxMoment
		self.W = adjacency_matrix.copy()
		self.central = central
		self.H = None

	def computeTransform(self,
		X:np.ndarray) -> np.ndarray:


		
		M =np.argwhere(np.std(X,axis=0)==0)
		
		PsiX= self.H @ X
		
		
		exponents = np.arange(1,self.maxMoment+1)
		

		coeffs = np.array([])

		for exp in exponents:

			if exp == 1 and self.central:

				moment_coeffs = np.mean(PsiX,axis=1)

			elif exp == 2 and self.central:

				moment_coeffs = np.var(PsiX,axis =1)

			elif exp > 2 and self.central:

				mu = np.mean(PsiX, axis=1, keepdims = True)
				sigma = np.std(PsiX, axis=1, keepdims = True)
				# mu = np.mean(PsiX, axis=1)
				# sigma = np.std(PsiX, axis=1)
				sigma[np.where(sigma==0)] = 1

				
				moment_coeffs = (PsiX - mu)/sigma

				if np.isnan(moment_coeffs).any():
				
					idx = np.argwhere(np.isnan(moment_coeffs))
					print('nan issue')


					# print(PsiX.shape)
					# print(sigma.shape)
					# print(mu.shape)
					
								

				moment_coeffs = np.sum(np.power(np.abs(moment_coeffs),exp),axis=1)

			else:

				moment_coeffs = np.sum(np.power(np.abs(PsiX),exp),axis=1)
				
			coeffs = np.hstack([coeffs, moment_coeffs]) if coeffs.size else moment_coeffs

		return coeffs.flatten()


class DiffusionWMT(WaveletMomentTransform):
	def __init__(self,
		numScales:int,
		maxMoment:int,
		adjacency_matrix:np.ndarray,
		central:bool):


		super().__init__(numScales, maxMoment, adjacency_matrix,central)

		N = adjacency_matrix.shape[0]
		D_invsqrt = np.diag(1/np.sqrt(np.sum(self.W,axis =1)))
		# D_sqrt = np.sqrt(D)
		A = D_invsqrt @ self.W @ D_invsqrt

		T = 0.5*(np.eye(A.shape[0]) + A)
		

		H = (np.eye(N) - T).reshape(1, N, N)
		
 
		for j in range(1,numScales):
			new_wavelet = np.linalg.matrix_power(T,2**(j-1)) - np.linalg.matrix_power(T,2**j)
			H = np.concatenate((H,new_wavelet.reshape(1,N,N)),axis=0)
		self.H = H














