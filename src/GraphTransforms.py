import sys
import networkx as nx
import numpy as np


class WaveletMomentTransform:
	def __init__(self,
		numScales:int,
		maxMoment:int,
		T:np.ndarray,
		central:bool):

		self.numScales = numScales
		self.maxMoment = maxMoment
		self.T = T 
		self.central = central

		assert T.shape[0] == T.shape[1], "T must be a square matrix"
		
		N = T.shape[0]

		H = (np.eye(N) - T).reshape(1, N, N)
		
		for j in range(1,self.numScales):
			new_wavelet = np.linalg.matrix_power(T,2**(j-1)) - np.linalg.matrix_power(T,2**j)
			H = np.concatenate((H,new_wavelet.reshape(1,N,N)),axis=0)
		self.H = H


	def computeTransform(self,
		X:np.ndarray) -> np.ndarray:
		# Assume X is N x F where N is nodes F is channels
		# H s J X N X N

		new_X = self.H @ X # J X N X F
		exponents = np.arange(1,self.maxMoment)
		print(np.mean(new_X,axis=1).shape)
		X_res= np.array([])

		sys.exit(1)
		for exp in exponents:

			if exp == 1 and self.central:

				coeffs = np.mean(sample_transforms,axis=1)

			elif exp == 2 and self.central:

				coeffs = np.var(sample_transforms,axis =1)

			elif exp > 2 and self.central:

				mu = np.mean(sample_transforms, axis=1, keepdims = True)
				sigma = np.std(sample_transforms, axis=1, keepdims = True)
				coeffs = (sample_transforms - mu)/sigma
				coeffs = np.mean(np.power(coeffs,exp),axis=1)

			else:

				coeffs = np.sum(np.abs(np.power(sample_transforms,exp)),axis=1)
			
			sample_coeffs = np.hstack([sample_coeffs, coeffs]) if sample_coeffs.size else coeffs



			
			

				
			X_res = np.vstack( [X_res, sample_coeffs]) if X_res.size else sample_coeffs
		
		if np.isnan(X_res).any():
			print("self reporting an issue with scale {J} and max moment {p}".format(J=self.J,p=self.p_max))
			sys.exit(1)
		return X_res


# class GFT:
# 	def __init__(
# 		self,
# 		adjacency_matrix:np.ndarray,
# 		normalize: False
# 		) -> None:
	

# 	self.adjacency_matrix = adjacency_matrix
# 	self.normalize = normalize



if __name__ == '__main__':
	G = nx.fast_gnp_random_graph(10,0.3, 1234)
	A = nx.adjacency_matrix(G)
	D = np.diag(np.sum(A,axis =1))
	D_sqrt = np.sqrt(D)
	W = D_sqrt @ A @ D_sqrt
	T = 0.5*(np.eye(A.shape[0])+W)
	transformer = WaveletMomentTransform(5,2,T,True)
	X = np.random.rand(10,9)
	transformer.computeTransform(X)





# class WaveletMomentTransform(BaseEstimator, TransformerMixin):
# 	def __init__(self,
# 		T:np.ndarray,
# 		J:int = 5,
# 		p_max:int = 3,
# 		standardize:bool = True
# 		) -> None:


# 		# assert moment_type in ['raw','central','standard']

# 		self.T = T
# 		self.J = J
# 		self.p_max = p_max
# 		self.standardize = standardize





# 	def transform(self,
# 		X:np.ndarray,
# 		y=None
# 		) -> np.ndarray:

# 		# assumes X is n_samples x n_nodes
# 		# returns a S x J*p_max array
# 		# print(self.H.shape)
# 		# print(X.shape)
# 		# sys.exit(1)
# 		S = X.shape[0]
# 		N = X.shape[1]
		
		
		


# 	def fit(self, X=None,y=None):
# 		assert self.T.shape[0] == self.T.shape[1], "T must be a square matrix"
# 		T = self.T
# 		N = self.T.shape[0]
		
		
# 		return self
		




# zeroTolerance = 1e-9 # Values below this number are considered zero.
# infiniteNumber = 1e12 # infinity equals this number





# def diffusionWavelets(J, T):
#     assert J > 0
#     N = T.shape[0] # Number of nodes
#     assert T.shape[1] == N # Check it's a square matrix
#     I = np.eye(N) # Identity matrix
#     H = (I - T).reshape(1, N, N) # 1 x N x N
#     for j in range(1,J):
#         thisPower = 2 ** (j-1) # 2^(j-1)
#         powerT = np.linalg.matrix_power(T, thisPower) # T^{2^{j-1}}
#         thisH = powerT @ (I - powerT) # T^{2^{j-1}} * (I - T^{2^{j-1}})
#         H = np.concatenate((H, thisH.reshape(1,N,N)), axis = 0)
#     return H

# class GraphScatteringTransform:
#     """
#     graphScatteringTransform: base class for the computation of the graph
#         scattering transform coefficients

#     Initialization:

#     Input:
#         numScales (int): number of wavelet scales (size of the filter bank)
#         numLayers (int): number of layers
#         adjacencyMatrix (np.array): of shape N x N

#     Output:
#         Creates graph scattering transform base handler

#     Methods:

#         Phi = .computeTransform(x): computes the graph scattering coefficients
#             of input x (where x is a np.array of shape B x F x N, with B the
#             batch size, F the number of node features, and N the number of
#             nodes)
#     """

#     # We use this as base class to then specify the wavelet and the self.U
#     # All of them use np.abs() as noinlinearity. I could make this generic
#     # afterward as well, but not for now.

#     def __init__(self, numScales, numLayers, adjacencyMatrix):

#         self.J = numScales
#         self.L = numLayers
#         self.W = adjacencyMatrix.copy()
#         self.N = self.W.shape[0]
#         assert self.W.shape[1] == self.N
#         self.U = None
#         self.H = None

#     def computeTransform(self, x):
#         # Check the dimension of x: batchSize x numberFeatures x numberNodes
#         assert len(x.shape) == 3
#         B = x.shape[0] # batchSize
#         F = x.shape[1] # numberFeatures
#         assert x.shape[2] == self.N
#         # Start creating the output
#         #   Add the dimensions for B and F in low-pass operator U
#         U = self.U.reshape([1, self.N, 1]) # 1 x N x 1
#         #   Compute the first coefficient
#         Phi = x @ U # B x F x 1
#         rhoHx = x.reshape(B, 1, F, self.N) # B x 1 x F x N
#         # Reshape U once again, because from now on, it will have to multiply
#         # J elements (we want it to be 1 x J x N x 1)
#         U = U.reshape(1, 1, self.N, 1) # 1 x 1 x N x 1
#         U = np.tile(U, [1, self.J, 1, 1])
#         # Now, we move to the rest of the layers
#         for l in range(1,self.L): # l = 1,2,...,L
#             nextRhoHx = np.empty([B, 0, F, self.N])
#             for j in range(self.J ** (l-1)): # j = 0,...,l-1
#                 thisX = rhoHx[:,j,:,:] # B x J x F x N
#                 thisHx = thisX.reshape(B, 1, F, self.N) \
#                             @ self.H.reshape(1, self.J, self.N, self.N)
#                     # B x J x F x N
#                 thisRhoHx = np.abs(thisHx) # B x J x F x N
#                 nextRhoHx = np.concatenate((nextRhoHx, thisRhoHx), axis = 1)

#                 phi_j = thisRhoHx @ U # B x J x F x 1
#                 phi_j = phi_j.squeeze(3) # B x J x F
#                 phi_j = phi_j.transpose(0, 2, 1) # B x F x J
#                 Phi = np.concatenate((Phi, phi_j), axis = 2) # Keeps adding the
#                     # coefficients
#             rhoHx = nextRhoHx.copy()

#         return Phi

# class DiffusionScattering(GraphScatteringTransform):
#     """
#     DiffusionScattering: diffusion scattering transform

#     Initialization:

#     Input:
#         numScales (int): number of wavelet scales (size of the filter bank)
#         numLayers (int): number of layers
#         adjacencyMatrix (np.array): of shape N x N

#     Output:
#         Instantiates the class for the diffusion scattering transform

#     Methods:

#         Phi = .computeTransform(x): computes the diffusion scattering
#             coefficients of input x (np.array of shape B x F x N, with B the
#             batch size, F the number of node features, and N the number of
#             nodes)
#     """

#     def __init__(self, numScales, numLayers, adjacencyMatrix):
#         super().__init__(numScales, numLayers, adjacencyMatrix)
#         d = np.sum(self.W, axis = 1)
#         killIndices = np.nonzero(d < zeroTolerance)[0] # Nodes with zero
#             # degree or negative degree (there shouldn't be any since (i) the
#             # graph is connected -all nonzero degrees- and (ii) all edge
#             # weights are supposed to be positive)
#         dReady = d.copy()
#         dReady[killIndices] = 1.
#         # Apply sqrt and invert without fear of getting nans or stuff
#         dSqrtInv = 1./np.sqrt(dReady)
#         # Put back zeros in those numbers that had been failing
#         dSqrtInv[killIndices] = 0.
#         # Inverse diagonal squareroot matrix
#         DsqrtInv = np.diag(dSqrtInv)
#         # Normalized adjacency
#         A = DsqrtInv.dot(self.W.dot(DsqrtInv))
#         # Lazy diffusion
#         self.T = 1/2*(np.eye(self.N) + A)
#         # Low-pass average operator
#         self.U = d/np.linalg.norm(d, 1)
#         # Construct wavelets
#         self.H = diffusionWavelets(self.J, self.T)








