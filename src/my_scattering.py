class MyDiffScattering:
	def __init__(self, 
		numScales:int,
		numLayers:int,  
		adjacency_matrix: np.ndarray
		) -> None:
		pass 

	assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1], "adjacency_matrix must be square"
	self.numScales = numScales
	self.numLayers = numLayers
	self.adjacency_matrix = adjacency_matrix

	self.U = None # low pass operator




	def computeTransform(self,X):
		# assume X is N x F
		# where N s number of nodes, F is the features/channels

		assert X.shape[0] == self.adjacency_matrix[0]

		newCoeffs = self.U.T @ X  # should be 1 x F


		
		for l in range(1,self.numLayers):
			for J in range(1,self.numScales):
				pass