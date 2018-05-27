import math
import numpy as np

class LocallyWeightedLinearRegression :
	def __init__( self, trainingData ) :
		self.m, self.n = trainingData.shape

		ones = np.ones( ( self.m, 1 ) )
		trainingData = np.concatenate( ( ones, trainingData ), axis = 1 )

		self.X = trainingData[ :, : self.n ]
		self.Y = trainingData[ :, self.n ]

		self.ScaleTrainingData()

	def ShuffleTrainingData( self ) :
		trainingData = np.concatenate( ( self.X, self.Y.reshape( ( self.m, 1 ) ) ), axis = 1 )
		np.random.shuffle( trainingData )

		self.X = trainingData[ :, : self.n ]
		self.Y = trainingData[ :, self.n ]

	def ScaleTrainingData( self ) :
		self.means = np.zeros( self.n )
		self.stdDevs = np.ones( self.n )

		for i in range( 1, self.n ) :
			column = self.X[ :, i ]

			mean = np.mean( column )
			stdDev = np.std( column )
			
			column -= mean
			column /= stdDev

			self.means[ i ] = mean
			self.stdDevs[ i ] = stdDev

	def CalculateWeights( self, x, tau ) :
		self.W = np.array( [ math.exp( - ( x - X ).dot( ( x - X ).T ) / ( 2 * tau ** 2 ) ) for X in self.X ] )

	''' This function uses Batch Gradient Descent to minimize the cost function. Different
		heuristics, such as Stochastic Gradient Descent, may be used as well. '''
	def MinimizeCostFunction( self, convergenceParameters ) :
		Theta = np.zeros( self.n )
		alpha = 0.5
		convergenceTolerance = 0.01
		maxIterations = 10000

		if "Theta" in convergenceParameters :
			Theta = convergenceParameters[ "Theta" ]

		if "alpha" in convergenceParameters :
			alpha = convergenceParameters[ "alpha" ]

		if "convergenceTolerance" in convergenceParameters :
			convergenceTolerance = convergenceParameters[ "convergenceTolerance" ]

		if "maxIterations" in convergenceParameters :
			maxIterations = convergenceParameters[ "maxIterations" ]

		self.ShuffleTrainingData()

		for iteration in range( maxIterations ) :
			H = self.X.dot( Theta.T )

			gradient = np.array( [ ( 1.0 / self.m ) * ( self.W * ( self.Y - H ) * ( X ) ).sum( axis = 0 ) for X in self.X.T ] )
			
			Theta1 = Theta + alpha * gradient

			if abs( self.Cost( Theta ) - self.Cost( Theta1 ) ) < convergenceTolerance :
				self.Theta = Theta1
				
				return True

			Theta = Theta1

		return False

	def ScaleQuery( self, X ) :
		for i in range( 1, self.n ) :
			X[ i ] = ( X[ i ] - self.means[ i ] ) / self.stdDevs[ i ]

		return X

	def Predict( self, X, convergenceParameters, tau = 1 ) :
		ones = np.ones( 1 )
		X = np.concatenate( ( ones, X ), axis = 0 )
		XScaled = self.ScaleQuery( X )

		self.CalculateWeights( XScaled, tau )
		converged = self.MinimizeCostFunction( convergenceParameters )

		if converged == False :
			return False

		return XScaled.dot( self.Theta.T )

	def Cost( self, Theta ) :
		H = self.X.dot( Theta.T )

		return ( 1.0 / ( 2.0 * self.m ) ) * np.square( H - self.Y ).sum( axis = 0 )

if __name__ == "__main__" :
	trainingData = np.loadtxt( "portlandHouseData.txt", delimiter = "," )
	lwr = LocallyWeightedLinearRegression( trainingData )

	convergenceParameters = {
		"alpha" : 0.01,
		"convergenceTolerance" : 100,
		"maxIterations" : 10000
	}

	queries = np.loadtxt( "queries.txt", delimiter = "," )
	
	for query in queries :
		result = lwr.Predict( query, convergenceParameters, tau = 5.0 )

		if result == False :
			print( "The price of a house with an area of %.0f square feet and %.0f rooms could not be predicted." % ( query[ 0 ], query[ 1 ] ) )

		else :
			print( "A house with an area of %.0f square feet and %.0f rooms will cost approximately $%.2f." % ( query[ 0 ], query[ 1 ], result ) )