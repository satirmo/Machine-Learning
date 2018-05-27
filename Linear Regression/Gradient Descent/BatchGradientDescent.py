import numpy as np

class BatchGradientDescent :
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

	def MinimizeCostFunction( self, Theta0 = None, alpha = 0.5, convergenceTolerance = 0.01, maxIterations = 10000 ) :
		Theta = Theta0

		if Theta is None :
			Theta = np.zeros( self.n )

		self.ShuffleTrainingData()

		for iteration in range( maxIterations ) :
			H = self.X.dot( Theta.transpose() )
			gradient = np.array( [ ( 1.0 / self.m ) * ( ( self.Y - H ) * ( X ) ).sum( axis = 0 ) for X in self.X.transpose() ] )
			
			
			
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

	def Predict( self, X ) :
		ones = np.ones( 1 )
		X = np.concatenate( ( ones, X ), axis = 0 )
		XScaled = self.ScaleQuery( X )

		return XScaled.dot( self.Theta.transpose() )

	def Cost( self, Theta ) :
		H = self.X.dot( Theta.transpose() )

		return ( 1.0 / ( 2.0 * self.m ) ) * np.square( H - self.Y ).sum( axis = 0 )

if __name__ == "__main__" :
	trainingData = np.loadtxt( "portlandHouseData.txt", delimiter = "," )
	bgd = BatchGradientDescent( trainingData )
	converged = bgd.MinimizeCostFunction( alpha = 0.25, convergenceTolerance = 100, maxIterations = 10000 )

	if converged :
		queries = np.loadtxt( "queries.txt", delimiter = "," )
		predictions = np.apply_along_axis( bgd.Predict, 1, queries )

		for i in range( predictions.size ) :
			print( "A house with an area of %.0f square feet and %.0f rooms will cost approximately $%.2f." % ( queries[ i, 0 ], queries[ i, 1 ], predictions[ i ] ) )

	else :
		print( "Theta did not converge." )
