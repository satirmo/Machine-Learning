import csv
import math
import numpy as np

class LogisticRegression :
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

	def G( self, z ) :
		return 1 / ( 1 + math.exp( - z ) )

	''' Our model is trained by using Batch Gradient Descent. Different heuristics,
		such as Stochastic Gradient Descent, may be used as well. '''
	def TrainModel( self, Theta0 = None, alpha = 0.5, convergenceTolerance = 0.01, maxIterations = 10000 ) :
		Theta = Theta0

		if Theta is None :
			Theta = np.zeros( self.n )

		self.ShuffleTrainingData()

		for iteration in range( maxIterations ) :
			Z = self.X.dot( Theta.transpose() )
			H = np.apply_along_axis( self.G, 0, Z.reshape( 1, self.m ) )

			gradient = np.array( [ ( 1.0 / self.m ) * ( ( self.Y - H ) * ( X ) ).sum( axis = 0 ) for X in self.X.transpose() ] )

			Theta1 = Theta + alpha * gradient

			if self.CheckConvergence( Theta, Theta1, convergenceTolerance ) :
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

		z = XScaled.dot( self.Theta.transpose() )

		result = self.G( z )

		return True if result > 0.5 else False

	def CheckConvergence( self, Theta, Theta1, convergenceTolerance ) :
		return np.linalg.norm( Theta - Theta1 ) < convergenceTolerance 

if __name__ == "__main__" :
	''' Clean training data. If a passenger's age is not listed, the passenger is
		assigned its gender's median age. It is expected that this decision will
		have a noticeable impact on our model. More accurate methods will be
		investigated in the future. '''

	trainingDataFile = open( "titanicSurvivalTrainingData.txt" )
	trainingDataCSV = csv.reader( trainingDataFile )

	trainingDataList = []
	indexes = [ 2, 4, 5, 6, 7, 1 ]

	maleAges = []
	femaleAges = []

	for row in trainingDataCSV :
		if row[ 0 ] == "PassengerId" :
			continue

		if row[ 5 ] == "" :
			continue

		if row[ 4 ] == "male" :
			row[ 4 ] = 1
			maleAges.append( float( row[ 5 ] ) )

		else :
			row[ 4 ] = 0
			femaleAges.append( float( row[ 5 ] ) )

		trainingDataList.append( [ row[ index ] for index in indexes ] )

	maleMedianAge = np.median( np.array( maleAges ) )
	femaleMedianAge = np.median( np.array( femaleAges ) )

	for i in range( len( trainingDataList ) ) :
		if trainingDataList[ i ][ 1 ] == "" :
			trainingDataList[ i ][ 1 ] = maleMedianAge if trainingDataList[ i ][ 4 ] == 1 else femaleMedianAge

		trainingDataList[ i ] = map( float, trainingDataList[ i ] )

	trainingData = np.array( trainingDataList )

	''' Create Logistic Regression Model '''

	logreg = LogisticRegression( trainingData )
	converged = logreg.TrainModel( alpha = 0.9, convergenceTolerance = 0.00001, maxIterations = 10000 )

	if converged == False :
		print( "The model could not be trained." )

	else :
		''' Clean testing data. If a passenger's age is not listed, the passenger is
			assigned its gender's median age. '''

		testingDataFile = open( "titanicSurvivalTestingData.txt" )
		testingDataCSV = csv.reader( testingDataFile )

		testingDataList = []
		indexes = [ 1, 3, 4, 5, 6 ]
		
		for row in testingDataCSV :
			if row[ 0 ] == "PassengerId" :
				continue

			row[ 3 ] = 1 if row[ 3 ] == "male" else 0

			if row[ 4 ] == "" :
				row[ 4 ] = maleMedianAge if row[ 3 ] == 1 else femaleMedianAge

			testingDataList.append( [ float( row[ index ] )  for index in indexes ] )

		testingData = np.array( testingDataList )
		predictions = np.apply_along_axis( logreg.Predict, 1, testingData )

		print( "    Pclass        Sex        Age      SibSp      Parch   Survived" )

		for i in range( len( testingData ) ) :
			row = testingData[ i ]
			prediction = predictions[ i ]

			output = ( row[ 0 ], "male" if row[ 1 ] == 1 else "female", row[ 2 ], row[ 3 ], row[ 4 ], prediction )

			print( "%10d %10s %10.2f %10d %10d %10s" % output )