from math import sqrt
import numpy as np

class TrainingParameters:
    def __init__(self, theta_init=None, alpha=0.5, convergence_tolerance=0.0001,
                 max_iterations=10000):
        self.theta_init = theta_init
        self.alpha = alpha
        self.convergence_tolerance = convergence_tolerance
        self.max_iterations = max_iterations

class BatchGradientDescent:
    def __init__(self, training_data_file, training_parameters):
        self.parse_training_data_file(training_data_file)
        self.scale_training_data()

        self.batch_gradient_descent(training_parameters)

    def parse_training_data_file(self, training_data_file):
        training_data = np.loadtxt(training_data_file, delimiter=",")
        self.m, self.n = training_data.shape

        np.random.shuffle(training_data)

        training_data_features = training_data[:, :self.n-1]
        ones = np.ones((self.m, 1))
        self.X = np.concatenate((ones, training_data_features), axis=1)

        self.Y = training_data[:, self.n-1]

    def scale_training_data(self):
        eps = np.finfo(float).eps
        self.means = np.zeros(self.n)
        self.std_devs = np.ones(self.n)

        for i in range(1, self.n):
            column = self.X[:, i]

            self.means[i] = np.mean(column)

            std_dev = np.std(column)
            self.std_devs[i] = sqrt(std_dev * std_dev + eps)

            column -= self.means[i]
            column /= self.std_devs[i]

    def batch_gradient_descent(self, training_parameters):
        theta = np.zeros(self.n)

        if training_parameters.theta_init is not None:
            theta = training_parameters.theta_init

        alpha = training_parameters.alpha
        convergence_tolerance = training_parameters.convergence_tolerance
        max_iterations = training_parameters.max_iterations

        iteration = 0

        while iteration < max_iterations:
            gradient = self.calculate_gradient(theta)

            theta1 = theta + alpha * gradient

            current_cost = self.calculate_cost(theta)
            next_cost = self.calculate_cost(theta1)

            if abs(current_cost - next_cost) < convergence_tolerance:
                self.theta = theta1.transpose()
                return

            theta = theta1
            iteration += 1

        self.theta = None

    def calculate_gradient(self, theta):
        hyp = self.X.dot(theta)
        gradient = np.array([((self.Y - hyp) * X).sum(axis=0) for X in self.X.transpose()])

        return gradient / self.m

    def calculate_cost(self, theta):
        hyp = self.X.dot(theta)

        return (1.0 / (2.0 * self.m)) * np.square(hyp - self.Y).sum(axis=0)

    def evaluate_queries(self, query_file):
        queries = np.loadtxt(query_file, delimiter=",")

        if self.theta is None:
            return np.array([None] * queries.shape[0])

        return np.array([self.evaluate_query(query) for query in queries])

    def evaluate_query(self, query):
        scaled_query = np.array([1.0] * self.n)

        for i in range(self.n-1):
            scaled_query[i+1] = (query[i] - self.means[i+1]) / self.std_devs[i+1]

        return scaled_query.dot(self.theta)

if __name__ == "__main__":
    TRAINING_DATA_FILE = "PortlandHouseData.csv"
    QUERY_FILE = "Queries.csv"

    TRAINING_PARAMETERS = TrainingParameters()
    BATCH_GRADIENT_DESCENT = BatchGradientDescent(TRAINING_DATA_FILE, TRAINING_PARAMETERS)
    OUTPUT = BATCH_GRADIENT_DESCENT.evaluate_queries(QUERY_FILE)

    print(OUTPUT)
