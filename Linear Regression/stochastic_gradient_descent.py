from math import sqrt
import numpy as np

class StochasticGradientDescent:
    def __init__(self, features, labels, convergence_parameters):
        ones = np.ones((features.shape[0], 1))
        data = np.concatenate((ones, features, labels), axis=1)
        np.random.shuffle(data)

        self.X = data[:, :-1]
        self.Y = data[:, -1]

        self.r, self.c = self.X.shape

        self.scale_training_data()
        self.stochastic_gradient_descent(convergence_parameters)

    def scale_training_data(self):
        eps = np.finfo(float).eps
        self.means = np.zeros(self.c)
        self.std_devs = np.ones(self.c)

        for i in range(1, self.c):
            column = self.X[:, i]

            self.means[i] = np.mean(column)

            std_dev = np.std(column)
            self.std_devs[i] = sqrt(std_dev * std_dev + eps)

            column -= self.means[i]
            column /= self.std_devs[i]

    def stochastic_gradient_descent(self, convergence_parameters):
        theta = np.zeros(self.c)

        if convergence_parameters.theta_init is not None:
            theta = convergence_parameters.theta_init

        alpha = convergence_parameters.alpha
        conv_tol = convergence_parameters.conv_tol

        iteration = 1
        max_iterations = convergence_parameters.max_iterations

        indexes = np.array([i for i in range(self.r)])

        while iteration <= max_iterations:
            np.random.shuffle(indexes)

            theta1 = theta

            for index in indexes:
                gradient = self.calculate_gradient(theta1, self.X[index], self.Y[index])
                theta1 = theta1 + alpha * gradient

            current_cost = self.calculate_cost(theta)
            next_cost = self.calculate_cost(theta1)

            if abs(current_cost - next_cost) < conv_tol:
                self.theta = theta1
                return

            theta = theta1
            alpha *= 0.99

        self.theta = None

    def calculate_gradient(self, theta, x_i, y_i):
        hyp = x_i.dot(theta)

        return (y_i - hyp) / self.r * x_i

    def calculate_cost(self, theta):
        hyp = self.X.dot(theta)

        return (1.0 / (2.0 * self.r)) * np.square(hyp - self.Y).sum(axis=0)

    def evaluate_queries(self, queries):
        if self.theta is None:
            return np.array([None] * len(queries))

        return np.array([self.evaluate_query(query) for query in queries])

    def evaluate_query(self, query):
        if self.theta is None:
            return None

        scaled_query = np.array([1.0] * self.c)

        for i in range(self.c-1):
            scaled_query[i+1] = (query[i] - self.means[i+1]) / self.std_devs[i+1]

        return scaled_query.dot(self.theta)
