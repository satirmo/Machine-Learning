from math import exp, sqrt
import numpy as np

class LocallyWeightedLinearRegression:
    def __init__(self, features, labels):
        ones = np.ones((features.shape[0], 1))
        data = np.concatenate((ones, features, labels), axis=1)
        np.random.shuffle(data)

        self.X = data[:, :-1]
        self.Y = data[:, -1]

        self.r, self.c = self.X.shape

        self.scale_training_data()

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

    def batch_gradient_descent(self, convergence_parameters, weights):
        theta = np.zeros(self.c)

        if convergence_parameters.theta_init is not None:
            theta = convergence_parameters.theta_init

        alpha = convergence_parameters.alpha
        conv_tol = convergence_parameters.conv_tol

        iteration = 1
        max_iterations = convergence_parameters.max_iterations

        while iteration <= max_iterations:
            gradient = self.calculate_gradient(theta, weights)

            theta1 = theta + alpha * gradient

            current_cost = self.calculate_cost(theta)
            next_cost = self.calculate_cost(theta1)

            if abs(current_cost - next_cost) < conv_tol:
                return theta

            theta = theta1
            iteration += 1

        return None

    def calculate_gradient(self, theta, weights):
        hyp = self.X.dot(theta)
        gradient = np.array([(weights * (self.Y - hyp) * X).sum(axis=0)
                             for X in self.X.transpose()])

        return gradient / self.r

    def calculate_cost(self, theta):
        hyp = self.X.dot(theta)

        return (1.0 / (2.0 * self.r)) * np.square(hyp - self.Y).sum(axis=0)

    def evaluate_queries(self, queries, tau, convergence_parameters):
        return np.array([self.evaluate_query(query, convergence_parameters, tau)
                         for query in queries])

    def evaluate_query(self, query, tau, convergence_parameters):
        scaled_query = np.array([1.0] * self.c)

        for i in range(self.c-1):
            scaled_query[i+1] = (query[i] - self.means[i+1]) / self.std_devs[i+1]

        weights = self.calculate_weights(scaled_query, tau)
        theta = self.batch_gradient_descent(convergence_parameters, weights)

        if theta is None:
            return None

        return scaled_query.dot(theta)

    def calculate_weights(self, query, tau):
        calculate_weight = lambda X: exp(-(query - X).dot(query - X) / (2 * tau ** 2))
        return np.array([calculate_weight(x_i) for x_i in self.X])
