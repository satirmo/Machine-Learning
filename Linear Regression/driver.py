import numpy as np
import matplotlib.pyplot as plt
from batch_gradient_descent import BatchGradientDescent
from stochastic_gradient_descent import StochasticGradientDescent
from locally_weighted_linear_regression import LocallyWeightedLinearRegression
from convergence_parameters import ConvergenceParameters

def display_results(queries, results_actual, bgd, sgd, lwr, conv_param, loess_tau):
    for query, result_actual in zip(queries, results_actual):
        result_bgd = bgd.evaluate_query(query)
        result_sgd = sgd.evaluate_query(query)
        result_lwr = lwr.evaluate_query(query, loess_tau, conv_param)

        print("AREA: %d" % (query[0]))
        print("BATCH: %s" % format_result(result_bgd))
        print("STOCH: %s" % format_result(result_sgd))
        print("LOESS: %s" % format_result(result_lwr))
        print("ACTUAL: %.2f\n" % result_actual)

def format_result(result):
    return "%.2f" % result if isinstance(result, float) else "None"

def generate_plot(features, labels, queries, bgd, sgd, lwr, conv_param, loess_tau):
    plt.scatter(features, labels, c='blue')

    plt.ylabel('House Price (Dollars)')
    plt.xlabel('Area of House (Sq. Ft.)')
    plt.title('House Price Per Square Foot')

    x_min = min(features)
    x_max = max(features)
    xs_lin = np.linspace(x_min, x_max, 100)

    if bgd.theta is not None:
        ys_bgd = [bgd.evaluate_query(x_lin) for x_lin in xs_lin]
        plt.plot(xs_lin, ys_bgd, '-g', label='Best-Fit Line: Batch Gradient Descent')

    if sgd.theta is not None:
        ys_sgd = [sgd.evaluate_query(x_lin) for x_lin in xs_lin]
        plt.plot(xs_lin, ys_sgd, '-m', label='Best-Fit Line: Stochastic Gradient Descent')

    xs_lwr = []
    ys_lwr = []

    for query in queries:
        lwr_result = lwr.evaluate_query(query, loess_tau, conv_param)

        if lwr_result is not None:
            xs_lwr.append(query)
            ys_lwr.append(lwr_result)

    if xs_lwr:
        plt.scatter(xs_lwr, ys_lwr, c='red', label='LOESS Prediction')

    plt.legend()
    plt.show()

def import_training_data(training_data_file):
    training_data = np.loadtxt(training_data_file, delimiter=",")

    features = training_data[:, :-2]
    labels = training_data[:, -1:]

    return (features, labels)

def import_queries(queries_file):
    data = np.loadtxt(queries_file, delimiter=",")

    queries = data[:, :-2]
    results = data[:, -1]

    return (queries, results)

TRAINING_DATA_FILE = "PortlandHouseData.csv"
QUERIES_FILE = "Queries.csv"

FEATURES, LABELS = import_training_data(TRAINING_DATA_FILE)
CONV_PARAM = ConvergenceParameters()
LOESS_TAU = 0.4

BGD = BatchGradientDescent(FEATURES, LABELS, CONV_PARAM)
SGD = StochasticGradientDescent(FEATURES, LABELS, CONV_PARAM)
LWR = LocallyWeightedLinearRegression(FEATURES, LABELS)

QUERIES, RESULTS_ACTUAL = import_queries(QUERIES_FILE)

display_results(QUERIES, RESULTS_ACTUAL, BGD, SGD, LWR, CONV_PARAM, LOESS_TAU)
generate_plot(FEATURES, LABELS, QUERIES, BGD, SGD, LWR, CONV_PARAM, LOESS_TAU)
