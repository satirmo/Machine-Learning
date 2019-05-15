class ConvergenceParameters:
    def __init__(self, theta_init=None, alpha=0.5, conv_tol=0.0001, max_iterations=10000):
        self.theta_init = theta_init
        self.alpha = alpha
        self.conv_tol = conv_tol
        self.max_iterations = max_iterations
