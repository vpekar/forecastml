"""
Based on https://henri.io/posts/using-gradient-boosting-with-early-stopping.html

Usage:

model.fit(x, y, monitor=Monitor(x, y))
"""

import logging
from sklearn.ensemble._gradient_boosting import predict_stage


LOGGER = logging.getLogger("main.early_stopping")


class Monitor():
    """Monitor for early stopping in Gradient Boosting for classification.

    The monitor checks the validation loss between each training stage. When
    too many successive stages have increased the loss, the monitor will return
    true, stopping the training early.

    Parameters
    ----------
    X_valid : array-like, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples
        and n_features is the number of features.
    y_valid : array-like, shape = [n_samples]
        Target values (integers in classification, real numbers in
        regression)
        For classification, labels must correspond to classes.
    max_consecutive_decreases : int, optional (default=5)
        Early stopping criteria: when the number of consecutive iterations that
        result in a worse performance on the validation set exceeds this value,
        the training stops.
    """

    def __init__(self, X_valid, y_valid, max_consecutive_decreases=5):
        self.X_valid = X_valid.copy(order='C')
        self.y_valid = y_valid.copy(order='C')
        self.max_consecutive_decreases = max_consecutive_decreases
        self.losses = []


    def __call__(self, i, clf, args):
        if i == 0:
            self.consecutive_decreases_ = 0
            self.predictions = clf._init_decision_function(self.X_valid)

        predict_stage(clf.estimators_, i, self.X_valid, clf.learning_rate,
                      self.predictions)
        self.losses.append(clf.loss_(self.y_valid, self.predictions))

        if len(self.losses) >= 2 and self.losses[-1] > self.losses[-2]:
            self.consecutive_decreases_ += 1
        else:
            self.consecutive_decreases_ = 0

        if self.consecutive_decreases_ >= self.max_consecutive_decreases:
            LOGGER.debug("Too many consecutive decreases of loss on validation set"
                  "({}): stopping early at iteration {}.".format(
                          self.consecutive_decreases_, i))
            return True
        else:
            return False
