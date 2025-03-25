import numpy as np


class TrainingManager:
    def __init__(self):
        self.best_perf = -np.inf
        self.last_score = -np.inf
        self.not_improved_for = 0
        self.limit = 5000
        self.list_of_scores = []

    def converged(self):
        if self.last_score <= self.best_perf:
            if self.not_improved_for > self.limit:
                return True
        return False

    def update_after_iteration(self, score):
        self.last_score = score
        if score > self.best_perf:
            self.best_perf = score
            self.not_improved_for = 0
        else:
            self.not_improved_for += 1
        self.list_of_scores.append(score)

    def plot_scores(self):
        import matplotlib.pyplot as plt
        plt.plot(self.list_of_scores)
        plt.show()
