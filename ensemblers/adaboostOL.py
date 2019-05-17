from collections import defaultdict
import numpy as np

class AdaBoostOL(object):

    def __init__(self, Learner, classes, M=10):
        self.M = M
        self.learners = [Learner(classes) for _ in range(self.M)]
        self.alpha = np.zeros(self.M)
        self.u = np.ones(self.M)
        self.T = 0

    def update(self, features, label):
        st = 0.0
        self.T += 1
        expert_predictions = np.zeros(self.M)

        for i in range(self.M):
            expert_predictions[i] = self.predict_experts(features, i)

        for i,learner in enumerate(self.learners):
            zt = learner.predict(features) * label
            s_t_prev = st
            st += self.alpha[i]*zt
            eta = 4 / np.sqrt(self.T)
            tmp = self.alpha[i] + (eta * zt)/(1 + np.exp(st))
            self.alpha[i] = max(-2,min(2,tmp))
            w = 1 / (1 + np.exp(s_t_prev))
            learner.partial_fit(features, label, sample_weight=w)
            if label!=expert_predictions[i]:
                self.u[i] = self.u[i] * np.exp(-1)


    def raw_predict(self, features):
        return sum(learner.predict(features) for learner in self.learners)

    def predict_experts(self, features, subset_idx):
        label_weights = defaultdict(int)
        if subset_idx==0:
            label = self.learners[0].predict(features)
            label_weights[label] += self.alpha[0]
        for i in range(subset_idx):
            label = self.learners[i].predict(features)
            label_weights[label] += self.alpha[i]
        return max(label_weights.iterkeys(), key=(lambda key: label_weights[key]))

    def predict(self, features):
        expert_predictions = np.zeros(self.M)

        for i, learner in enumerate(self.learners):
            expert_predictions[i] = self.predict_experts(features, i+1)
        norm = [float(i) / sum(self.u) for i in self.u]
        sampled_idx = np.where(norm == np.random.choice(norm))
        return max(expert_predictions[sampled_idx])


