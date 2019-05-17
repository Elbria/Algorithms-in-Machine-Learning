from collections import defaultdict
import math

def binomial(n, r):
    ''' Binomial coefficient, nCr, aka the "choose" function
        n! / (r! * (n - r)!)
    '''
    p = 1
    for i in range(1, min(r, n - r) + 1):
        p *= n
        p //= i
        n -= 1
    return p

class OnlineBBM(object):

    def __init__(self, Learner, classes, M=10, g=0.10):
        self.M = M
        self.learners = [Learner(classes) for _ in range(self.M)]
        self.gamma = g
        self.w_history = [[] for i in range(self.M)]

    def update(self, features, label):
        st = 0.0
        N = self.M
        for i,learner in enumerate(self.learners):
            i += 1
            kt = int(math.floor((N-i-st+1)/2))
            st += learner.predict(features) * label
            wt = binomial(N-i,kt)*(0.5 + 0.5*self.gamma)**kt * (0.5-0.5 * self.gamma)**(N-i-kt)
            self.w_history[i-1].append(wt)
            w = wt / max(self.w_history[i-1])
            learner.partial_fit(features, label, sample_weight=w)

    def raw_predict(self, features):
        return sum(learner.predict(features) for learner in self.learners)

    def predict(self, features):
        label_weights = defaultdict(int)
        for i in range(self.M):
            label = self.learners[i].predict(features)
            label_weights[label] += 1

        return max(label_weights.iterkeys(), key=(lambda key: label_weights[key]))


