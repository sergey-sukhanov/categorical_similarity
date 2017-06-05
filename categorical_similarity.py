import numpy as np
import time

class CategoricalSimilarity():

    def __init__(self, data):
        self.data = data
        self.n = data.shape[0]
        self.d = data.shape[1]

    def overlap(self):
        score = np.zeros((self.n, self.n))
        for i, X in enumerate(self.data):
            score[i] = 1. * np.sum(data==X, axis=1) / self.d
        return score

    def eskin(self):
        score = np.zeros((self.n, self.n))
        n_k = [len(np.unique(data[:, x])) for x in xrange(data.shape[1])]
        for i, X in enumerate(self.data):
            tmp_bool = data==X
            tmp_x = np.zeros(self.data.shape)
            for k in xrange(self.d):
                tmp_x[:, k][tmp_bool[:, k] == True] = 1.0 / self.d
                tmp_x[:, k][tmp_bool[:,k] == False] = 1.0 * n_k[k]**2 / (n_k[k]**2 + 2) / self.d
            score[i] = np.sum(tmp_x, axis=1)
        return score

    def iof(self):
        score = np.zeros((self.n, self.n))
        f_k_d = [dict(zip(np.unique(data[:, x], return_counts=True)[0], np.unique(data[:, x], return_counts=True)[1])) for x in xrange(self.d)]
        for i, X in enumerate(self.data):
            tmp_x = np.zeros(self.data.shape)
            for k in xrange(self.d):
                f_k_x = f_k_d[k].get(X[k], 0)
                for val, f_k_y in f_k_d[k].iteritems():
                    if val == X[k]:
                        tmp_x[:, k][data[:, k] == val] = 1. / self.d
                    else:
                        tmp_x[:, k][data[:, k] == val] = 1. / (1 + np.log(f_k_x) * np.log(f_k_y)) / self.d
            score[i] = np.sum(tmp_x, axis=1)
        return score

    def of(self):
        score = np.zeros((self.n, self.n))
        f_k_d = [dict(zip(np.unique(data[:, x], return_counts=True)[0], np.unique(data[:, x], return_counts=True)[1])) for x in xrange(self.d)]
        for i, X in enumerate(self.data):
            tmp_x = np.zeros(self.data.shape)
            for k in xrange(self.d):
                f_k_x = f_k_d[k].get(X[k], 0)
                for val, f_k_y in f_k_d[k].iteritems():
                    if val == X[k]:
                        tmp_x[:, k][data[:, k] == val] = 1. / self.d
                    else:
                        tmp_x[:, k][data[:, k] == val] = 1. / (1 + np.log(1.*self.n/f_k_x) * np.log(1.*self.n/f_k_y)) / self.d
            score[i] = np.sum(tmp_x, axis=1)
        return score

    # def lin(self):
    #     score = np.zeros((self.n, self.n))
    #     p_hat_k_d = [dict(zip(np.unique(data[:, x], return_counts=True)[0], 1.*np.unique(data[:, x], return_counts=True)[1]/self.n)) for x in xrange(self.d)]
    #     for i, X in enumerate(self.data):
    #         tmp_x = np.zeros(self.data.shape)
    #         for k in xrange(self.d):
    #             p_hat_k_x = p_hat_k_d[k].get(X[k], 0)
    #             for val, p_hat_k_y in p_hat_k_d[k].iteritems():
    #                 if val == X[k]:
    #                     tmp_x[:, k][data[:, k] == val] = 2. * np.log(p_hat_k_x) #TODO: normalize
    #                 else:
    #                     tmp_x[:, k][data[:, k] == val] = 2. * np.log(p_hat_k_x + p_hat_k_y) #TODO: normalize
    #         score[i] = np.sum(tmp_x, axis=1)
    #     return score

    def goodall3(self):
        score = np.zeros((self.n, self.n))
        p_2_k_d = [dict(zip(np.unique(data[:, x], return_counts=True)[0], 1.*np.unique(data[:, x], return_counts=True)[1]*(np.unique(data[:, x], return_counts=True)[1]-1)/self.n/(self.n-1))) for x in xrange(self.d)]
        for i, X in enumerate(self.data):
            tmp_bool = data==X
            tmp_x = np.zeros(self.data.shape)
            for k in xrange(self.d):
                tmp_x[:, k][tmp_bool[:, k] == True] = 1. * (1 - p_2_k_d[k].get(X[k], 0)) / self.d
            score[i] = np.sum(tmp_x, axis=1)
        return score

    def goodall4(self):
        score = np.zeros((self.n, self.n))
        p_2_k_d = [dict(zip(np.unique(data[:, x], return_counts=True)[0], 1.*np.unique(data[:, x], return_counts=True)[1]*(np.unique(data[:, x], return_counts=True)[1]-1)/self.n/(self.n-1))) for x in xrange(self.d)]
        for i, X in enumerate(self.data):
            tmp_bool = data==X
            tmp_x = np.zeros(self.data.shape)
            for k in xrange(self.d):
                tmp_x[:, k][tmp_bool[:, k] == True] = 1. * p_2_k_d[k].get(X[k], 0) / self.d
            score[i] = np.sum(tmp_x, axis=1)
        return score

if __name__ == '__main__':
    # Test Data
    data = np.array([
            [1, 1, 3],
            [1, 1, 3],
            [3, 2, 1],
            [3, 2, 3]
            ])

    sim = CategoricalSimilarity(data)
    print(sim.overlap())
    print(sim.eskin())
    print(sim.iof())
    print(sim.of())
    print(sim.goodall3())
    print(sim.goodall4())

