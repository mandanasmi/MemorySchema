import numpy as np
import math
from scipy.stats import beta

from scipy.special import gammaln

from dag_gflownet.scores.base import BasePrior



class TemporalPrior(BasePrior):
    def __init__(self, num_variables=None, A=None, hit_counts=None, d=1, c=0.3):
        super().__init__(num_variables)
        self.A = A
        self.hit_counts = hit_counts
        self.d = d
        self.c = c
        self._log_prior = self.calculate_log_prior()

    @jit
    def calculate_log_prior(self):
        n = self.A.shape[0]
        # Compute Q matrix as per the new definition
        Q = self.compute_Q_matrix(self.hit_counts, self.d)
        
        # Calculate log prior using the Q matrix
        sum_present = np.sum(self.A * (np.log(Q) + np.log(self.c)))
        sum_absent = np.sum((1 - self.A) * np.log(1 - self.c))
        
        return sum_present + sum_absent

    def compute_Q_matrix(self, hit_counts, d):
        n = hit_counts.shape[0]
        Q = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    hits_for_i_before_j = np.sum(hit_counts[i, :j])
                    start_j = j + d
                    end_j = hit_counts.shape[1]
                    start_i = i + 2 * d
                    end_i = hit_counts.shape[1]
                    
                    hits_for_j = np.sum(hit_counts[j, start_j:end_j]) if start_j < end_j else 0
                    hits_for_i = np.sum(hit_counts[i, start_i:end_i]) if start_i < end_i else 0
                    
                    hits_for_j_and_i_plus_2d = hits_for_j + hits_for_i
                    
                    if hits_for_j_and_i_plus_2d > 0:
                        Q = Q.at[i, j].set(hits_for_i_before_j / hits_for_j_and_i_plus_2d)
        return Q

    @property
    def log_prior(self):
        return self._log_prior

class UniformPrior(BasePrior):
    @property
    def log_prior(self):
        if self._log_prior is None:
            self._log_prior = np.zeros((self.num_variables,))
        return self._log_prior


class ErdosRenyiPrior(BasePrior):
    def __init__(self, num_variables=None, num_edges_per_node=1.):
        super().__init__(num_variables)
        self.num_edges_per_node = num_edges_per_node

    @property
    def log_prior(self):
        if self._log_prior is None:
            num_edges = self.num_variables * self.num_edges_per_node  # Default value
            p = num_edges / ((self.num_variables * (self.num_variables - 1)) // 2)
            all_parents = np.arange(self.num_variables)
            self._log_prior = (all_parents * math.log(p)
                + (self.num_variables - all_parents - 1) * math.log1p(-p))
        return self._log_prior


class EdgePrior(BasePrior):
    def __init__(self, num_variables=None, beta=1.):
        super().__init__(num_variables)
        self.beta = beta

    @property
    def log_prior(self):
        if self._log_prior is None:
            self._log_prior = np.arange(self.num_variables) * math.log(self.beta)
        return self._log_prior


class FairPrior(BasePrior):
    @property
    def log_prior(self):
        if self._log_prior is None:
            all_parents = np.arange(self.num_variables)
            self._log_prior = (
                - gammaln(self.num_variables + 1)
                + gammaln(self.num_variables - all_parents + 1)
                + gammaln(all_parents + 1)
            )
        return self._log_prior
    
class BetaPrior(BasePrior):
    def __init__(self, num_variables=None, alpha=1., beta=1.):
        super().__init__(num_variables)
        self.alpha = alpha
        self.beta = beta

    @property
    def log_prior(self): 
        self.log_prior = beta.logpdf(value, self.alpha, self.beta)
        
        return self._log_prior

