import numpy as np
import pickle
from wcmdp import SingleArmAnalyzer
from matplotlib import pyplot as plt
import warnings
import os

# todo: save the object using its attributes rather than the whole class

class RandomExampleFullyHet(object):
    def __init__(self, sspa_size, aspa_size, max_N, K, distr, parameters):
        """
        Generate a random wcmdp; the statistics of each arm is independently generated
        :param sspa_size: size of the state space
        :param aspa_size: size of the action space
        :param max_N: maximum number of arms in any instantialization of this wcmdp example
        :param K: number of constraints
        :param distr: string, so far only "dirichlet" is implemented; the distribution for generating each kernel
        :param parameters: parameters of the distribution
        """
        self.typed_het = False
        self.sspa_size = sspa_size
        self.aspa_size = aspa_size
        self.max_N = max_N
        self.K = K
        self.distr = distr
        self.parameters = parameters
        if distr == "dirichlet":
            if len(parameters) == 1:
                parameters = [parameters[0]] * self.sspa_size
            else:
                parameters = parameters
            self.trans_tensor = np.random.dirichlet(parameters, (max_N, sspa_size, aspa_size))
            self.reward_tensor = np.random.uniform(0, 1, size=(max_N, sspa_size, aspa_size))
            self.reward_tensor[:,:,0] = 0 # action 0 generates zero reward
            # truncate very small probabilities
            self.trans_tensor = clip_and_normalize(self.trans_tensor, 1e-7, axis=3)
            self.cost_tensor_list = []
            for k in range(K):
                cost_tensor_k = np.random.uniform(0, 1, size=(max_N, sspa_size, aspa_size))
                cost_tensor_k[:,:,0] = 0 # action 0 has zero cost
                cost_tensor_k *= (cost_tensor_k >= 1e-7)
                self.cost_tensor_list.append(cost_tensor_k)
        else:
            raise NotImplementedError
        # make sure alpha is not too close to 0 or 1; round to integer multiples of 20
        self.alpha_list = list(np.round(20*np.random.uniform(0.1,0.9, size=K))/20)

    def get_reward_tensor(self, N):
        return self.reward_tensor[0:N,:,:]

    def get_trans_tensor(self, N):
        return self.trans_tensor[0:N,:,:,:]

    def get_cost_tensor_list(self, N):
        return [cost_tensor[0:N,:,:] for cost_tensor in self.cost_tensor_list]

    def print(self, verbose=False):
        print("typed_het={}, sspa_size={}, aspa_size={}\n max_N={} K={},distr={}, parameters={}".format(
            self.typed_het, self.sspa_size, self.aspa_size, self.max_N, self.K, self.distr, self.parameters))
        print("rmax = {}, cmax = {}, alphamin = {}".format(np.max(self.reward_tensor),
                                                           np.max(np.array(self.cost_tensor_list)),
                                                           np.min(np.array(self.alpha_list))))
        print("alpha list:")
        print(self.alpha_list)
        if verbose:
            print("reward tensor:")
            print(self.reward_tensor)
            print("trans tensor:")
            print(self.trans_tensor)
            print("cost tensor list:")
            print(self.cost_tensor_list)




class RandomExampleTypedHet(object):
    def __init__(self, sspa_size, aspa_size, nominal_type_frac, K, distr, parameters):
        """
        Generate a random wcmdp; the statistics of each arm is independently generated
        :param sspa_size: size of the state space
        :param aspa_size: size of the action space
        :param nominal_type_frac: the nominal fraction of arms of each type; the actual fractions depend on N and rounding
        :param K: number of constraints
        :param distr: string, "uniform" or "dirichlet"; the distribution for generating each kernel
        :param parameters: parameters of the distribution
        """
        self.typed_het = True
        self.sspa_size = sspa_size
        self.aspa_size = aspa_size
        self.num_types = len(nominal_type_frac)
        self.nominal_type_frac = nominal_type_frac
        self.K = K
        self.distr = distr
        self.parameters = parameters
        if distr == "dirichlet":
            if len(parameters) == 1:
                parameters = [parameters[0]] * self.sspa_size
            else:
                parameters = parameters
            self.trans_tensor = np.random.dirichlet(parameters, (self.num_types, sspa_size, aspa_size))
            self.reward_tensor = np.random.uniform(0, 1, size=(self.num_types, sspa_size, aspa_size))
            self.reward_tensor[:,:,0] = 0 # action 0 generates zero reward
            # truncate very small probabilities
            self.trans_tensor = clip_and_normalize(self.trans_tensor, 1e-7, axis=3)
            self.cost_tensor_list = []
            for k in range(K):
                cost_tensor_k = np.random.uniform(0, 1, size=(self.num_types, sspa_size, aspa_size))
                cost_tensor_k[:,:,0] = 0 # action 0 has zero cost
                cost_tensor_k *= (cost_tensor_k >= 1e-7)
                self.cost_tensor_list.append(cost_tensor_k)
        else:
            raise NotImplementedError
        # make sure alpha is not too close to 0 or 1; round to integer multiples of 20
        self.alpha_list = list(np.round(20*np.random.uniform(0.1,0.9), size=K)/20)

    def get_actual_type_frac(self, N):
        frac_partial_sums = np.zeros((self.num_types,))
        frac_partial_sums[0] = self.nominal_type_frac[0]
        for type in range(1, self.num_types):
            frac_partial_sums[type] = frac_partial_sums[type-1] + self.nominal_type_frac[type]
        for type in range(self.num_types):
            frac_partial_sums[type] = round(N*frac_partial_sums[type])/N  # round to integer multiples of N
        actual_type_frac = np.zeros((self.num_types,))
        actual_type_frac[0] = frac_partial_sums[0]
        for type in range(self.num_types):
            actual_type_frac[type] = frac_partial_sums[type] - frac_partial_sums[type-1]
        return actual_type_frac


def clip_and_normalize(tensor, epsilon, axis):
    tensor *= (tensor > epsilon)
    tensor /= np.sum(tensor, axis=axis, keepdims=True)
    return tensor


if __name__ == "__main__":
    np.random.seed(42)
    sspa_size = 5
    aspa_size = 3
    max_N = 200
    K = 3
    for ell in range(1):
        example = RandomExampleFullyHet(sspa_size, aspa_size, max_N, K, "dirichlet", [1])
        save_path = "examples/uniform-S{}A{}N{}K{}fh-{}".format(sspa_size, aspa_size, max_N, K, ell)
        print("saving to: ", save_path)
        if os.path.exists(save_path):
            pass
        else:
            with open(save_path, "wb") as f:
                pickle.dump(example, f)

        with open(save_path, "rb") as f:
            loaded_example = pickle.load(f)
        loaded_example.print(verbose=True)
        assert np.allclose(loaded_example.reward_tensor, example.reward_tensor, atol=1e-10)
        assert np.allclose(loaded_example.trans_tensor, example.trans_tensor, atol=1e-10)
        for k in range(example.K):
            assert np.allclose(loaded_example.cost_tensor_list[k], example.cost_tensor_list[k], atol=1e-10)
