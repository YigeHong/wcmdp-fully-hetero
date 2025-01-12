import numpy as np
import pickle
from wcmdp import SingleArmAnalyzer
from matplotlib import pyplot as plt
import warnings
import os

# todo: save the object using its attributes rather than the whole class

# all classes need to have the following properties: sspa_size, aspa_size, alpha_list, K
# and methods: get_trans_tensor, get_reward_tensor, get_cost_tensor_list, get_type_vec, get_actual_type_frac

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
        # self.typed_het = False
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
        raw_alphas = np.random.uniform(0.1,0.9, size=K)
        self.alpha_list = [round(20*alpha)/20 for alpha in raw_alphas]

    def get_type_vec(self, N):
        return np.arange(N, dtype=int)

    def get_actual_type_fracs(self, N):
        return np.ones((N,), dtype=np.float64) / N

    def get_reward_tensor(self, N):
        return self.reward_tensor[0:N,:,:]

    def get_trans_tensor(self, N):
        return self.trans_tensor[0:N,:,:,:]

    def get_cost_tensor_list(self, N):
        return [cost_tensor[0:N,:,:] for cost_tensor in self.cost_tensor_list]

    def print(self, verbose=False):
        print("sspa_size={}, aspa_size={}\n max_N={} K={},distr={}, parameters={}".format(
            self.sspa_size, self.aspa_size, self.max_N, self.K, self.distr, self.parameters))
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
    def __init__(self, sspa_size, aspa_size, nominal_type_frac, K, distr, parameters, is_rb=False):
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
            self.trans_tensor = np.random.dirichlet(parameters, (num_types, sspa_size, aspa_size))
            self.reward_tensor = np.random.uniform(0, 1, size=(num_types, sspa_size, aspa_size))
            self.reward_tensor[:,:,0] = 0 # action 0 generates zero reward
            # truncate very small probabilities
            self.trans_tensor = clip_and_normalize(self.trans_tensor, 1e-7, axis=3)
            if not is_rb:
                self.cost_tensor_list = []
                for k in range(K):
                    cost_tensor_k = np.random.uniform(0, 1, size=(num_types, sspa_size, aspa_size))
                    cost_tensor_k[:,:,0] = 0 # action 0 has zero cost
                    cost_tensor_k *= (cost_tensor_k >= 1e-7)
                    self.cost_tensor_list.append(cost_tensor_k)
            else:
                assert K == 1
                assert aspa_size == 2
                cost_tensor = np.zeros((num_types, sspa_size, aspa_size))
                cost_tensor[:,:,1] = 1
                self.cost_tensor_list = [cost_tensor]
                print(self.cost_tensor_list)
        else:
            raise NotImplementedError
        # make sure alpha is not too close to 0 or 1; round to integer multiples of 20
        raw_alphas = np.random.uniform(0.1,0.3, size=K)
        self.alpha_list = [round(20*alpha)/20 for alpha in raw_alphas]

        # the CDF of the arms' nominal type distribution
        self.nominal_type_cdf = np.zeros((self.num_types,))
        self.nominal_type_cdf[0] = self.nominal_type_frac[0]
        for t_ind in range(1, self.num_types):
            self.nominal_type_cdf[t_ind] = self.nominal_type_cdf[t_ind-1] + self.nominal_type_frac[t_ind]
        # store the id2type vectors that have been computed, to save computations
        self.all_id2type = {}

    def get_actual_type_fracs(self, N):
        actual_type_frac = np.zeros((self.num_types,))
        actual_type_frac[0] = round(self.nominal_type_cdf[0]*N)/N # round to integer multiples of N
        for t_ind in range(1, self.num_types):
            actual_type_frac[t_ind] = round(self.nominal_type_cdf[t_ind]*N)/N - round(self.nominal_type_cdf[t_ind-1]*N)/N
        return actual_type_frac

    def get_type_vec(self, N):
        if N in self.all_id2type:
            return self.all_id2type[N]
        else:
            id2type = np.zeros((N,), dtype=int)
            for t_ind in range(self.num_types):
                start_id = int(round(self.nominal_type_cdf[t_ind-1]*N)) if t_ind >=1 else 0
                end_id = int(round(self.nominal_type_cdf[t_ind]*N))
                for i in range(start_id, end_id):
                    id2type[i] = t_ind
            self.all_id2type[N] = id2type
            return id2type

    def get_reward_tensor(self, N):
        return self.reward_tensor.copy()

    def get_trans_tensor(self, N):
        return self.trans_tensor.copy()

    def get_cost_tensor_list(self, N):
        return self.cost_tensor_list.copy()

    def print(self, verbose=False):
        print("sspa_size={}, aspa_size={}\n num_types={} K={},distr={}, parameters={}".format(
            self.sspa_size, self.aspa_size, self.num_types, self.K, self.distr, self.parameters))
        print("rmax = {}, cmax = {}, alphamin = {}".format(np.max(self.reward_tensor),
                                                           np.max(np.array(self.cost_tensor_list)),
                                                           np.min(np.array(self.alpha_list))))
        print("alpha list:")
        print(self.alpha_list)
        # print("nominal type fractions:")
        # print(self.nominal_type_frac)
        if verbose:
            print("reward tensor:")
            print(self.reward_tensor)
            print("trans tensor:")
            print(self.trans_tensor)
            print("cost tensor list:")
            print(self.cost_tensor_list)


def clip_and_normalize(tensor, epsilon, axis):
    tensor *= (tensor > epsilon)
    tensor /= np.sum(tensor, axis=axis, keepdims=True)
    return tensor


if __name__ == "__main__":
    np.random.seed(42)
    sspa_size = 10
    aspa_size = 2
    # max_N = 1000
    num_types = 1000
    nominal_type_frac = np.ones((num_types,), dtype=np.float64)/num_types
    K = 1
    is_rb = True
    for ell in range(1):
        # example = RandomExampleFullyHet(sspa_size, aspa_size, max_N, K, "dirichlet", [1])
        example = RandomExampleTypedHet(sspa_size, aspa_size, nominal_type_frac, K, "dirichlet", [0.1], is_rb=is_rb)
        if not is_rb:
            save_path = "examples/uniform-S{}A{}types{}K{}-{}".format(sspa_size, aspa_size, num_types, K, ell)
        else:
            save_path = "examples/uniform-S{}A{}types{}K{}rb-{}".format(sspa_size, aspa_size, num_types, K, ell)
        print("saving to: ", save_path)
        if os.path.exists(save_path):
            print("file already exists, new example not saved")
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
