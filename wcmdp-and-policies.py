"""
This file defines the weakly-coupled MDP with heterogenous arms,
the helper class for solving the LP relaxations,
classes for RB policies,
along with a few helper functions
"""

import numpy as np
import cvxpy as cp
import scipy
import warnings

class WCMDP(object):
    """
    :param trans_tensor: n-d vector with dims=(N, sspa_size, aspa_size, sspa_size) and dtype=float
    note that the cost constraint is not encoded in this class
    """
    def __init__(self, sspa_size, aspa_size, N, trans_tensor, reward_tensor, init_states=None):
        self.sspa_size = sspa_size
        self.sspa = np.array(list(range(self.sspa_size)))
        self.aspa_size = aspa_size
        self.aspa = np.array(list(range(self.aspa_size)))
        self.sa_pairs = []
        for s in self.sspa:
            for a in self.aspa:
                self.sa_pairs = self.sa_pairs.append((s, a))
        self.N = N
        self.trans_tensor = trans_tensor
        self.reward_tensor = reward_tensor
        # initialize the state of the arms at 0
        if init_states is not None:
            self.states = init_states.copy()
        else:
            self.states = np.zeros((self.N,))

    def get_states(self):
        return self.states.copy()

    def step(self, actions):
        """
        :param actions: a 1-d array with length N. Each entry is an int in the range [0,self.aspa-1], denoting the action of each arm
        :return: intantaneous reward of this time step
        """
        instant_reward = 0
        for i in range(self.N):
            cur_s = self.states[i]
            cur_a = actions[i]
            instant_reward += self.reward_tensor[i, cur_s, cur_a]
            self.states[i] = np.random.choice(self.sspa, 1, p=self.trans_tensor[i, cur_s, cur_a:])
        instant_reward = instant_reward / self.N  # we normalize it by the number of arms
        return instant_reward

    def check_budget_constraints(self):
        pass

    def get_s_counts(self):
        s_counts = np.zeros((self.sspa_size,))
        # find out the arms whose (state, action) = sa_pair
        for s in self.sspa:
            s_counts[s] = len(np.where([self.states == s])[0])
        return s_counts

    def get_s_fracs(self):
        s_counts = self.get_s_counts()
        s_fracs = np.zeros((self.sspa_size,))
        for s in self.sspa:
            s_fracs[s] = s_counts[s] / self.N
        return s_fracs
    
    def reassign_ids(self, costs_per_arm, eps, d):
        """
        :param costs_per_arm: the costs of each arm, in the shape of N x K
        :param eps: the threshold for the cost of each interval
        :param d: the length of the interval
        :return: the new assignment of the arms, in the shape of N
        """
        intervals = self.N // d
        new_ids = np.full((self.N,),-1)
        used_ids = np.zeros((self.N,))
        K = costs_per_arm.shape[1] if len(costs_per_arm.shape) > 1 else 1
        for i in range(intervals):
            cur_id = i * d
            for k in range(K):
                cur_cost_k = sum(costs_per_arm[j, k] for j in range(self.N) if i * d <= new_ids[j] < (i + 1) * d)
                if cur_cost_k < eps:
                    for j in range(self.N):
                        if not used_ids[j] and costs_per_arm[j, k] >= eps:
                            new_ids[j] = cur_id
                            cur_id += 1
                            used_ids[j] = 1
        unused_ids = set(range(self.N)) - set(new_ids)
        for j in range(self.N):
            if new_ids[j] == -1:
                new_ids[j] = unused_ids.pop()
                used_ids[j] = 1 # mark it as used
        return new_ids



class SingleArmAnalyzer(object):
    def __init__(self, sspa_size, aspa_size, N, trans_tensor, reward_tensor, K, cost_tensor_list, alpha_list):
        self.sspa_size = sspa_size
        self.sspa = np.array(list(range(self.sspa_size)))
        self.aspa_size = aspa_size
        self.aspa = np.array(list(range(self.aspa_size)))
        self.sa_pairs = []
        for s in self.sspa:
            for a in self.aspa:
                self.sa_pairs = self.sa_pairs.append((s, a))
        self.N = N
        self.trans_tensor = trans_tensor
        self.reward_tensor = reward_tensor
        self.K = K
        self.cost_tensor_list = cost_tensor_list
        self.alpha_list = alpha_list

        # any numbers smaller than self.EPS are regard as zero
        self.EPS = 1e-8

        # variables
        self.y = cp.Variable((N, self.sspa_size, self.aspa_size))
        # self.dualvars = cp.Parameter((K,), name="dualvar")  # the subsidy parameter for solving Whittle's index policy

        # store some data of the solution, only needed for solving the LP-Priority policy
        # the values might change, so they are not safe to use unless immediately after solving the LP.
        self.opt_value = None
        # self.avg_reward = None
        # self.opt_subsidy = None
        # self.value_func_relaxed = np.zeros((self.sspa_size,))
        # self.q_func_relaxed = np.zeros((self.sspa_size, 2))

        self.state_probs = None  # optimal state frequency for each arm, ndarray, dims=(N, sspa), dtype=float
        self.policies = None  # optimal single-armed policies for each arm, ndarray, dims=(N, sspa, aspa), dtype=float
        self.Ps = None # transition matrix under the optimal single-armed policy for each arm dims=(N, sspa, sspa), dtype=float

    def get_stationary_constraints(self):
        stationary_constrs = []
        for i in range(self.N):
            for cur_s in self.sspa:
                mu_s = cp.sum(cp.multiply(self.y[i,:,:], self.trans_tensor[i,:,:,cur_s]))
                stationary_constrs.append(mu_s == cp.sum(self.y[i, cur_s, :]))
        return stationary_constrs

    def get_budget_constraints(self):
        budget_constrs = []
        for k in range(self.K):
            cost_tensor = self.cost_tensor_list[k]
            alpha = self.alpha_list[k]
            budget_constrs.append(cp.sum(cp.multiply(self.y, cost_tensor)) == alpha*self.N)  
        return budget_constrs

    def get_basic_constraints(self):
        # the constraints that make sure we solve a probability distribution
        basic_constrs = []
        basic_constrs.append(self.y >= 0.1*self.EPS)
        basic_constrs.append(cp.sum(self.y, axis=(1,2)) == 1)
        return basic_constrs

    def get_objective(self):
        objective = cp.Maximize(cp.sum(cp.multiply(self.y, self.reward_tensor))/self.N)
        return objective

    def solve_lp(self):
        objective = self.get_objective()
        constrs = self.get_stationary_constraints() + self.get_budget_constraints() + self.get_basic_constraints()
        problem = cp.Problem(objective, constrs)
        self.opt_value = problem.solve(verbose=False)
        y = self.y.value
        # to take account of imprecise solutions, make sure y[i,:,:] is non-negative and sums to 1
        y = y * (y>=0)
        y = y / np.sum(y, axis=(1,2), keep_dim=True)
        self.state_probs = np.sum(y, axis=2)
        self.policies = np.zeros((self.N, self.sspa_size, self.aspa_size))
        for i in range(self.N):
            for s in self.sspa:
                if self.state_probs[i, s] > self.EPS:
                    self.policies[i, s, :] = y[i, s, :] / self.state_probs[i, s]
                else:
                    self.policies[i, s, :] = 1 / self.aspa_size
        self.Ps = np.zeros((self.N, self.sspa_size, self.sspa_size,))
        for i in range(self.N):
            for a in self.aspa:
                self.Ps[i,:,:] += self.trans_tensor[i,:,a,:]*np.expand_dims(self.policies[i,:,a], axis=1)
        return (self.opt_value, y)

    def print_LP_solution(self, arm_id):
        """
        :param arm_id: the ID of the arm whose LP solution you want to print; if arm_id == -1, print the average
        """
        if arm_id >= 0:
            yi = self.y.value[arm_id,:,:]
        elif arm_id == -1:
            yi = np.sum(self.y.value, axis=0)
        else:
            raise NotImplementedError
        print("--------The solution of the {}-th arm-------".format(arm_id))
        print("Expected reward: ", yi*self.reward_tensor[arm_id,:,:])
        for k in range(self.K):
            print("Expected type-{} cost: ", yi*self.cost_tensor_list[k][arm_id,:,:])
        print("Optimal var")
        print(yi)
        print("Optimal state frequency=", self.state_probs[arm_id,:])
        print("Single armed policy=", self.policies[arm_id,:,:])
        print("---------------------------")


class IDPolicy(object):
    def __init__(self, sspa_size, aspa_size, N, policies, K, cost_tensor_list, alpha_list):
        self.sspa_size = sspa_size
        self.sspa = np.array(list(range(self.sspa_size)))
        self.aspa_size = aspa_size
        self.aspa = np.array(list(range(self.aspa_size)))
        self.sa_pairs = []
        for s in self.sspa:
            for a in self.aspa:
                self.sa_pairs = self.sa_pairs.append((s, a))
        self.N = N
        self.policies = policies
        self.K = K
        self.cost_tensor_list = cost_tensor_list
        self.alpha_list = alpha_list
        self.EPS = 1e-8

        # # compute the single-armed policies from the solution y
        # self.state_probs = np.sum(self.y, axis=2)
        # self.policies = np.zeros((self.N, self.sspa_size, self.aspa_size))
        # for i in range(self.N):
        #     for s in self.sspa:
        #         if self.state_probs[i, s] > self.EPS:
        #             self.policies[i, s, :] = y[i, s, :] / self.state_probs[i, s]
        #         else:
        #             self.policies[i, s, :] = 1 / self.aspa_size
        if not np.allclose(np.sum(self.policies, axis=2), np.ones((self.N, self.sspa_size)), atol=1e-4):
            print("policy definition wrong, the action probs do not sum up to 1. The wrong arms and policies are")
            for i in range(self.N):
                if not np.allclose(np.sum(self.policies[i,:,:], axis=1), np.ones((self.sspa_size,)), atol=1e-4):
                    print("i={}, pibar_i={}".format(i, self.policies[i,:,:]))
            raise ValueError

    def get_actions(self, cur_states):
        """
        :param cur_states: the current states of the arms
        :return: the actions taken by the arms under the policy
        """
        pass




