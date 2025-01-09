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
import functools
import operator
import logging
logging.basicConfig(level=logging.WARNING)


class SingleArmAnalyzer(object):
    def __init__(self, sspa_size, aspa_size, N, trans_tensor, reward_tensor, K, cost_tensor_list, alpha_list):
        self.sspa_size = sspa_size
        self.sspa = np.array(list(range(self.sspa_size)), dtype=int)
        self.aspa_size = aspa_size
        self.aspa = np.array(list(range(self.aspa_size)), dtype=int)
        self.sa_pairs = []
        for s in self.sspa:
            for a in self.aspa:
                self.sa_pairs.append((s, a))
        self.N = N
        self.trans_tensor = trans_tensor
        self.reward_tensor = reward_tensor
        self.K = K
        self.cost_tensor_list = cost_tensor_list
        self.alpha_list = alpha_list
        # any numbers smaller than self.EPS are regard as zero
        self.EPS = 1e-8
        # check dimensions
        assert self.trans_tensor.shape[0] == N
        assert self.reward_tensor.shape[0] == N
        for k in range(K):
            assert self.cost_tensor_list[k].shape[0] == N

        # two parameters used in reassignment
        self.cmax = np.max(np.array(self.cost_tensor_list))
        self.alphamin = np.min(np.array(alpha_list))

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
            budget_constrs.append(cp.sum(cp.multiply(self.y, cost_tensor)) <= alpha*self.N)
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
        y = y / np.sum(y, axis=(1,2), keepdims=True)
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
        return self.opt_value, y

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


    def plot_cost_slopes(self, cost_table:np.ndarray):
        from matplotlib import pyplot as plt
        for k in range(cost_table.shape[0]):
            plt.plot(cost_table[k,:], label="type-{}".format(k),color="C{}".format(k))
        plt.xlabel("arms")
        plt.ylabel("expected costs")
        plt.legend()
        plt.savefig("cost_slopes.png")
            

    def reassign_ID(self, method):
        """
        run this method after solving the LP
        :method: "random" or "intervals"
        :return: new_orders, which is a permuted list of 0,1,...,N-1. The new_orders[i] indicates old id of the arm whose new ID=i
        """
        if method == "random":
            return np.random.permutation(self.N)
        elif method == "intervals":
            exp_cost_table = np.zeros((self.K, self.N,))
            for k in range(self.K):
                for i in range(self.N):
                    exp_cost_table[k,i] = np.dot(self.cost_tensor_list[k][i,:,:].flatten(), self.y.value[i,:,:].flatten())
            active_constrs = np.sum(exp_cost_table, axis=1) >= (0.5*np.array(self.alpha_list)*self.N)
            logging.debug("active constraints = {}".format(active_constrs))
            self.plot_cost_slopes(exp_cost_table)
            # solve cost_thresh from a qaudratic equation to optimize the cost sloe
            # coefficients of a quadratic 
            co_a = 2*self.K - 1
            co_b = 0.5*self.alphamin - 2*self.K*self.cmax - 0.5*self.alphamin*self.K
            co_c = 0.5*self.alphamin*self.cmax*self.K
            cost_thresh = (-co_b - np.sqrt(co_b**2 - 4*co_a*co_c)) / (2*co_a)
            # nominal_intv_len = self.K * (self.cmax - cost_thresh) / (0.5*self.alphamin - cost_thresh)
            rem_costly_arms_table = exp_cost_table >= cost_thresh
            num_costly_arms = np.sum(rem_costly_arms_table, axis=1)
            nominal_intv_len = self.K * self.N / np.min(num_costly_arms)
            logging.debug("cost_thresh={}, nominal_intv_len={}".format(cost_thresh, nominal_intv_len))
            if nominal_intv_len > self.N:
                nominal_intv_len = self.N

            rem_costly_arms_table = exp_cost_table >= cost_thresh
            logging.debug("rem_costly_arms_table=\n{}".format(rem_costly_arms_table))
            non_costly_arms_list = list(np.where(np.sum(rem_costly_arms_table, axis=0) == 0)[0])
            points_to_next_costly = np.zeros((self.K,), dtype=int) # pointers to the smallest-index arm whose type-k cost is larger than the threshold
            for k in range(self.K):
                if active_constrs[k] == 1:
                    type_k_costly = np.where(rem_costly_arms_table[k,:]>0)[0]
                    # type_k_costly must be non-empty if the k-th budget contraint is active
                    points_to_next_costly[k] = np.min(type_k_costly) #if len(type_k_costly) > 0 else self.N
                else:
                    points_to_next_costly[k] = self.N
            costly_groups = [] # groups of arms; the total type-k cost of each group will be larger than cost_thresh, for each k
            rem_arms_list = []
            while True:
                # each outer loop creates a group of costly arms
                costly_groups.append([])
                fulfilled_cost_types = np.zeros((self.K,))
                for k in range(self.K):
                    if fulfilled_cost_types[k] == 0:
                        # find the next arm to add
                        arm_to_add = points_to_next_costly[k]
                        if arm_to_add >= self.N:
                            continue
                        assert rem_costly_arms_table[k,arm_to_add] > 0 # temporary, check correctness of the code
                        # add the arm, update tables
                        costly_groups[-1].append(arm_to_add)
                        fulfilled_cost_types += rem_costly_arms_table[:,arm_to_add]
                        rem_costly_arms_table[:,arm_to_add] = 0
                        # for each cost type k, find the next costly arm that hasn't been added into the groups
                        for _k in range(self.K):
                            new_pointer = points_to_next_costly[_k]
                            while (new_pointer < self.N) and (rem_costly_arms_table[_k, new_pointer] == 0):
                                new_pointer += 1
                            points_to_next_costly[_k] = new_pointer
                    else:
                        continue
                logging.debug(points_to_next_costly)
                if np.any((points_to_next_costly >= self.N) * active_constrs):
                    if np.any((fulfilled_cost_types == 0) * active_constrs):
                        # if the cost of last group is not fulfilled, remove from the groups, and add them into the remaining arms
                        unfulfilled_group = costly_groups.pop()
                        rem_arms_list.extend(unfulfilled_group)
                    # merge the costly arms that are not added into the groups into non_costly_arms_list
                    rem_costly_arms_list = list(np.where(np.sum(rem_costly_arms_table, axis=0)>0)[0])
                    rem_arms_list.extend(rem_costly_arms_list)
                    rem_arms_list.extend(non_costly_arms_list)
                    logging.debug("number of arms in costly groups={}, rem_costly_arms_list={}, non_costly_arms_list={}".format(
                            sum([len(group) for group in costly_groups]), len(rem_costly_arms_list), len(non_costly_arms_list)))

                    break
            logging.debug("number of arms in costly groups={}, remaining number of arms={}".format(
                sum([len(group) for group in costly_groups]), len(rem_arms_list)))

            # calculate the total number of intervals of each length
            intv_len_1 = int(np.ceil(nominal_intv_len))
            num_intvs = int(np.floor(self.N / intv_len_1))
            intv_len_2 = intv_len_1 - 1
            num_len_1_intv = self.N - num_intvs * intv_len_2 # todo: there could be num_len_1_intv > num_intvs
            assert num_len_1_intv * intv_len_1 + (num_intvs - num_len_1_intv) * intv_len_2 == self.N
            logging.debug("num_len_1_intv={}, intv_len_1={}, intv_len_2={}".format(num_len_1_intv, intv_len_1, intv_len_2))
            logging.debug("costly group lengths = {}".format([len(group) for group in costly_groups]))
            # merge the extra groups into remaining arms
            for group in costly_groups[num_intvs:]:
                rem_arms_list.extend(group)
            costly_groups = costly_groups[0:num_intvs]
            np.random.shuffle(rem_arms_list)
            logging.debug("number of arms in costly groups={}, remaining number of arms={}".format(
                sum([len(group) for group in costly_groups]), len(rem_arms_list)))
            # combine the costly arm groups and the remaining arms into intervals of the lengths specified above
            rem_arms_pt = 0
            for ell in range(num_intvs):
                num_exist = len(costly_groups[ell])
                num_to_add = intv_len_1-num_exist if ell < num_len_1_intv else intv_len_2-num_exist
                assert num_to_add >= 0
                arms_to_add = rem_arms_list[rem_arms_pt:(rem_arms_pt+num_to_add)]
                costly_groups[ell].extend(arms_to_add)
                rem_arms_pt += num_to_add
            logging.debug("number of arms in costly groups={}, remaining number of arms={}".format(
                sum([len(group) for group in costly_groups]), len(rem_arms_list)))

            new_orders = functools.reduce(operator.iconcat, costly_groups, [])
            new_orders = np.array(new_orders)

            assert np.allclose(np.sort(new_orders), np.arange(self.N, dtype=int))
            for k in range(self.K):
                if active_constrs[k] == 1:
                    temp_pt = 0
                    for ell in range(num_intvs):
                        intv_len = intv_len_1 if ell < num_len_1_intv else intv_len_2
                        cur_interval_exp_cost =  sum([exp_cost_table[k,i] for i in new_orders[ell:(ell+intv_len)]])
                        logging.debug("cur_interval_exp_cost={}".format(cur_interval_exp_cost))
                        assert cur_interval_exp_cost>= cost_thresh, \
                            "{}-th interval violates type-{} cost-slope require requirement: actual expected cost= {} < cost threshold = {};" \
                            " involving arms with old {}".format(ell, k, cur_interval_exp_cost, cost_thresh, new_orders[ell:(ell+intv_len)])
                        temp_pt += intv_len

            return new_orders
        else:
            raise NotImplementedError


class WCMDP(object):
    """
    :param trans_tensor: n-d vector with dims=(N, sspa_size, aspa_size, sspa_size) and dtype=float
    note that the cost constraint is not encoded in this class
    """
    def __init__(self, sspa_size, aspa_size, N, trans_tensor, reward_tensor, init_states=None):
        self.sspa_size = sspa_size
        self.sspa = np.array(list(range(self.sspa_size)), dtype=int)
        self.aspa_size = aspa_size
        self.aspa = np.array(list(range(self.aspa_size)), dtype=int)
        self.sa_pairs = []
        for s in self.sspa:
            for a in self.aspa:
                self.sa_pairs.append((s, a))
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
            self.states[i] = np.random.choice(self.sspa, 1, p=self.trans_tensor[i, cur_s, cur_a, :])
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


class IDPolicy(object):
    def __init__(self, sspa_size, aspa_size, N, policies, K, cost_tensor_list, alpha_list):
        self.sspa_size = sspa_size
        self.sspa = np.array(list(range(self.sspa_size)), dtype=int)
        self.aspa_size = aspa_size
        self.aspa = np.array(list(range(self.aspa_size)), dtype=int)
        self.sa_pairs = []
        for s in self.sspa:
            for a in self.aspa:
                self.sa_pairs.append((s, a))
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
        actions = np.zeros((self.N,), dtype=int)
        for i in range(self.N):
            actions[i] = np.random.choice(self.aspa, size=1, p=self.policies[i,cur_states[i],:])

        cost_partial_sum_table = np.zeros((self.K, self.N,))
        for k in range(self.K):
            for i in range(self.N):
                if i == 0:
                    cost_partial_sum_table[k,i] = self.cost_tensor_list[k][i, cur_states[i], actions[i]]
                else:
                    cost_partial_sum_table[k,i] = cost_partial_sum_table[k,i-1] + self.cost_tensor_list[k][i, cur_states[i], actions[i]]
        budget_vec = self.N * np.array(self.alpha_list)
        conform_table = cost_partial_sum_table < np.expand_dims(budget_vec, axis=1)
        conform_arms_vec = np.prod(conform_table, axis=0)
        non_conform_arms_list = np.where(conform_arms_vec==0)[0]
        N_star = np.min(non_conform_arms_list) if len(non_conform_arms_list) > 0 else self.N # i < N_star follow the single-armed policy
        actions[N_star:] = 0

        for k in range(self.K):
            assert sum([self.cost_tensor_list[k][i, cur_states[i], actions[i]] for i in range(self.N)]) <= budget_vec[k], "{}-th cost exceeds budget"

        return actions, N_star



