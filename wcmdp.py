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
    def __init__(self, sspa_size, aspa_size, N, type_fracs, trans_tensor, reward_tensor, K, cost_tensor_list, alpha_list):
        self.sspa_size = sspa_size
        self.sspa = np.array(list(range(self.sspa_size)), dtype=int)
        self.aspa_size = aspa_size
        self.aspa = np.array(list(range(self.aspa_size)), dtype=int)
        self.sa_pairs = []
        for s in self.sspa:
            for a in self.aspa:
                self.sa_pairs.append((s, a))
        self.N = N
        self.num_types = len(type_fracs)
        self.type_fracs = type_fracs # fraction of arms of each type (true fraction, s.t. type_fracs[j]*N is integer)
        self.trans_tensor = trans_tensor
        self.reward_tensor = reward_tensor
        self.K = K
        self.cost_tensor_list = cost_tensor_list
        self.alpha_list = alpha_list
        # any numbers smaller than self.EPS are regard as zero
        self.EPS = 1e-8
        # check dimensions
        assert self.N >= self.num_types
        assert len(self.type_fracs.shape) == 1
        assert self.trans_tensor.shape == (self.num_types, sspa_size, aspa_size, sspa_size)
        assert self.reward_tensor.shape == (self.num_types, sspa_size, aspa_size)
        for k in range(K):
            assert self.cost_tensor_list[k].shape == (self.num_types, sspa_size, aspa_size)
            assert (type(self.alpha_list[k]) == float) or (len(self.alpha_list[k].shape) == 0)

        # two parameters used in reassignment
        self.cmax = np.max(np.array(self.cost_tensor_list))
        self.alphamin = np.min(np.array(alpha_list))

        # variables
        self.y = cp.Variable((self.num_types, self.sspa_size, self.aspa_size))
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
        for j in range(self.num_types):
            for cur_s in self.sspa:
                mu_s = cp.sum(cp.multiply(self.y[j,:,:], self.trans_tensor[j,:,:,cur_s]))
                stationary_constrs.append(mu_s == cp.sum(self.y[j, cur_s, :]))
        return stationary_constrs

    def get_budget_constraints(self):
        budget_constrs = []
        for k in range(self.K):
            cost_tensor = self.cost_tensor_list[k]
            alpha = self.alpha_list[k]
            cost_vec = cp.sum(cp.multiply(self.y, cost_tensor), axis=(1,2))
            budget_constrs.append(self.type_fracs @ cost_vec <= alpha)
        return budget_constrs

    def get_basic_constraints(self):
        # the constraints that make sure we solve a probability distribution
        basic_constrs = []
        basic_constrs.append(self.y >= 0.1*self.EPS)
        basic_constrs.append(cp.sum(self.y, axis=(1,2)) == 1)
        return basic_constrs

    def get_objective(self):
        reward_vec = cp.sum(cp.multiply(self.y, self.reward_tensor), axis=(1,2))
        objective = cp.Maximize(self.type_fracs @ reward_vec)
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
        self.policies = np.zeros((self.num_types, self.sspa_size, self.aspa_size,))
        for j in range(self.num_types):
            for s in self.sspa:
                if self.state_probs[j, s] > self.EPS:
                    self.policies[j, s, :] = y[j, s, :] / self.state_probs[j, s]
                else:
                    self.policies[j, s, :] = 1 / self.aspa_size
        self.Ps = np.zeros((self.num_types, self.sspa_size, self.sspa_size,))
        for j in range(self.num_types):
            for a in self.aspa:
                self.Ps[j,:,:] += self.trans_tensor[j,:,a,:]*np.expand_dims(self.policies[j,:,a], axis=1)
        return self.opt_value, y

    def print_LP_solution(self, arm_type):
        """
        :param arm_id: the ID of the arm whose LP solution you want to print; if arm_id == -1, print the average
        """
        if arm_type >= 0:
            yj = self.y.value[arm_type,:,:]
        elif arm_type == -1:
            yj = np.sum(self.y.value * np.expand_dims(self.type_fracs, axis=(1,2)), axis=0)
        else:
            raise NotImplementedError
        print("--------The solution of the {}-th arm-------".format(arm_type))
        print("Expected reward: ", np.sum(yj*self.reward_tensor[arm_type,:,:]))
        for k in range(self.K):
            print("Expected type-{} cost: ".format(k), np.sum(yj*self.cost_tensor_list[k][arm_type,:,:]))
        print("Optimal var")
        print(yj)
        print("Optimal state frequency=", self.state_probs[arm_type,:])
        print("Single armed policy=", self.policies[arm_type,:,:])
        print("---------------------------")


    def plot_cost_slopes(self, cost_table:np.ndarray):
        from matplotlib import pyplot as plt
        for k in range(cost_table.shape[0]):
            plt.plot(cost_table[k,:], label="type-{}".format(k),color="C{}".format(k))
        plt.xlabel("arms")
        plt.ylabel("expected costs")
        plt.legend()
        plt.savefig("cost_slopes.png")
        plt.close()


    def reassign_ID(self, method):
        """
        run this method after solving the LP
        :method: "random" or "intervals"
        :return: new_orders, which is a permuted list of 0,1,...,N-1. The new_orders[i] indicates old id of the arm whose new ID=i
        """
        if method == "random":
            return np.random.permutation(self.N)
        elif method == "ascending":
            exp_cost_table = np.zeros((self.K, self.N,))
            for k in range(self.K):
                for i in range(self.N):
                    exp_cost_table[k,i] = np.dot(self.cost_tensor_list[k][i,:,:].flatten(), self.y.value[i,:,:].flatten())
            self.plot_cost_slopes(exp_cost_table)
            new_orders = np.argsort(np.sum(exp_cost_table, axis=0))
            self.plot_cost_slopes(exp_cost_table[:,new_orders])
            return new_orders
        elif method == "descending":
            exp_cost_table = np.zeros((self.K, self.N,))
            for k in range(self.K):
                for i in range(self.N):
                    exp_cost_table[k,i] = np.dot(self.cost_tensor_list[k][i,:,:].flatten(), self.y.value[i,:,:].flatten())
            self.plot_cost_slopes(exp_cost_table)
            new_orders = np.argsort(-np.sum(exp_cost_table, axis=0))
            self.plot_cost_slopes(exp_cost_table[:,new_orders])
            return new_orders
        elif method == "intervals":
            # arrange each type of arms sequentially, and map from type to IDs
            type2ids_table = np.zeros((self.num_types, self.N))
            type_frac_partial_sums = np.zeros((self.num_types))
            for j in range(self.num_types):
                if j == 0:
                    type_frac_partial_sums[j] = self.type_fracs[j]
                else:
                    type_frac_partial_sums[j] = type_frac_partial_sums[j-1] + self.type_fracs[j]
            for j in range(self.num_types):
                start_id = int(round(type_frac_partial_sums[j-1])) if j >= 1 else 0
                end_id = int(round(type_frac_partial_sums[j]))
                type2ids_table[j][start_id:end_id] = 1
            # generate a table summarizing the expected cost of each arm
            exp_cost_table = np.zeros((self.K, self.N,))
            for k in range(self.K):
                for i in range(self.N):
                    i_type = np.where(type2ids_table[:,i] == 0)[0]
                    exp_cost_table[k,i] = np.dot(self.cost_tensor_list[k][i_type,:,:].flatten(), self.y.value[i_type,:,:].flatten())
            active_constrs = np.sum(exp_cost_table, axis=1) >= (0.5*np.array(self.alpha_list)*self.N)
            logging.debug("active constraints = {}".format(active_constrs))
            self.plot_cost_slopes(exp_cost_table)

            # solve cost_thresh from a quadratic equation to optimize the cost slope
            # coefficients of a quadratic function
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
            intv_len = int(np.ceil(nominal_intv_len))
            num_intervals = int(np.floor(self.N / intv_len))
            used_arms = np.zeros((self.N,), dtype=int)
            new_orders = np.full((self.N,), -1, dtype=int)
            for interval in range(num_intervals):
                cur = interval * intv_len
                for k in range(self.K):
                    k_cost_interval_sum = np.sum(exp_cost_table[k,i] for i in range(self.N) if new_orders[i] >= interval*intv_len and new_orders[i] < (interval+1)*intv_len)
                    if k_cost_interval_sum < cost_thresh:
                        for i in range(self.N):
                            if used_arms[i] == 0 and rem_costly_arms_table[k,i] == 1:
                                new_orders[cur] = i    ## check whether new_orders[i] = cur or reverse
                                used_arms[i] = 1
                                cur += 1
                                break
            unused_arms = set(np.where(used_arms == 0)[0])
            for i in range(self.N):
                if new_orders[i] == -1:
                    new_orders[i] = unused_arms.pop()
            ## sanity check
            assert len(unused_arms) == 0
            assert np.all(np.sort(new_orders) == np.arange(self.N)), "new_orders is not a permutation of 0,1,...,N-1"
            self.plot_cost_slopes(exp_cost_table[:,new_orders])
            return new_orders


            # non_costly_arms_list = list(np.where(np.sum(rem_costly_arms_table, axis=0) == 0)[0])
            # points_to_next_costly = np.zeros((self.K,), dtype=int) # pointers to the smallest-index arm whose type-k cost is larger than the threshold
            # for k in range(self.K):
            #     if active_constrs[k] == 1:
            #         type_k_costly = np.where(rem_costly_arms_table[k,:]>0)[0]
            #         # type_k_costly must be non-empty if the k-th budget contraint is active
            #         points_to_next_costly[k] = np.min(type_k_costly) #if len(type_k_costly) > 0 else self.N
            #     else:
            #         points_to_next_costly[k] = self.N
            # costly_groups = [] # groups of arms; the total type-k cost of each group will be larger than cost_thresh, for each k
            # rem_arms_list = []
            # while True:
            #     # each outer loop creates a group of costly arms
            #     costly_groups.append([])
            #     fulfilled_cost_types = np.zeros((self.K,))
            #     for k in range(self.K):
            #         if fulfilled_cost_types[k] == 0:
            #             # find the next arm to add
            #             arm_to_add = points_to_next_costly[k]
            #             if arm_to_add >= self.N:
            #                 continue
            #             assert rem_costly_arms_table[k,arm_to_add] > 0 # temporary, check correctness of the code
            #             # add the arm, update tables
            #             costly_groups[-1].append(arm_to_add)
            #             fulfilled_cost_types += rem_costly_arms_table[:,arm_to_add]
            #             rem_costly_arms_table[:,arm_to_add] = 0
            #             # for each cost type k, find the next costly arm that hasn't been added into the groups
            #             for _k in range(self.K):
            #                 new_pointer = points_to_next_costly[_k]
            #                 while (new_pointer < self.N) and (rem_costly_arms_table[_k, new_pointer] == 0):
            #                     new_pointer += 1
            #                 points_to_next_costly[_k] = new_pointer
            #         else:
            #             continue
            #     logging.debug(points_to_next_costly)
            #     if np.any((points_to_next_costly >= self.N) * active_constrs):
            #         if np.any((fulfilled_cost_types == 0) * active_constrs):
            #             # if the cost of last group is not fulfilled, remove from the groups, and add them into the remaining arms
            #             unfulfilled_group = costly_groups.pop()
            #             rem_arms_list.extend(unfulfilled_group)
            #         # merge the costly arms that are not added into the groups into non_costly_arms_list
            #         rem_costly_arms_list = list(np.where(np.sum(rem_costly_arms_table, axis=0)>0)[0])
            #         rem_arms_list.extend(rem_costly_arms_list)
            #         rem_arms_list.extend(non_costly_arms_list)
            #         logging.debug("number of arms in costly groups={}, rem_costly_arms_list={}, non_costly_arms_list={}".format(
            #                 sum([len(group) for group in costly_groups]), len(rem_costly_arms_list), len(non_costly_arms_list)))

            #         break
            # logging.debug("number of arms in costly groups={}, remaining number of arms={}".format(
            #     sum([len(group) for group in costly_groups]), len(rem_arms_list)))

            # # calculate the total number of intervals of each length
            # intv_len_1 = int(np.ceil(nominal_intv_len))
            # num_intvs = int(np.floor(self.N / intv_len_1))
            # intv_len_2 = intv_len_1 - 1
            # num_len_1_intv = self.N - num_intvs * intv_len_2
            # assert num_len_1_intv * intv_len_1 + (num_intvs - num_len_1_intv) * intv_len_2 == self.N
            # logging.debug("num_len_1_intv={}, intv_len_1={}, intv_len_2={}".format(num_len_1_intv, intv_len_1, intv_len_2))
            # logging.debug("costly group lengths = {}".format([len(group) for group in costly_groups]))
            # # merge the extra groups into remaining arms
            # for group in costly_groups[num_intvs:]:
            #     rem_arms_list.extend(group)
            # costly_groups = costly_groups[0:num_intvs]
            # np.random.shuffle(rem_arms_list)
            # logging.debug("number of arms in costly groups={}, remaining number of arms={}".format(
            #     sum([len(group) for group in costly_groups]), len(rem_arms_list)))
            # # combine the costly arm groups and the remaining arms into intervals of the lengths specified above
            # rem_arms_pt = 0
            # for ell in range(num_intvs):
            #     num_exist = len(costly_groups[ell])
            #     num_to_add = intv_len_1-num_exist if ell < num_len_1_intv else intv_len_2-num_exist
            #     assert num_to_add >= 0
            #     arms_to_add = rem_arms_list[rem_arms_pt:(rem_arms_pt+num_to_add)]
            #     costly_groups[ell].extend(arms_to_add)
            #     rem_arms_pt += num_to_add
            # logging.debug("number of arms in costly groups={}, remaining number of arms={}".format(
            #     sum([len(group) for group in costly_groups]), len(rem_arms_list)))

            # new_orders = functools.reduce(operator.iconcat, costly_groups, [])
            # new_orders = np.array(new_orders)

            # assert np.allclose(np.sort(new_orders), np.arange(self.N, dtype=int))
            # for k in range(self.K):
            #     if active_constrs[k] == 1:
            #         temp_pt = 0
            #         for ell in range(num_intvs):
            #             intv_len = intv_len_1 if ell < num_len_1_intv else intv_len_2
            #             cur_interval_exp_cost =  sum([exp_cost_table[k,i] for i in new_orders[ell:(ell+intv_len)]])
            #             logging.debug("cur_interval_exp_cost={}".format(cur_interval_exp_cost))
            #             assert cur_interval_exp_cost>= cost_thresh, \
            #                 "{}-th interval violates type-{} cost-slope require requirement: actual expected cost= {} < cost threshold = {};" \
            #                 " involving arms with old {}".format(ell, k, cur_interval_exp_cost, cost_thresh, new_orders[ell:(ell+intv_len)])
            #             temp_pt += intv_len

            # return new_orders
        else:
            raise NotImplementedError


class WCMDP(object):
    """
    :param trans_tensor: n-d vector with dims=(N, sspa_size, aspa_size, sspa_size) and dtype=float
    note that the cost constraint is not encoded in this class
    """
    def __init__(self, sspa_size, aspa_size, N, trans_tensor, reward_tensor, id2types, init_states=None):
        self.sspa_size = sspa_size
        self.sspa = np.array(list(range(self.sspa_size)), dtype=int)
        self.aspa_size = aspa_size
        self.aspa = np.array(list(range(self.aspa_size)), dtype=int)
        self.N = N
        self.num_types = trans_tensor.shape[0] #len(type_fracs)
        # self.type_fracs = type_fracs
        self.trans_tensor = trans_tensor
        self.reward_tensor = reward_tensor
        self.id2types = id2types.copy()
        # initialize the state of the arms at 0
        if init_states is not None:
            assert init_states.shape == (self.N,)
            self.states = init_states.copy()
        else:
            self.states = np.zeros((self.N,))
        # check the dimensions
        # assert len(self.type_fracs.shape) == 1
        assert self.reward_tensor.shape == (self.num_types, self.sspa_size, self.aspa_size)
        assert self.trans_tensor.shape == (self.num_types, self.sspa_size, self.aspa_size, self.sspa_size)
        assert self.id2types.shape == (self.N,)

        # precompute possible combinations of type, state and actions
        self.tsa_tuples = [] #type, state, action
        for t_ind in range(self.num_types):
            for s in self.sspa:
                for a in self.aspa:
                    self.tsa_tuples.append((t_ind, s, a))

    def get_states(self):
        return self.states.copy()

    def step(self, actions):
        """
        :param actions: a 1-d array with length N. Each entry is an int in the range [0,self.aspa-1], denoting the action of each arm
        :return: intantaneous reward of this time step
        """
        # choose a more efficient way to compute
        if self.N > self.num_types * self.sspa_size * self.aspa_size:
            logging.debug("using typed het update method")
            tsa2indices = {} # each key is a tuple (type, state, action), whose value is the list of indices of such arms
            # find out the arms whose (state, action) = sa_pair
            for tsa in self.tsa_tuples:
                tsa2indices[tsa] = np.where(np.all([self.id2types == tsa[0], self.states == tsa[1], actions == tsa[2]], axis=0))[0]
            instant_reward = 0
            for tsa in self.tsa_tuples:
                next_states = np.random.choice(self.sspa, len(tsa2indices[tsa]), p=self.trans_tensor[tsa[0], tsa[1], tsa[2],:])
                self.states[tsa2indices[tsa]] = next_states
                instant_reward += self.reward_tensor[tsa[0], tsa[1], tsa[2]] * len(tsa2indices[tsa])
            instant_reward = instant_reward / self.N  # normalize the total reward by the number of arms
            return instant_reward
        else:
            logging.debug("using fully het update method")
            instant_reward = 0
            for i in range(self.N):
                cur_s = self.states[i]
                cur_a = actions[i]
                instant_reward += self.reward_tensor[self.id2types[i], cur_s, cur_a]
                self.states[i] = np.random.choice(self.sspa, 1, p=self.trans_tensor[self.id2types[i], cur_s, cur_a, :])
            instant_reward = instant_reward / self.N  # normalize by the number of arms
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
    def __init__(self, sspa_size, aspa_size, N, policies, K, cost_tensor_list, alpha_list, permuted_orders):
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
        self.num_types = self.policies.shape[0]
        self.K = K
        self.cost_tensor_list = cost_tensor_list
        self.alpha_list = alpha_list
        self.permuted_orders = permuted_orders # the permuted order under which ID-based prioritization is applied
        self.EPS = 1e-8

        # check the dimensions
        for cost_tensor in cost_tensor_list:
            assert cost_tensor.shape == (self.num_types, self.sspa_size, self.aspa_size)
        for alpha in self.alpha_list:
            assert (type(alpha) == float) or (len(alpha.shape) == 0)
        # check that the single-armed policy is a valid set of conditional distributions
        if not np.allclose(np.sum(self.policies, axis=2), np.ones((self.num_types, self.sspa_size)), atol=1e-4):
            print("policy definition wrong, the action probs do not sum up to 1. The wrong arms and policies are")
            for j in range(self.num_types):
                if not np.allclose(np.sum(self.policies[j,:,:], axis=1), np.ones((self.sspa_size,)), atol=1e-4):
                    print("j={}, pibar_j={}".format(j, self.policies[j,:,:]))
            raise ValueError

        # precompute possible combinations of type and states
        self.ts_pairs = []
        for t_ind in range(self.num_types):
            for s in self.sspa:
                self.ts_pairs.append((t_ind, s))

    def get_actions(self, id2types, cur_states):
        """
        :param id2types: length-N list of integers, each denoting the type of the n-th arm
        :param cur_states: the current states of the arms
        :return: the actions taken by the arms under the policy
        """
        # randomly sample actions using the single-armed policies
        actions = np.zeros((self.N,), dtype=int)
        # choose a more efficient way to sample the ideal actions
        if self.N > self.num_types * self.sspa_size * self.aspa_size:
            for ts in self.ts_pairs:
                cur_ts_indices =  np.where(np.all([id2types==ts[0], cur_states==ts[1]], axis=0))[0]
                actions[cur_ts_indices] = np.random.choice(self.aspa, size=len(cur_ts_indices), p=self.policies[ts[0], ts[1],:])
        else:
            for i in range(self.N):
                actions[i] = np.random.choice(self.aspa, size=1, p=self.policies[id2types[i], cur_states[i], :])

        # permute the states according to self.permuted_orders
        permuted_states = cur_states[self.permuted_orders]
        permuted_actions = actions[self.permuted_orders]

        # compute partial sums of cost consumption under the permuted IDs
        cost_partial_sum_table = np.zeros((self.K, self.N,))
        for k in range(self.K):
            types_under_permuted_orders = id2types[self.permuted_orders]
            permuted_cost_tensor = self.cost_tensor_list[k][types_under_permuted_orders,:,:] # a tensor of shape (N,S,A)
            for i in range(self.N):
                if i == 0:
                    cost_partial_sum_table[k,i] = permuted_cost_tensor[i, permuted_states[i], permuted_actions[i]]
                else:
                    cost_partial_sum_table[k,i] = cost_partial_sum_table[k,i-1] + permuted_cost_tensor[i, permuted_states[i], permuted_actions[i]]
        # find a maximal subset of arms that can follow the single-armed policy
        budget_vec = self.N * np.array(self.alpha_list)
        conform_table = cost_partial_sum_table <= np.expand_dims(budget_vec, axis=1)
        conform_arms_vec = np.prod(conform_table, axis=0)
        non_conform_arms_list = np.where(conform_arms_vec==0)[0]
        N_star = np.min(non_conform_arms_list) if len(non_conform_arms_list) > 0 else self.N # i < N_star follow the single-armed policy
        # rectify permuted actions accordingly
        permuted_actions[N_star:] = 0

        # checking budget conformity
        for k in range(self.K):
            types_under_permuted_orders = id2types[self.permuted_orders]
            permuted_cost_tensor = self.cost_tensor_list[k][types_under_permuted_orders,:,:]
            assert sum([permuted_cost_tensor[i, permuted_states[i], permuted_actions[i]] for i in range(self.N)]) <= budget_vec[k], "{}-th cost exceeds budget"

        # map the change back to the actions in the original order
        actions[self.permuted_orders] = permuted_actions

        return actions, N_star



