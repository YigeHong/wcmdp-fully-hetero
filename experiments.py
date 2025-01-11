import numpy as np
import cvxpy as cp
import scipy

from wcmdp import *
import examples
from examples import RandomExampleFullyHet, RandomExampleTypedHet
import time
import pickle
import os
import multiprocessing as mp
import bisect


def run_policies(setting_name, policy_name, init_method, T, setting_path=None, Ns=None, skip_N_below=None, no_run=False, debug=False, note=None):
    if setting_path is not None:
        with open(setting_path, "rb") as f:
            setting = pickle.load(f)
    else:
        raise NotImplementedError
    print(setting_name)
    print("Ns = ", Ns)
    for N in Ns:
        # ensure N is a multiple of 20 so that alpha N is an integer
        assert N % 20 == 0
    # some hyperparameters
    num_reps = 1
    save_mean_every = 1000
    setting.print()
    print()

    if no_run:
        return

    if note is None:
        data_file_name = "fig_data/{}-{}-N{}-{}-{}".format(setting_name, policy_name, Ns[0], Ns[-1], init_method)
    else:
        data_file_name = "fig_data/{}-{}-N{}-{}-{}-{}".format(setting_name, policy_name, Ns[0], Ns[-1], init_method, note)
    if os.path.exists(data_file_name):
        # check the meta-data of the file that we want to overwrite
        with open(data_file_name, "rb") as f:
            setting_and_data = pickle.load(f)
            assert setting_and_data["num_reps"] == num_reps
            assert setting_and_data["T"] == T
            assert setting_and_data["Ns"] == Ns
            assert setting_and_data["setting_name"] == setting_name
            assert setting_and_data["policy_name"] == policy_name
            assert setting_and_data["init_method"] == init_method
            if "save_mean_every" in setting_and_data:
                assert setting_and_data["save_mean_every"] == save_mean_every
            else:
                assert save_mean_every == 1
            assert ("full_reward_trace" in setting_and_data) and ("reward_array" in setting_and_data)
    else:
        setting_and_data = {"num_reps": num_reps,
                            "T": T,
                            "Ns": Ns,
                            "setting_name": setting_name,
                            "policy_name": policy_name,
                            "init_method": init_method}

    tic = time.time()
    avg_reward_array = {}
    full_reward_trace = {}
    upper_bound_dict = {}
    if policy_name == "id":
        new_orders_dict = {}
    for N in Ns:
        if (skip_N_below is not None) and (N <= skip_N_below):
            continue
        sspa_size = setting.sspa_size
        aspa_size = setting.aspa_size
        trans_tensor_N = setting.get_trans_tensor(N)  # shape = (num_types, ...); when fully heterogeneous, num_types=N
        reward_tensor_N = setting.get_reward_tensor(N)
        cost_tensor_list_N = setting.get_cost_tensor_list(N)
        id2types_N = setting.get_type_vec(N)
        type_fracs_N = setting.get_actual_type_fracs(N)
        num_types = len(type_fracs_N)
        alpha_list = setting.alpha_list
        K = setting.K
        tic_lp = time.time()
        analyzer = SingleArmAnalyzer(sspa_size=sspa_size, aspa_size=aspa_size, N=N, type_fracs=type_fracs_N,
                                     trans_tensor=trans_tensor_N, reward_tensor=reward_tensor_N, K=K,
                                     cost_tensor_list=cost_tensor_list_N, alpha_list=alpha_list)
        opt_value, y = analyzer.solve_lp()
        # for j in range(num_types):
        #     analyzer.print_LP_solution(j)
        # analyzer.print_LP_solution(-1)
        toc_lp = time.time()
        print("Time for solving LP = {}, |S|={}, |A|={}, N={}, num_types={}".format(toc_lp-tic_lp, sspa_size, aspa_size, N, num_types))
        single_armed_policies = analyzer.policies
        upper_bound_dict[N] = opt_value
        print("Reward upper bound from LP = {}".format(opt_value))
        # preprocessing steps for each policy
        if policy_name == "id":
            # if policy is id policy, reassign the ID, or obtain the existing reassigned order from the file
            if ("new_orders_dict" in setting_and_data) and (N in setting_and_data["new_orders_dict"]):
                new_orders = setting_and_data["new_orders_dict"][N]
            else:
                tic_reassign = time.time()
                new_orders = analyzer.reassign_ID(method="intervals")
                toc_reassign = time.time()
                print("Time for reassign ID = {}".format(toc_reassign-tic_reassign))
            new_orders_dict[N] = new_orders
            print("new_orders= ", new_orders)
        elif policy_name == "lpindex":
            priority_list = analyzer.solve_LP_Priority()
            fluid_active_ts = []
            fluid_neutral_ts = []
            fluid_passive_ts = []
            fluid_null_ts = []
            for j in range(num_types):
                for s in range(sspa_size):
                    if (y[j,s,1] > 1e-4) and (y[j,s,0] <= 1e-4):
                        fluid_active_ts.append((j,s))
                    elif (y[j,s,1] > 1e-4) and (y[j,s,0] > 1e-4):
                        fluid_neutral_ts.append((j,s))
                    elif (y[j,s,1] <= 1e-4) and (y[j,s,0] > 1e-4):
                        fluid_passive_ts.append((j,s))
                    else:
                        fluid_null_ts.append((j,s))
            print("fluid active (type, state)-pairs", fluid_active_ts)
            print("fluid neutral (type, state)-pairs", fluid_neutral_ts)
            print("fluid passive (type, state)-pairs", fluid_passive_ts)
            print("fluid null (type, state)-pairs", fluid_null_ts)
            print("priority_list", priority_list)
            assert set(priority_list[0:len(fluid_active_ts)]) == set(fluid_active_ts), "the priority list does not seem to prioritize fluid active states"
        else:
            raise NotImplementedError
        # simulation loops
        for rep in range(num_reps):
            full_reward_trace[(rep, N)] = []
            # generate the initial states
            if init_method == "random":
                init_states_N = np.random.choice(np.arange(0, setting.sspa_size), N, replace=True)
            elif init_method == "same":
                init_states_N = np.zeros((N,))
            else:
                raise NotImplementedError
            # initialize the simulation for this N and replication
            wcmdp = WCMDP(sspa_size=sspa_size, aspa_size=aspa_size, N=N, trans_tensor=trans_tensor_N,
                          reward_tensor=reward_tensor_N, id2types=id2types_N, init_states=init_states_N)
            # simulations for each policy
            if policy_name == "id":
                # define the id policy
                policy = IDPolicy(sspa_size=sspa_size, aspa_size=aspa_size, N=N, policies=single_armed_policies, K=K,
                                  cost_tensor_list=cost_tensor_list_N, alpha_list=alpha_list, permuted_orders=new_orders)
                # start simulations loop
                total_reward = 0
                recent_total_reward = 0
                recent_total_Nstar = 0
                for t in range(T):
                    cur_states = wcmdp.get_states()
                    actions, N_star = policy.get_actions(id2types=id2types_N, cur_states=cur_states)
                    instant_reward = wcmdp.step(actions)
                    total_reward += instant_reward
                    recent_total_reward += instant_reward
                    recent_total_Nstar += N_star
                    if (t+1)%save_mean_every == 0:
                        full_reward_trace[(rep, N)].append(recent_total_reward / save_mean_every)
                        print("t={}, recent_average_reward/upper_bound={} recent_average_Nstar/N={}".format(
                            t, recent_total_reward/save_mean_every/opt_value, recent_total_Nstar/save_mean_every/N))
                        recent_total_reward = 0
                        recent_total_Nstar = 0
            elif policy_name == "lpindex":
                # define the policy
                policy = PriorityPolicy(sspa_size=sspa_size,num_types=num_types,priority_list=priority_list,N=N,alpha=alpha_list[0])
                # start simulations loop
                total_reward = 0
                recent_total_reward = 0
                for t in range(T):
                    cur_states = wcmdp.get_states()
                    actions = policy.get_actions(id2types=id2types_N, cur_states=cur_states)
                    instant_reward = wcmdp.step(actions)
                    total_reward += instant_reward
                    recent_total_reward += instant_reward
                    if (t+1)%save_mean_every == 0:
                        full_reward_trace[(rep, N)].append(recent_total_reward / save_mean_every)
                        print("t={}, recent_average_reward/upper_bound={}".format(
                            t, recent_total_reward/save_mean_every/opt_value))
                        recent_total_reward = 0
            else:
                raise NotImplementedError
            avg_reward = total_reward / T
            avg_reward_array[(rep, N)] = avg_reward
            print("setting={}, policy={}, N={}, rep_id={}, avg_reward/upper_bound={}, total gap={}, note={}".format(
                setting_name, policy_name, N, rep, avg_reward/opt_value, N*(opt_value-avg_reward), note))

            # save the data
            if not debug:
                if os.path.exists(data_file_name):
                    # write the data in place; overwrite those traces with the same (rep, N)
                    with open(data_file_name, 'rb') as f:
                        setting_and_data = pickle.load(f)
                        setting_and_data["reward_array"][(rep, N)] = avg_reward
                        setting_and_data["full_reward_trace"][(rep, N)] = full_reward_trace[(rep, N)].copy()
                        setting_and_data["upper_bound_dict"][N] = upper_bound_dict[N]
                        if policy_name == "id":
                            setting_and_data["new_orders_dict"][N] = new_orders_dict[N]
                else:
                    setting_and_data["setting"] = setting
                    setting_and_data["reward_array"] = avg_reward_array
                    setting_and_data["full_reward_trace"] = full_reward_trace
                    setting_and_data["y"] = y
                    setting_and_data["upper_bound_dict"] = upper_bound_dict
                    setting_and_data["save_mean_every"] = save_mean_every
                    if policy_name == "id":
                        setting_and_data["new_orders_dict"] = new_orders_dict
                with open(data_file_name, 'wb') as f:
                    pickle.dump(setting_and_data, f)
    print("time for running one policy with T={} and {} data points is {}".format(T, len(Ns), time.time()-tic))


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    np.set_printoptions(linewidth=600)
    if not os.path.exists("examples"):
        os.mkdir("examples")
    if not os.path.exists("fig_data"):
        os.mkdir("fig_data")
    for random_example_name in ["uniform-S10A2types10K1rb-0"]: #, "uniform-S10A4N1000K4fh-0"]: #"uniform-S10A4N1000K4fh-0" # "uniform-S5A3N200K3fh-0"
        Ns = list(range(100, 1100, 100))
        T = 10**4
        run_policies(random_example_name, "lpindex", "random", T=T, setting_path="examples/"+random_example_name, Ns=Ns)

