import numpy as np
import cvxpy as cp
import scipy

from wcmdp import *
import examples
from examples import RandomExampleFullyHet
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
        if setting.typed_het == False:
            # obtain the parameters
            sspa_size = setting.sspa_size
            aspa_size = setting.aspa_size
            trans_tensor = setting.get_trans_tensor(N)
            reward_tensor = setting.get_reward_tensor(N)
            cost_tensor_list = setting.get_cost_tensor_list(N)
            alpha_list = setting.alpha_list
            K = setting.K
            # solve LP, obtain single-armed policies
            tic_lp = time.time()
            analyzer = SingleArmAnalyzer(sspa_size, aspa_size, N, trans_tensor, reward_tensor, K, cost_tensor_list, alpha_list)
            opt_value, y = analyzer.solve_lp()
            toc_lp = time.time()
            print("Time for solving LP = {}, with fully Heterogeneous arms, |S|={}, |A|={} and N={}".format(toc_lp-tic_lp, sspa_size, aspa_size, N))
            single_armed_policies = analyzer.policies
            print("Reward upper bound from LP = {}".format(opt_value))
            upper_bound_dict[N] = opt_value
        else:
            raise NotImplementedError
        if policy_name == "id":
            if ("new_orders_dict" in setting_and_data) and (N in setting_and_data["new_orders_dict"]):
                new_orders = setting_and_data["new_orders_dict"][N]
            else:
                tic_reassign = time.time()
                new_orders = analyzer.reassign_ID(method="random") ### todo: use method="intervals" after fixing it
                toc_reassign = time.time()
                print("Time for reassign ID = {}".format(toc_reassign-tic_reassign))
                new_orders_dict[N] = new_orders
            print("new_orders= ", new_orders)
        # simulation loops
        for rep in range(num_reps):
            full_reward_trace[(rep, N)] = []
            if init_method == "random":
                init_states = np.random.choice(np.arange(0, setting.sspa_size), N, replace=True)
            elif init_method == "same":
                init_states = np.zeros((N,))
            else:
                raise NotImplementedError
            if policy_name == "id":
                # instantiate WCMDP and IDPolicy; permute the dimensions according to the new_orders generated from ID reassignment
                wcmdp = WCMDP(sspa_size, aspa_size, N, trans_tensor[new_orders,:,:,:], reward_tensor[new_orders,:,:], init_states[new_orders])
                permuted_cost_tensor_list = [cost_tensor[new_orders,:,:] for cost_tensor in cost_tensor_list]
                policy = IDPolicy(sspa_size, aspa_size, N, single_armed_policies[new_orders,:,:], K, permuted_cost_tensor_list, alpha_list)
                # start simulations loop
                total_reward = 0
                recent_total_reward = 0
                recent_total_Nstar = 0
                for t in range(T):
                    cur_states = wcmdp.get_states()
                    actions, N_star = policy.get_actions(cur_states)
                    instant_reward = wcmdp.step(actions)
                    total_reward += instant_reward
                    recent_total_reward += instant_reward
                    recent_total_Nstar += N_star
                    if (t+1)%save_mean_every == 0:
                        full_reward_trace[(rep, N)].append(recent_total_reward / save_mean_every)
                        print("t={}, recent_average_reward/opt_value={} recent_average_Nstar/N={}".format(
                            t, recent_total_reward/save_mean_every/opt_value, recent_total_Nstar/save_mean_every/N))
                        recent_total_reward = 0
                        recent_total_Nstar = 0
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
    random_example_name = "uniform-S10A4N1000K4fh-0" #"uniform-S10A4N1000K3fh-0" # "uniform-S5A3N200K3fh-0"
    Ns = list(range(100, 1100, 100))
    T = 10**5
    run_policies(random_example_name, "id", "random", T=T, setting_path="examples/"+random_example_name, Ns=Ns)

