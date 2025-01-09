from matplotlib import pyplot as plt
from wcmdp import *
import examples
from examples import RandomExampleFullyHet
import pickle
import os
import bisect



def make_figure_from_multiple_files_flexible_N(note=None):
    """
    Plotting function that reads data files with Ns to combine into one plot.
    """
    settings = ["uniform-S10A4N1000K3fh-0"]
    policies = ["id"]
    linestyle_str = ["-.", "-", "--", "-.", "--", "-", "-", "-."]
    policy_markers = ["v",".","^","s","p","*", "v", "P"]
    policy_colors = ["m","c","y","r","g","b", "brown", "orange"]
    policy2label = {"id":"ID policy"}
    plot_CI = True
    batch_size_mode = "fixed" #"adaptive" #"fixed" or "adaptive"
    batch_size = 1000 # only if batch_size_mode = "fixed"
    burn_in_batch = 1
    N_range = [100, 1000] # closed interval
    init_method = "random"
    mode = "opt-ratio"
    file_dirs = ["fig_data"]

    all_batch_means = {}
    for setting_name in settings:
        for policy_name in policies:
            # look for the figure data files whose names match with setting or policies to be plotted
            file_prefix = "{}-{}".format(setting_name, policy_name)
            file_paths = []
            for file_dir in file_dirs:
                if note is None:
                    file_paths.extend([os.path.join(file_dir,file_name) for file_name in os.listdir(file_dir)
                                  if file_name.startswith(file_prefix) and (init_method in file_name.split("-")[(-2):])])
                else:
                    file_paths.extend([os.path.join(file_dir,file_name) for file_name in os.listdir(file_dir)
                                  if file_name.startswith(file_prefix) and (init_method in file_name.split("-")[(-2):])
                                  and (note in file_name.split("-")[(-1):])])
            print("{}:{}".format(file_prefix, file_paths))
            if len(file_paths) == 0:
                raise FileNotFoundError("no file that match the prefix {} and init_method = {}".format(file_prefix, init_method))
            # load the data and calculate the batch means
            N2batch_means = {}
            upper_bound_dict = {}
            N_longest_T = {} # only plot with the longest T; N_longest_T helps identifying the file with longest T
            for file_path in file_paths:
                with open(file_path, 'rb') as f:
                    setting_and_data = pickle.load(f)
                    for N in setting_and_data["Ns"]:
                        if (N < N_range[0]) or (N > N_range[1]):
                            continue
                        if (0, N) not in setting_and_data["full_reward_trace"]:
                            print("N={} not available in {}".format(N, file_path))
                            continue
                        if N not in N2batch_means:
                            N2batch_means[N] = []
                            N_longest_T[N] = 0
                        if N_longest_T[N] > setting_and_data["T"]:
                            continue
                        else:
                            if "save_mean_every" in setting_and_data:
                                save_mean_every = setting_and_data["save_mean_every"]
                            else:
                                save_mean_every = 1
                            if N_longest_T[N] == setting_and_data["T"]:
                                continue #### only use one batch of data with largest horizon; comment out otherwise
                                # print(setting_name, N, "appending data from ", file_path)
                            else:
                                N_longest_T[N] = setting_and_data["T"]
                                N2batch_means[N] = []
                                print(setting_name, N, "replaced with data from ", file_path)
                            if batch_size_mode == "adaptive":
                                batch_size = round(N_longest_T[N] / 20)
                                print("batch_size=", batch_size)
                            assert batch_size % save_mean_every == 0, "batch size is not a multiple of save_mean_every={}".format(save_mean_every)
                            for t in range(round(batch_size / save_mean_every)*burn_in_batch, round(setting_and_data["T"] / save_mean_every), round(batch_size / save_mean_every)):
                                N2batch_means[N].append(np.mean(setting_and_data["full_reward_trace"][(0,N)][t:(t+round(batch_size / save_mean_every))]))
                        upper_bound_dict[N] = setting_and_data["upper_bound_dict"][N]
            for N in N2batch_means:
                N2batch_means[N] = np.array(N2batch_means[N])
            all_batch_means[(setting_name,policy_name)] = N2batch_means
            all_batch_means[(setting_name,"upper_bound_dict")] = upper_bound_dict


    for setting_name in settings:
        upper_bound_dict = all_batch_means[(setting_name,"upper_bound_dict")]
        if mode == "opt-ratio":
            # plot the upper bound for optimality gap ratio, which is 1
            plt.plot([N_range[0], N_range[1]], np.array([1, 1]), label="Upper bound", linestyle="--", color="k")
        for i, policy_name in enumerate(policies):
            Ns_local = []
            avg_rewards_local = []
            yerrs_local = []
            cur_policy_batch_means = all_batch_means[(setting_name, policy_name)]
            for N in cur_policy_batch_means:
                Ns_local.append(N)
                avg_rewards_local.append(np.mean(cur_policy_batch_means[N]))
                yerrs_local.append(1.96 * np.std(cur_policy_batch_means[N]) / np.sqrt(len(cur_policy_batch_means[N])))
            Ns_local = np.array(Ns_local)
            avg_rewards_local = np.array(avg_rewards_local)
            yerrs_local = np.array(yerrs_local)
            sorted_indices = np.argsort(Ns_local)
            Ns_local_sorted = Ns_local[sorted_indices]
            avg_rewards_local_sorted = avg_rewards_local[sorted_indices]
            yerrs_local_sorted = yerrs_local[sorted_indices]
            print(setting_name, policy_name, avg_rewards_local_sorted, yerrs_local_sorted)

            if not plot_CI:
                # plot the figure without CI
                if mode == "opt-ratio":
                    plt.plot(Ns_local_sorted, avg_rewards_local_sorted / upper_bound_dict[N],
                                         label=policy2label[policy_name], linewidth=1.5, linestyle=linestyle_str[i],
                                            marker=policy_markers[i], markersize=8, color=policy_colors[i])
                else:
                    raise NotImplementedError
            else:
                # plot the figure with CI
                if mode == "opt-ratio":
                    plt.errorbar(Ns_local_sorted, avg_rewards_local_sorted / upper_bound_dict[N],
                                 yerr=yerrs_local_sorted / upper_bound_dict[N], label=policy2label[policy_name],
                                 linewidth=1.5, linestyle=linestyle_str[i], marker=policy_markers[i], markersize=8,
                                 color=policy_colors[i])
                else:
                    raise NotImplementedError

        # figure properties
        plt.xlabel("N", fontsize=14)
        plt.xticks(fontsize=14)
        if mode == "opt-ratio":
            plt.ylabel("Optimality ratio", fontsize=14)
        else:
            raise NotImplementedError
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.grid()
        plt.legend(fontsize=14)
        # save the figure
        if mode == "opt-ratio":
            plt.savefig("figs/{}-N{}-{}-{}.png".format(setting_name, N_range[0], N_range[1], init_method))
        else:
            raise NotImplementedError
        plt.show()


if __name__ == "__main__":
    if not os.path.exists("figs"):
        os.mkdir("figs")

    make_figure_from_multiple_files_flexible_N()
