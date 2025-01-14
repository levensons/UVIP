import numpy as np
import pandas as pd
import scipy.stats as stats

def ma(series, window_size):
    return np.convolve(series, np.ones(window_size)/window_size, mode='valid')

def find_discord(relative_err, timestamps, threshold=0.01):
    for i in range(len(relative_err)):
        if relative_err[i] < threshold:
            return relative_err[i], timestamps[i]

    return relative_err[-1], np.inf

def process_data(exp_path, threshold=0.01):
    relative_err = np.load(exp_path + "relative_err_hist.npy")
    timestamps = np.load(exp_path + "timestamps.npy")

    window_size = 5
    ma_err = ma(relative_err, window_size)

    # print(rewards_agent_e)
    # print(timestamps)
    # print(ma(relative_err, window_size))
    # print(values)
    # print(np.mean((rewards_agent_e - values)**2))
    # print(Vpi)
    err, timestamp = find_discord(ma_err, timestamps, threshold)
    return err, timestamp

def main():
    seeds = [3, 9, 33, 42, 1812]
    n_iters = [1250, 2500, 5000]
    threshold = 0.01

    d_err = {"seed": [], "n_iter": [], "relative_err": []}
    d_timestamp = {"seed": [], "n_iter": [], "timestamp": []}
    # seeds = [3]
    # n_iters = [2500]

    for n_iter in n_iters:
        for seed in seeds:
            exp_path = f"./TwinRoomsExp{n_iter}_{seed}/"
            # if n_iter == 2500:
            #     threshold = 0.015
            # else:
            #     threshold = 0.01

            err, timestamp = process_data(exp_path, threshold)
            print(f"seed={seed}, n_iter={n_iter}, relative_err={err}, timestamp={timestamp}")
            d_err["relative_err"].append(err)
            d_err["seed"].append(seed)
            d_err["n_iter"].append(n_iter)
            d_timestamp["timestamp"].append(timestamp)
            d_timestamp["seed"].append(seed)
            d_timestamp["n_iter"].append(n_iter)

    d_err_df = pd.DataFrame(d_err)
    d_timestamp_df = pd.DataFrame(d_timestamp)
    d_err_df_pivot = d_err_df.pivot(index="n_iter", columns="seed", values="relative_err")
    d_timestamp_df_pivot = d_timestamp_df.pivot(index="n_iter", columns="seed", values="timestamp")

    print(d_err_df_pivot)
    print(d_timestamp_df_pivot)


    d_timestamp_df_pivot_filtered = d_timestamp_df_pivot.where(d_timestamp_df_pivot < 200)
    print(d_timestamp_df_pivot_filtered)

    mean_ts = d_timestamp_df_pivot.mean(axis=1)
    std_ts = d_timestamp_df_pivot.std(axis=1, ddof=1)

    print(mean_ts)
    # print(std_ts)

    std_error = std_ts / np.sqrt(5)
    print(stats.t.ppf(0.975, 4) * std_error)
    # print(mean_ts - stats.t.ppf(0.975, 4) * std_error, mean_ts + stats.t.ppf(0.975, 4) * std_error)

    mean_ts_filtered = d_timestamp_df_pivot_filtered.mean(axis=1, skipna=True)
    std_ts_filtered = d_timestamp_df_pivot_filtered.std(axis=1, ddof=1, skipna=True)

    print(mean_ts_filtered)
    # print(std_ts_filtered)

    std_error = std_ts_filtered / np.sqrt(5)
    print(stats.t.ppf(0.975, 4) * std_error)
    # print(mean_ts_filtered - stats.t.ppf(0.975, 4) * std_error, mean_ts_filtered + stats.t.ppf(0.975, 4) * std_error)

if __name__ == '__main__':
    main()