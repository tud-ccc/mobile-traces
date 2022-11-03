# Copyright (C) 2022 TU Dresden
# Licensed under the ISC license (see LICENSE.txt)
#
# Authors: Julian Robledo Mejia

import os
import math
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import *

def main():
    """Please update the variable folder with the path to real traces."""
    # instantiate trace generator
    tg = TraceGenerator()
    # generate spatio-temporal traffic scenario
    traffic_scenario = tg.set_traffic_model()
    # generate traces
    tg.generate_traces(traffic_scenario)

    # generate plots
    tg.plot_traffic_model(traffic_scenario)
    traffic_scenario = tg.load_traffic_model("traffic_output/output_traffic.csv")
    tg.plot_traffic_model(traffic_scenario)


class TraceGenerator:
    """Random LTE traces generator.

    This function generates multiple output files containing a set of LTE
    subframes at different times of the day. The files follow the following pattern:
    each subframe contains a  random number of UE with random modulation scheme,
    and number of prbs. The random parameters above, follow a probability
    function extracted from real LTE traces.

    However, the number of UEs per subframe is controlled in order to fit the
    traffic model presented in https://arxiv.org/pdf/1703.10804.pdf
    """

    def __init__(self):
        # define LTE parameters
        self.num_sc_per_prb = 12
        self.num_slots = 2
        self.num_symbols = 7
        self.subframe_time = 0.001  # 1ms

        # make output folder
        self.path = os.path.join("traffic_output/")
        try:
            os.mkdir(self.path)
        except OSError as error:
            pass


    def set_traffic_model(self):
        """
        Set geographic location of base station by setting amplitud and phase of the
        waveform.
        """
        # create time vector 0-24h with step 0.001
        time = pd.Series(np.arange(0, 24, 0.001))

        # Amplitude parameters were escalated to fit in max of 12Mbits
        traffic = pd.Series(
            7.24
            + 4.60 * np.sin((np.pi / 12) * time + 3.10)
            + 1.99 * np.sin((np.pi / 6) * time + 2.36)
        )

        # create dataframe
        traffic = pd.DataFrame({"time": time, "traffic": traffic, "area": "Park"})

        # add noise to the signal based on normal probability
        noise = np.random.normal(0, 0.5, int(time.size))
        random_index = np.random.randint(0, time.size, int(time.size * 0.5))
        new_values = traffic["traffic"].iloc[random_index] + noise[random_index]
        traffic["traffic"].iloc[random_index] = new_values

        # save file
        traffic.to_csv("traffic_output/traffic_scenario.csv",index=False)

        return traffic


    def load_traffic_model(self, path):
        """
        load traffic scenario from existing file
        """
        traffic = pd.read_csv(path)
        return traffic


    def generate_traces(self, traffic_model):
        """Generate new random traces at different hours of the day. A set of
        traces is generated for each of the different indicated times of the day.
        The list of desired hours of interest for trace generation is given by
        <timestep> array. It can be modified by changing the following parameters:

        start_time: first element of the list
        stop_time: last element of the list
        step_time: add intermediate hours with the desired step time

        The default behavior is a one element array with the time at which
        the real traces are placed in the traffic model. Moreover, one could set
        the number of milliseconds of traffic to be generated for every hour.

        nsubframes: Number of milliseconds of traffic to be generated for each
        specified hour.
        """
        # mean traffic per BS
        mean_traffic_per_bs = 6.4702734045567425

        # load probabilities
        prob_prbs = pd.read_csv("prob_prbs.csv")
        prob_mods = pd.read_csv("prob_mods.csv")
        prob_ues = pd.read_csv("prob_ues.csv")
        prob_bs = pd.read_csv("prob_bs.csv")

        # set list of hours
        start_time = 0.0
        stop_time = 24.0
        step = 0.1
        nsubframes = 1000  # 1 seg

        # max of subframes that fit before starting the next step
        max_subframes = step * 3600000  # convert to ms
        if nsubframes > max_subframes:
            nsubframes = max_subframes

        # start subframe generation
        df_list = list()
        timestep = np.arange(start_time, stop_time, step)
        for target_time in timestep:

            # get the target traffic at the target time
            index_of_min = (abs(traffic_model["time"] - target_time)).idxmin()
            approx_time_of_traffic = traffic_model["time"][index_of_min]
            index = traffic_model.index[traffic_model["time"] == approx_time_of_traffic].tolist()
            target_traffic_per_bs = traffic_model.iloc[index[0]]["traffic"]

            # Generate a num of subframes with a random number of UEs
            offset = target_traffic_per_bs / mean_traffic_per_bs
            num_ues = np.random.choice(prob_ues['x'], size=nsubframes, replace=True, p=prob_ues['probs'])
            num_ues = num_ues * offset
            num_ues = np.round(num_ues).astype(int)
            new_traces = np.repeat(np.arange(1, nsubframes + 1), num_ues)
            new_traces = pd.DataFrame(new_traces, columns=["subframe"],)

            # generate random base stations ids
            new_traces["bs"] = np.random.choice(prob_bs['x'], size=sum(num_ues), replace=True, p=prob_bs['probs'])

            # Add column with enumerated UEs
            temp = new_traces.groupby(["subframe"])["bs"].count()
            temp = np.hstack([np.arange(1, temp.max() + 1)[:k] for k in temp])
            new_traces["crnti"] = pd.DataFrame(temp, columns=["bs"],)

            # Generate number of PRBs
            new_traces["prbs"] = np.random.choice(prob_prbs['x'], size=len(new_traces.index), replace=True, p=prob_prbs['probs'])

            # Generate modulation scheme
            new_traces["mod"] = np.random.choice(prob_mods['x'], size=len(new_traces.index), replace=True, p=prob_mods['probs'])

            # calculate total traffic
            new_traces["traffic"] = (
                new_traces["prbs"]
                * self.num_sc_per_prb
                * self.num_symbols
                * new_traces["mod"]
                * self.num_slots
                / self.subframe_time
            )
            new_traces["traffic"] = new_traces["traffic"] / 1000000  # convert to Mb

            # add timing info to dataframe
            new_traces["time"] = target_time
            df_list.append(new_traces)

        # print traces of a target base station to external file
        target_bs = 0
        mean_traffic_list = list()
        for df in df_list:
            # select target base station
            target_time = df.loc[0]["time"]
            target_hour = math.floor(target_time)
            target_min = math.floor((target_time % 1) * 100)
            df_bs1 = df.query("bs == @target_bs")
            df_bs = df_bs1.drop(columns=["traffic", "time"])

            # fill empty subframes with NaN
            num_bs = (
                df_bs.groupby(["subframe"])["bs"]
                .count()
                .reset_index(name="num_bs")
            )
            num_bs = (
                num_bs.set_index("subframe")
                .reindex(range(1, df_bs["subframe"].max() + 1))
                .fillna(0)
                .reset_index()
                .astype(int)
            )
            num_bs = num_bs.query("num_bs == 0").loc[:, ["subframe"]]

            # save file
            file_name = f"traffic_output/traces_{target_hour}_{target_min}.csv"
            df_bs = pd.concat([df_bs, num_bs], axis=0, ignore_index=True)
            df_bs = df_bs.sort_values(["subframe"]).reset_index(drop=True)
            df_bs = df_bs.convert_dtypes()  # cast to int
            df_bs.to_csv(file_name,index=False)

            # calculate mean_traffic
            df_mean_traffic_per_bs = (df_bs1["traffic"].sum())
            df_mean_traffic_per_bs = df_mean_traffic_per_bs / nsubframes
            mean_traffic_list.append(df_mean_traffic_per_bs)

        # save file
        total = pd.DataFrame({"time": timestep, "traffic": mean_traffic_list})
        total.to_csv("traffic_output/output_traffic.csv", index=False)


    def plot_traffic_model(self, traffic):
        tm = (
                ggplot(traffic)
                + geom_line(aes(x="time", y="traffic"), alpha=0.7)
                + labs(x="hour", y="Traffic (Mb)", title="BS Traffic model")
        )
        print(tm)


if __name__ == "__main__":
    main()
