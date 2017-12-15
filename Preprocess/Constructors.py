#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-12-12 下午7:25
# @Author  : Aries
# @Site    : 
# @File    : Constructors.py
# @Software: PyCharm
from functools import partial

import networkx as nx
import os

from collections import defaultdict
from tqdm import tqdm

path_constructor = partial(os.path.join, "/home/maxen/Documents/Code/PycharmProjects/NodeRank")


class NetworkConstructor(object):

    def __init__(self, path="", ignore_time=False, normalization=30.0, min_time=1341100972, dangling=True):

        assert os.path.exists(path)
        assert os.path.isfile(path)

        self.graph = nx.DiGraph()
        self.norm = normalization
        self.min_time = min_time
        self.ignore_time = ignore_time

        with open(path, "r") as input_file:
            for line in tqdm(input_file.readlines()):
                raw = line.split()
                if raw[-1] != "RT": continue
                source, target, time = map(int, raw[:-1])
                self.graph.add_edge(source, target, time=self.__norm(time))

        if dangling:
            nodes = list(self.graph.nodes())
            for source in self.graph.nodes():
                if self.graph.out_degree(source) == 0:
                    self.graph.add_edges_from(zip([source]*len(nodes), nodes), time=0)

    def construct(self):
        return self.graph

    def __norm(self, time):
        if self.ignore_time: return 0
        else: return (time - self.min_time) / self.norm


def group_list(iterable, group_by=None, squeeze=True, squeeze_by=None):
    result = defaultdict(list)
    for item in iterable:
        result[group_by(item)].append(item)

    if squeeze:
        keys = list(sorted(result.keys()))
        for index, key in enumerate(keys[:-1]):
            if set([squeeze_by(x) for x in result[key]]) == set(squeeze_by(x) for x in result[keys[index + 1]]):
                result[keys[index + 1]].extend(result[key])
                del result[key]

    return result


def interval_constructor(iterable_dict):
    timestamps = sorted(iterable_dict.keys())
    intervals = sorted(zip([0] + timestamps[:-1], timestamps[:]), key=lambda x: x[0])
    result = defaultdict(list)

    direction_group = partial(group_list, group_by=lambda x: x[0], squeeze=False, squeeze_by=None)

    for index, interval in enumerate(intervals):
        if index == 0:
            result[interval] = direction_group(iterable_dict[interval[1]])
        else:
            result[interval] = direction_group(result[intervals[index - 1]]['Out'] + iterable_dict[interval[1]])

    return result, intervals


if __name__ == "__main__":

    file_path = path_constructor("data/higgs-activity_time.txt")
    graph = NetworkConstructor(file_path, dangling=False).construct()

    print("{0:<10} Edges and {1:<10} Nodes.".format(graph.number_of_edges(), graph.number_of_nodes()))

    largest_weakly_connected_component = max(nx.strongly_connected_component_subgraphs(graph), key=len)

    with open(path_constructor("data/higgs-activity_time_lwcc.txt"), "w+") as output_file:
        for edge in tqdm(largest_weakly_connected_component.edges(data='time')):
            output_file.write("{0} {1} {2}\n".format(*edge))


    pass
