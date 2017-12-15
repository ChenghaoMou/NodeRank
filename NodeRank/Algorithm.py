#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-12-12 下午7:58
# @Author  : Aries
# @Site    : 
# @File    : Algorithm.py
# @Software: PyCharm
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import networkx as nx
import numpy
import pandas as pd
from tqdm import tqdm

from Preprocess.Constructors import NetworkConstructor, path_constructor, group_list, interval_constructor


class NodeRank(object):

    def __init__(self, relation_network=None, dissemination_network=None):

        self.node_tables = dict()

        self.relation_graph = relation_network
        self.dissemination_graph = dissemination_network

        self.node_influence = pd.Series(0.0, index=list(self.dissemination_graph.nodes()))

    def initialize(self):
        for node in tqdm(self.dissemination_graph.nodes()):
            in_edges = map(lambda x: ["In"] + list(x), list(self.dissemination_graph.in_edges(node, data="time")))
            out_edges = map(lambda x: ["Out"] + list(x), list(self.dissemination_graph.out_edges(node, data="time")))

            intervals, column_index = interval_constructor(
                group_list(in_edges + out_edges, group_by=lambda x: x[-1], squeeze=True, squeeze_by=lambda x: x[0]))

            row_index = list(self.dissemination_graph.successors(node))

            f = pd.DataFrame(0.0, index=row_index, columns=column_index)

            for interval, edges in intervals.items():
                f.loc[[edge[2] for edge in edges["Out"]], interval] = 1.0

            f = f.apply(lambda x: 0 if not sum(x) else x / sum(x), axis=0)

            p = pd.Series(1.0 / nx.number_of_nodes(self.dissemination_graph) / len(row_index) if row_index else 0.0,
                          index=row_index, dtype=numpy.float32)
            c = pd.Series(0.0, index=column_index, dtype=numpy.float32)

            self.node_tables[node] = {
                "Intervals": intervals,
                "Index": column_index,
                "P": p,
                "F": f,
                "C": c
            }

    def run(self, iteration=2, epsilon=1e-10, damping=0.85):

        previous = self.node_influence.copy()

        for i in range(iteration):

            for node in self.dissemination_graph.nodes():
                if not self.node_tables[node]["C"].empty: self.collect(node, damping=damping)
                if not self.node_tables[node]["P"].empty: self.vote(node)
                self.node_influence[node] = self.node_tables[node]["C"].sum()

            if sum((self.node_influence - previous)) < epsilon:
                print(
                    "Finished {1}-th iteration with epsilon {0:.5f}".format(sum((self.node_influence - previous).abs()),
                                                                            i))
                break
            else:
                print(
                    "Finished {1}-th iteration with epsilon {0:.5f}".format(sum((self.node_influence - previous).abs()),
                                                                            i), end='\r')
                previous = self.node_influence.copy()

    def collect(self, node, damping=0.85):
        temp = (1 - damping)
        for interval, edges in self.node_tables[node]["Intervals"].items():
            self.node_tables[node]["C"].ix[
                self.node_tables[node]["Index"].index(interval), 0] = temp * nx.number_of_nodes(
                self.dissemination_graph) + damping * sum(
                self.node_tables[edge[1]]["P"][edge[2]] for edge in edges["In"])

    def vote(self, node):
        self.node_tables[node]["P"] = pd.Series(data=self.node_tables[node]["F"].dot(self.node_tables[node]["C"]),
                                                index=self.node_tables[node]["P"].index,
                                                dtype=numpy.float32)

    def sort(self):
        self.node_influence.sort_values(axis=0, ascending=False, inplace=True)
        temp = sum(self.node_influence)
        self.node_influence = self.node_influence.apply(lambda x: x / temp)


if __name__ == "__main__":
    dissemination_file_path = path_constructor("data/email-Eu-core-temporal-Dept1.txt")
    dissemination_graph = NetworkConstructor(dissemination_file_path, normalization=1, min_time=0,
                                             ignore_time=False).construct()
    print("Finished Construction...")
    rank = NodeRank(relation_network=None, dissemination_network=dissemination_graph)
    rank.initialize()
    rank.run(iteration=100, damping=0.85, epsilon=1e-6)
    rank.sort()

    page_ranks = nx.pagerank(dissemination_graph, alpha=0.85)
    page_ranks_index = OrderedDict()
    for index, node in enumerate(sorted(page_ranks.items(), key=lambda x: x[1], reverse=True)):
        page_ranks_index[node[0]] = (index, node[1])
    for idx, (node, value) in enumerate(rank.node_influence.iteritems()):
        print("{0:<10} ({1:<10}) {2:<10,.5f} ({3:<10}) {4:<10,.5f} {5}".format(node, idx, value, page_ranks_index[node][0], page_ranks_index[node][1],
                                                                       ("↑" if page_ranks_index[node][0] > idx else "↓") + str(abs(page_ranks_index[node][0] - idx))))
