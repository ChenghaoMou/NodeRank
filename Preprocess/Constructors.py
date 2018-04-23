#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-12-12 下午7:25
# @Author  : Aries
# @Site    : 
# @File    : Constructors.py
# @Software: PyCharm


"""NodeRank.

Usage:
  Constructors.py <conf>
  Constructors.py (-h|--help)
  Constructors.py --version

Options:
  <conf>        Specify the configuration section. e.g. relationship|retweet|reply|all.
  -h --help     Show this screen.
  --version     Show version.

"""
from collections import defaultdict
from functools import partial

import os

import networkx as nx
from docopt import docopt
from tqdm import tqdm

from Preprocess.Conf import configurations


def group_list(iterable, group_by=None, squeeze=True, squeeze_by=None):
    """
    1. Group the iterable by the given function group_by;
    2. Squeeze similar groups into one group by the given function squeeze_by.
    :param iterable: Input data.
    :param group_by: Group function.
    :param squeeze: Squeeze the groups or not.
    :param squeeze_by: Squeeze function.
    :return: Groups in a dict form.
    """
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


def interval_constructor(iterable_dict, group_by=None):
    """
    Construct the intervals for each dictionary of groups.
    :param iterable_dict: Input dict of groups.
    :param group_by: Group each interval elements into groups.
    :return: Intervals and Interval index.
    """
    timestamps = sorted(iterable_dict.keys())
    interval_index = sorted(zip([0] + timestamps[:-1], timestamps[:]), key=lambda x: x[0])

    intervals = defaultdict(list)

    direction_group = partial(group_list, group_by=group_by, squeeze=False, squeeze_by=None)

    for index, interval in enumerate(interval_index):
        if index == 0:
            intervals[interval] = direction_group(iterable_dict[interval[1]])
        else:
            intervals[interval] = direction_group(
                intervals[interval_index[index - 1]]['Out'] + iterable_dict[interval[1]])

    return intervals, interval_index


def time_series(path):
    paths = []
    with open(path, "r") as input_file:
        for line in tqdm(input_file.readlines()):
            raw = line.split()
            paths.append(map(int, raw[:-1]))
    g = group_list(paths, group_by=lambda x: x[-1], squeeze=False)
    return g


class GEXFConstructor(object):

    def __init__(self, graph, ranks, graph_type="static"):
        self.graph = graph
        self.ranks = ranks
        self.type = graph_type

    def construct(self, out_put_file):

        static_head = """<?xml version="1.0" encoding="utf-8"?>
            <gexf xmlns="http://www.gexf.net/1.2draft" xmlns:viz="http://www.gexf.net/1.1draft/viz" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.gexf.net/1.2draft http://www.gexf.net/1.2draft/gexf.xsd" version="1.2">
        		<graph defaultedgetype="directed">
        			<attributes class="node"><attribute id="modularity_class" title="modularity_class" type="float"></attribute></attributes>
        			<attributes class="edge"></attributes>
        			<nodes>
            """

        dynamic_head = """<?xml version="1.0" encoding="utf-8"?>
            <gexf xmlns="http://www.gexf.net/1.2draft" xmlns:viz="http://www.gexf.net/1.1draft/viz" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.gexf.net/1.2draft http://www.gexf.net/1.2draft/gexf.xsd" version="1.2">
        		<graph graphmode="dynamic" defaultedgetype="directed">
        			<attributes class="node"><attribute id="modularity_class" title="modularity_class" type="float"></attribute></attributes>
        			<attributes class="edge"></attributes>
        			<nodes>
            """

        tail = """</edges>
        	</graph>
        </gexf>
            """

        node_template = """
                <node id="{0}" label="{0}">
                    <attvalues>
                        <attvalue for="modularity_class" value="{1:.10f}"></attvalue>
                    </attvalues>
                    <viz:size value="{1:.10f}"></viz:size>
                    <viz:position x="12" y="13"></viz:position>
                    <viz:color b="{2}" g="{3}" r="{4}"></viz:color>
                </node>
        """

        edge_template = """
                <edge id="{0}" source="{1}" target="{2}" start="{3}">
                    <attvalues></attvalues>
                </edge>
        """

        with open(out_put_file, "w+") as f:

            f.writelines(static_head if self.type == "static" else dynamic_head)

            cm = pylab.get_cmap('YlOrRd')

            for n in tqdm(self.graph.nodes(), total=nx.number_of_nodes(self.graph)):
                color = cm(self.ranks[n])
                f.writelines(node_template.format(n,
                                                  self.ranks[n],
                                                  str(int(color[0] * 255)),
                                                  str(int(color[1] * 255)),
                                                  str(int(color[2] * 255))))

            f.write("</nodes>")
            f.write("<edges>")

            for i, e in tqdm(enumerate(self.graph.edges(data='time')), total=nx.number_of_edges(self.graph)):
                f.writelines(edge_template.format(i, e[0], e[1], e[2]))

            f.writelines(tail)


class GEXFNodeConstructor(GEXFConstructor):

    def __init__(self, node, graph, ranks, graph_type="dynamic"):
        self.node = node
        self.graph = graph.subgraph(
            list(set(list(graph.predecessors(self.node)) + list(graph.successors(self.node)) + [node])))
        super(GEXFNodeConstructor, self).__init__(self.graph, ranks, graph_type)


class NetworkConstructor(object):

    def __init__(self, path="",
                 is_relationship=False,
                 ignore_time=False,
                 dangling=True,
                 output="",
                 lscc=True,
                 normalization=360.0,
                 min_time=1341100972,
                 edge_type="ALL",
                 ):

        assert os.path.exists(path)
        assert os.path.isfile(path)

        self.graph = nx.DiGraph()
        self.output = output
        self.lscc = lscc
        self.is_relationship = is_relationship
        self.actual_min_time = min_time

        if self.is_relationship:
            with open(path, "r") as input_file:
                for line in tqdm(input_file.readlines()):
                    raw = line.split()[:2]
                    source, target = map(int, raw)
                    self.graph.add_edge(source, target)
        else:
            self.norm = normalization
            self.min_time = min_time
            self.ignore_time = ignore_time
            self.types = {"RT": 1.0, "RE": 0.8, "DEFAULT": 0.6, "MT": 0.4}

            print("Is directed: {0}".format(nx.is_directed(self.graph)))

            with open(path, "r") as input_file:
                for line in tqdm(input_file.readlines()):
                    raw = line.split()
                    if edge_type == "ALL" or raw[-1] == edge_type:
                        try:
                            source, target, time = map(int, raw[:-1])
                            time = self.__norm(time)
                            if self.graph.has_edge(source, target):
                                time = min(time, self.graph[source][target]["time"])
                                self.graph.remove_edge(source, target)
                            self.graph.add_edge(source, target, time=time, weight=self.types[raw[-1]],
                                                type=raw[-1])
                        except ValueError as e:
                            print(e.message)

            for edge in nx.selfloop_edges(self.graph):
                self.graph.remove_edge(*edge)

            if dangling:
                nodes = list(self.graph.nodes())
                for source in tqdm(self.graph.nodes()):
                    if self.graph.out_degree(source) == 0:
                        self.graph.add_edges_from(zip([source] * len(nodes), nodes), time=0,
                                                  weight=self.types["DEFAULT"], type="DEFAULT")

        if self.lscc:
            self.largest_strongly_connected_component(self.graph)

    def largest_strongly_connected_component(self, graph):
        from graph_tool import Graph
        import graph_tool.all as gt

        largest_connected_component = Graph(directed=True)
        if not self.is_relationship:
            edge_prop_time = largest_connected_component.new_edge_property("int")
            edge_prop_type = largest_connected_component.new_edge_property("string")

        for edge in tqdm(graph.edges(data=True)):
            e = tuple(edge[:2])
            largest_connected_component.add_edge(e[0], e[1])
            if not self.is_relationship:
                edge_prop_time[e] = edge[-1]["time"]
                edge_prop_type[e] = edge[-1]["type"]

        largest_connected_component_view = gt.label_largest_component(largest_connected_component)
        largest_connected_component = gt.GraphView(largest_connected_component, vfilt=largest_connected_component_view)

        print("Total nodes {0} in largest strongly connected component.".format(
            largest_connected_component.num_vertices()))
        print("Total edges {0} in largest strongly connected component.".format(
            largest_connected_component.num_edges()))

        with open(self.output, "w+") as output_file:
            for edge in tqdm(largest_connected_component.edges()):
                if not self.is_relationship:
                    output_file.write("{0} {1} {2} {3}\n".format(edge.source(),
                                                                 edge.target(),
                                                                 edge_prop_time[edge],
                                                                 edge_prop_type[edge]))
                else:
                    output_file.write(
                        "{0} {1}\n".format(edge.source(), edge.target()))

    def construct(self):

        return self.graph

    def __norm(self, time):
        self.actual_min_time = min(self.actual_min_time, time)
        if self.ignore_time:
            return 0
        else:
            return (time - self.min_time) / self.norm


# def text2net(input_file, output):
#
#     import jieba
#     import re
#     import jieba.posseg as pseg
#     text = re.sub("(，|！|？|；|：|（|）|［|］|【|】|“|”|。|\ )+", " ", open(input_file, "r").read())
#     # words = [w for w, f in pseg.cut(text) if f.startswith("n")]
#     print(text)
#     words = text.split(" ")
#     word2idx = {w: i for i, w in enumerate(set(words))}
#     idx2word = {i: w for w, i in word2idx.items()}
#
#     with open(output, "w") as output_file:
#         for i, (w1, w2) in enumerate(zip(words[:-1], words[1:])):
#             print("{0} {1} {2}".format(w2.encode("utf-8"), w1.encode("utf-8"), i))
#             output_file.write("{0} {1} {2} DEFAULT\n".format(word2idx.get(w2), word2idx.get(w1), i))
#
#     import pickle
#     pickle.dump(idx2word, open("wordindex.dict", "w"))
#
# text2net("/home/maxen/Documents/Code/PycharmProjects/NodeRank/data/text.txt",
#                "/home/maxen/Documents/Code/PycharmProjects/NodeRank/data/text_net.txt")

# if __name__ == "__main__":
#     arguments = docopt(__doc__, version='NodeRank Constructor 1.0')
#     print(configurations.get(arguments.get('<conf>')))
#     constructor = NetworkConstructor(**configurations.get(arguments.get('<conf>')))
#     graph = constructor.construct()
#     print(constructor.actual_min_time)











