from functools import partial

import os

path_constructor = partial(os.path.join, "/home/maxen/Documents/Code/PycharmProjects/NodeRank")

configurations = {
    "relationship": {
        "normalization": 1,
        "min_time": 0,
        "path": path_constructor("data/higgs/higgs-activity_time.txt"),
        "lscc": True,
        "dangling": False,
        "is_relationship": True,
        "edge_type": "ALL",
        "ignore_time": True,
        "output": path_constructor("data/higgs/higgs-social_network_lscc.edgelist")
    },
    "retweet": {
        "normalization": 360,
        "min_time": 1341100972,
        "path": path_constructor("data/higgs/higgs-activity_time.txt"),
        "lscc": True,
        "dangling": False,
        "is_relationship": False,
        "edge_type": "RT",
        "ignore_time": False,
        "output": path_constructor("data/higgs/higgs-activity_time_rt_lcc.txt")
    },
    "reply": {
        "normalization": 360,
        "min_time": 1341100972,
        "path": path_constructor("data/higgs/higgs-activity_time.txt"),
        "lscc": True,
        "dangling": False,
        "is_relationship": False,
        "edge_type": "RE",
        "ignore_time": False,
        "output": path_constructor("data/higgs/higgs-activity_time_re_lcc.txt")
    },
    "mention": {
        "normalization": 360,
        "min_time": 1341100972,
        "path": path_constructor("data/higgs/higgs-activity_time.txt"),
        "lscc": True,
        "dangling": False,
        "is_relationship": False,
        "edge_type": "MT",
        "ignore_time": False,
        "output": path_constructor("data/higgs/higgs-activity_time_mt_lcc.txt")
    },
    "all": {
        "normalization": 360,
        "min_time": 1341100972,
        "path": path_constructor("data/higgs/higgs-activity_time.txt"),
        "lscc": True,
        "dangling": False,
        "is_relationship": False,
        "edge_type": "ALL",
        "ignore_time": False,
        "output": path_constructor("data/higgs/higgs-activity_time_all_lcc.txt")
    },
    # "all": {
    #     "normalization": 1,
    #     "min_time": 0,
    #     "path": path_constructor("data/higgs/higgs-activity_time_all_lcc.txt"),
    #     "lscc": False,
    #     "dangling": False,
    #     "is_relationship": False,
    #     "edge_type": "ALL",
    #     "ignore_time": True,
    #     "output": path_constructor("data/higgs/higgs-activity_time_all_lcc.txt")
    # },
}

running_configurations = {
    "relationship": {
        "normalization": 1,
        "min_time": 0,
        "path": path_constructor("data/higgs/higgs-social_network_lscc.edgelist"),
        "lscc": False,
        "dangling": True,
        "is_relationship": True,
        "edge_type": "ALL",
        "ignore_time": True,
        "output": path_constructor("data/higgs/higgs-social_network_lscc.edgelist")
    },
    "retweet": {
        "normalization": 1,
        "min_time": 0,
        "path": path_constructor("data/higgs/higgs-activity_time_rt_lcc.txt"),
        "lscc": False,
        "dangling": True,
        "is_relationship": False,
        "edge_type": "RT",
        "ignore_time": False,
        "output": path_constructor("data/higgs/higgs-activity_time_rt_lcc.txt")
    },
    "mention": {
        "normalization": 1,
        "min_time": 0,
        "path": path_constructor("data/higgs/higgs-activity_time_mt_lcc.txt"),
        "lscc": False,
        "dangling": True,
        "is_relationship": False,
        "edge_type": "MT",
        "ignore_time": False,
        "output": path_constructor("data/higgs/higgs-activity_time_mt_lcc.txt")
    },
    "reply": {
        "normalization": 1,
        "min_time": 0,
        "path": path_constructor("data/higgs/higgs-activity_time_re_lcc.txt"),
        "lscc": False,
        "dangling": True,
        "is_relationship": False,
        "edge_type": "RE",
        "ignore_time": False,
        "output": path_constructor("data/higgs/higgs-activity_time_re_lcc.txt")
    },
    "all": {
        "normalization": 1,
        "min_time": 0,
        "path": path_constructor("data/higgs/higgs-activity_time_all_lcc.txt"),
        "lscc": False,
        "dangling": True,
        "is_relationship": False,
        "edge_type": "ALL",
        "ignore_time": False,
        "output": path_constructor("data/higgs/higgs-activity_time_all_lcc.txt")
    },
    "text": {
        "normalization": 1,
        "min_time": 0,
        "path": path_constructor("data/text_net.txt"),
        "lscc": False,
        "dangling": True,
        "is_relationship": False,
        "edge_type": "DEFAULT",
        "ignore_time": False,
        "output": path_constructor("data/text_net.txt")
    },
}
