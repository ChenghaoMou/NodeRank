#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-12-23 下午2:40
# @Author  : Aries
# @Site    : 
# @File    : SIR.py
# @Software: PyCharm

# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy
import numpy as np
import pylab as pl
import networkx as nx
import random

from tqdm import tqdm

from Preprocess.Constructors import time_series
from Preprocess.Conf import path_constructor

beta = 2
gamma = 0.05
TS = 1.0
ND = 70.0
S0 = 1 - 1e-6
I0 = 1e-6
INPUT = (S0, I0, 0.0)


def diff_eqs(INP, t):
    Y = np.zeros((3))
    V = INP
    Y[0] = - beta * V[0] * V[1]
    Y[1] = beta * V[0] * V[1] - gamma * V[1]
    Y[2] = gamma * V[1]
    return Y
# # mention
# n = """88
# 13808
# 9021
# 12751
# 677
# 3998
# 511
# 64911
# 35376
# 67382
# 6241
# 52908
# 6940
# 11991
# 3604
# 14957
# 38602
# 26158
# 9964
# 5226
# 1276
# 98204
# 4665
# 3972
# 89805
# 3547
# 93502
# 13798
# 84112
# 7690
# 2014
# 492
# 5137
# 42172
# 35375
# 26398
# 35729
# 2417
# 46117
# 36436
# 80429
# 4336
# 33833
# 16527
# 973
# 54222
# 103447
# 553
# 92274
# 50218
# """
# b="""88
# 3998
# 13808
# 52087
# 64911
# 677
# 3604
# 2417
# 12751
# 9021
# 1988
# 6241
# 67382
# 5226
# 12928
# 27723
# 28951
# 2014
# 13778
# 19604
# 6940
# 35376
# 54258
# 170861
# 456
# 511
# 16801
# 13798
# 48623
# 11991
# 11921
# 92274
# 66265
# 349
# 32204
# 39885
# 35729
# 110903
# 9964
# 4336
# 38602
# 2177
# 1276
# 103447
# 216
# 42182
# 2160
# 26486
# 6080
# 59195
# """
# reply
# n = """9021
# 52908
# 67382
# 677
# 13808
# 6241
# 12751
# 36436
# 33833
# 3604
# 9964
# 8855
# 17155
# 98204
# 80429
# 25006
# 92274
# 35376
# 42172
# 37502
# 80426
# 2280
# 134319
# 9025
# 52941
# 14611
# 5137
# 4665
# 3533
# 109647
# 50218
# 22470
# 152385
# 13798
# 20385
# 26159
# 35375
# 49610
# 492
# 14957
# 30478
# 6940
# 69891
# 12928
# 69883
# 2014
# 3972
# 6080
# 9036
# 4528
# """
# b = """677
# 9021
# 13808
# 36436
# 67382
# 52908
# 6241
# 5137
# 3604
# 13798
# 134319
# 12751
# 92274
# 80426
# 37502
# 26159
# 80429
# 8855
# 98204
# 9964
# 33833
# 12965
# 492
# 207364
# 201222
# 225859
# 152385
# 237807
# 118091
# 42177
# 2014
# 14957
# 28951
# 109647
# 9025
# 17155
# 12928
# 52150
# 3972
# 22470
# 511
# 52926
# 553
# 50218
# 141656
# 5155
# 6940
# 50244
# 42907
# 4528
# """
# retweet
# n = """1988
# 6940
# 11991
# 9021
# 677
# 14588
# 35376
# 19913
# 14075
# 3998
# 13808
# 75844
# 26398
# 103447
# 12751
# 156017
# 349
# 39885
# 3547
# 14615
# 28951
# 511
# 50218
# 3603
# 57105
# 49179
# 56968
# 12928
# 15210
# 35375
# 32972
# 42182
# 14625
# 35729
# 28072
# 9964
# 1276
# 3972
# 13798
# 51702
# 88
# 14957
# 33833
# 11692
# 5226
# 16801
# 27483
# 92274
# 13820
# 10179
# """
# b = """88
# 2342
# 64911
# 3998
# 39420
# 134095
# 169287
# 28951
# 13808
# 42172
# 11991
# 1988
# 39885
# 5226
# 39889
# 677
# 3547
# 12751
# 110903
# 349
# 56968
# 9964
# 6940
# 26158
# 26398
# 20385
# 57105
# 4665
# 13813
# 519
# 103447
# 13798
# 35729
# 14588
# 3571
# 19913
# 16801
# 11692
# 67382
# 511
# 14615
# 14625
# 121414
# 4336
# 89805
# 30383
# 3603
# 9704
# 42182
# 35376
# """
# all
# n = """88
# 677
# 1988
# 13808
# 3998
# 9021
# 6940
# 64911
# 349
# 511
# 14615
# 35376
# 12751
# 11991
# 19913
# 9964
# 35843
# 1276
# 32204
# 67382
# 3604
# 6241
# 26398
# 5226
# 42172
# 33833
# 2866
# 39885
# 89805
# 519
# 4665
# 2177
# 14588
# 14611
# 14075
# 3547
# 243
# 103447
# 3571
# 49179
# 11053
# 3808
# 3972
# 35375
# 152385
# 7690
# 2417
# 20385
# 12928
# 2567
# 38602
# 35729
# 14957
# 52908
# 37502
# 492
# 5137
# 57105
# 98204
# 10179
# 16527
# 44086
# 28951
# 42182
# 16801
# 5335
# 26677
# 71783
# 50218
# 42179
# 19604
# 47015
# 13798
# 37532
# 17155
# 15210
# 2280
# 2166
# 16011
# 26158
# 2014
# 9704
# 27483
# 14625
# 39124
# 56968
# 30225
# 43725
# 13820
# 32972
# 84647
# 92274
# 22470
# 26486
# 3533
# 45605
# 110903
# 553
# 28072
# 6585
# """

# # all-i
# n="""88
# 677
# 1988
# 64911
# 3998
# 13808
# 32204
# 349
# 12751
# 11991
# 6940
# 511
# 3604
# 9021
# 14615
# 35843
# 35376
# 42172
# 19913
# 26398
# 39885
# 1276
# 5226
# 67382
# 519
# 152385
# 2177
# 6241
# 9964
# 89805
# 84647
# 35729
# 14588
# 4665
# 26486
# 11053
# 14611
# 98204
# 3547
# 3571
# 28951
# 14075
# 2567
# 2417
# 33833
# 103447
# 3808
# 42182
# 20385
# 12928
# """
# all-ai
n = """88
64911
3998
13808
677
1988
26486
32204
3604
12751
349
11991
35376
26398
6940
42172
14615
67382
84647
9021
519
152385
511
2177
1276
5226
35729
35843
2417
374730
28951
89805
6241
98204
39885
19913
9964
14588
4665
121414
11053
14611
27483
12928
42182
13778
3547
2866
3808
54258
"""
# all-a
b="""88        
677       
64911     
3998      
13808     
1988      
9021      
6940      
35376     
14615     
349       
12751     
3604      
11991     
511       
32204     
1276      
2417      
519       
26398     
67382     
5226      
19913     
2177      
35843     
9964      
6241      
2866      
152385    
42172     
14588     
7690      
38602     
33833     
89805     
4665      
26486     
39885     
14611     
3808      
98204     
39124     
5335      
28951     
7533      
11053     
3547      
12928     
35729     
52908     
"""
# b="""88
# 3998
# 13813
# 13808
# 52087
# 677
# 64911
# 2342
# 39420
# 39885
# 1988
# 16801
# 103447
# 3604
# 5226
# 42172
# 28951
# 21585
# 11991
# 2417
# 32204
# 349
# 519
# 28974
# 12751
# 9964
# 511
# 35729
# 6940
# 134095
# 169287
# 13798
# 39889
# 3547
# 121414
# 19604
# 26398
# 56968
# 14615
# 79198
# 26158
# 35376
# 2014
# 42182
# 8136
# 67382
# 1276
# 6080
# 2177
# 110903
# 6241
# 12965
# 79299
# 374730
# 12928
# 27723
# 9704
# 3571
# 2567
# 29482
# 7690
# 4336
# 383
# 9021
# 92274
# 13778
# 2166
# 1997
# 4665
# 32972
# 15691
# 19913
# 7533
# 3549
# 89805
# 32243
# 5137
# 11714
# 11921
# 16527
# 5335
# 35843
# 8005
# 16011
# 184172
# 11692
# 456
# 25388
# 5193
# 80814
# 13795
# 20385
# 1880
# 57105
# 14625
# 14588
# 27483
# 14166
# 54258
# 48623
# """


class SIRDN(object):

    def __init__(self, time_series, seeds):
        self.time_series = time_series
        self.graph = nx.DiGraph()
        self.s = set()
        self.i = set(seeds)
        self.r = set()

        for time, edges in time_series.items():
            for edge in edges:
                self.graph.add_node(edge[0])
                self.graph.add_node(edge[1])

    def run(self):

        result = list()

        for t in tqdm(sorted(self.time_series.keys())):
            self.graph.add_edges_from([(edge[0], edge[1])for edge in self.time_series[t]])

            for node in self.i:
                self.s.update(list(self.graph.successors(node)))

            len_s = len(self.s)
            len_i = len(self.i)
            len_r = len(self.r)

            sample_s = set()
            if len_s >= int((beta * len_s * len_i / len(self.graph.edges))) >= 1:
                sample_s = set(random.sample(self.s, int((beta * len_s * len_i / len(self.graph.edges)))))
                self.s -= sample_s
            sample_i = set(random.sample(self.i, int(gamma * len_i)))
            self.i -= sample_i
            self.r.update(sample_i)
            self.i.update(sample_s)

            result.append(numpy.asarray([len(self.s), len(self.i), len(self.r)]))

        return numpy.asarray(result)

    def report(self):
        print("{0:<10} {1:<10} {2:<10}".format(len(self.s), len(self.i), len(self.r)))


time_series=time_series(path_constructor("data/higgs/higgs-activity_time_all_lcc.txt"))
trials = 100
RES1_overall = None
RES2_overall = None

for i in range(trials):
    print(i)

    psirdn = SIRDN(time_series=time_series,
                  seeds=map(int, [x for x in b.split("\n") if x]))

    nsirdn = SIRDN(time_series=time_series,
                  seeds=map(int, [x for x in n.split("\n") if x]))
    RES1 = psirdn.run()
    RES2 = nsirdn.run()

    if RES1_overall is None:
        RES1_overall = RES1
    else:
        RES1_overall += RES1
    if RES2_overall is None:
        RES2_overall = RES2
    else:
        RES2_overall += RES2

RES1_average = RES1_overall/trials
RES2_average = RES2_overall/trials

pl.plot(RES1_average[:, 0], '-b', label='Susceptibles-NR-A')  # I change -g to g--  # RES[:,0], '-g',
pl.plot(RES1_average[:, 2], '-g', label='Recovereds-NR-A')  # RES[:,2], '-k',
pl.plot(RES1_average[:, 1], '-r', label='Infectious-NR-A')
pl.plot(RES2_average[:, 0], '-c', label='Susceptibles-NR-AI')  # I change -g to g--  # RES[:,0], '-g',
pl.plot(RES2_average[:, 2], '-m', label='Recovereds-NR-AI')  # RES[:,2], '-k',
pl.plot(RES2_average[:, 1], '-y', label='Infectious-NR-AI')

pl.legend(loc=0)
pl.title('SIR epidemic simulation in {0} trials'.format(trials))
pl.xlabel('Time')
pl.ylabel('Susceptibles, Recovereds, and Infectious')
pl.savefig('SIR-high-all-a-vs-ai.png', dpi=300)  # This does, too
pl.show()
