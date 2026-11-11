# -*- coding: utf-8 -*-
"""S.7 选择校正检验：RDSG vs 全部 11 个外部基线的 Westfall–Young 单步 min-P。

方法：每网络取 10 种子配对召回差（RDSG − 基线），对全部 2^10=1024 个符号翻转
精确枚举；每方法单侧置换 p'_i(b) = #{c: T_i(c) >= T_i(b)}/1024（T = 差之和）；
单步 min-P 选择校正 p = #{b: min_i p'_i(b) <= min_i p'_i(全+)} / 1024。
数据：冻结的逐种子 JSON（e2_*.json / abl_at / abl_tranad / w6_rough_baseline）。
输出：results/paper/e2_select_adjusted.json
"""
import json
import numpy as np
from itertools import product

NETS = {'metr_la': 'e2_metr_la.json', 'pems_bay': 'e2_pems_bay.json',
        'beijing_air': 'e2_beijing_air.json'}
LABEL = {'b4_if': 'IF-agg', 'b4_lof': 'LOF-agg', 'b3_gdn': 'GDN',
         'b1_corr': 'Corr-matrix', 'b2_granger': 'Granger', 'b5a_mi': 'MI-graph',
         'b5b_pcorr': 'Partially-corr', 'b5c_var': 'VAR-resid',
         'rough': 'Rough-NN', 'at': 'Anomaly-Trans', 'tranad': 'TranAD'}


def _results_root():
    d = os.path.dirname(os.path.abspath(__file__))
    import os
    for base in (d, os.path.dirname(d), os.path.dirname(os.path.dirname(d))):
        if os.path.isdir(os.path.join(base, "results")):
            return os.path.join(base, "results")
    return os.path.normpath(os.path.join(d, "..", "..", "results"))


def sr(node):
    return np.array([(node[s]['recall'] if isinstance(node[s], dict) else node[s])
                     for s in sorted(node.keys(), key=int)])


def main():
    import os
    R = _results_root()
    res = {}
    for net, f in NETS.items():
        d = json.load(open(os.path.join(R, 'paper', f), encoding='utf-8'))
        rdsg = sr(d['rdsg'])
        bl = {}
        for k in LABEL:
            if k in d:
                bl[LABEL[k]] = sr(d[k])
        bl['Rough-NN'] = sr(json.load(open(os.path.join(R, 'paper', 'w6_rough_baseline.json'),
                                          encoding='utf-8'))[net]['seeds'])
        bl['Anomaly-Trans'] = sr(json.load(open(os.path.join(R, 'paper', 'abl_at.json'),
                                                encoding='utf-8'))[net])
        bl['TranAD'] = sr(json.load(open(os.path.join(R, 'paper', 'abl_tranad.json'),
                                         encoding='utf-8'))[net])
        n = min(len(rdsg), min(len(v) for v in bl.values()))
        names = list(bl.keys())
        D = np.array([rdsg[:n] - bl[k][:n] for k in names])
        S = np.array(list(product([1.0, -1.0], repeat=n)))  # 首行 = 全 +
        T = D @ S.T
        P = np.empty((len(names), 1024))
        for i in range(len(names)):
            P[i] = (T[i][None, :] >= T[i][:, None]).sum(axis=1) / 1024.0
        p_min = float(P[:, 0].min())
        adj = float((P.min(axis=0) <= p_min).sum() / 1024.0)
        i_strong = int(np.argmin(D.mean(axis=1)))
        res[net] = dict(strongest=names[i_strong], mean_gap=float(D.mean(axis=1)[i_strong]),
                        raw_min_p=p_min, wy_minP=adj, m=len(names), n=n)
        print(f'{net}: strongest={names[i_strong]} min-p={p_min:.5f} -> WY min-P={adj:.5f}')
    json.dump(res, open(os.path.join(R, 'paper', 'e2_select_adjusted.json'), 'w'),
              indent=1)


if __name__ == '__main__':
    main()
