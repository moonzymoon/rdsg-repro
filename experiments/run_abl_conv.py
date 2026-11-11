# -*- coding: utf-8 -*-
"""S.14 收敛审计：AT / TranAD 大容量臂训练到验证损失平台（早停 patience=30）。

预注册口径（运行前固定）：
- 与 run_abl_long.py 完全同协议（同架构 256/3/8、同 FA 匹配、同事件匹配、种子 1-5、
  三网络），仅两处不同：MAX_EPOCHS=300、PATIENCE=30，输出独立存 abl_conv.json。
- 解读规则：stop_epoch << 300 且早停触发 = 验证平台达成（"converged" 可称）；
  到 300 仍未平台则如实报"仍未确认收敛"。若收敛臂逼近或反超 RDSG，
  触发正文对比句诚实改写；若仍显著低于，S.14 可升级为收敛口径的表述。

输出：results/paper/abl_conv.json
"""
import os, sys, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)

import run_abl_long as L

L.MAX_EPOCHS = 300
L.PATIENCE = 30
L.OUT = os.path.join(L._results_root(), "paper", "abl_conv.json")

from rdsg.datasets import load_metr_la, load_pems_bay, load_beijing_air


def main():
    res = {}
    if os.path.exists(L.OUT):
        res = json.load(open(L.OUT, encoding="utf-8"))
    res.setdefault("config", {})["audit"] = "convergence"
    res["config"]["max_epochs"] = L.MAX_EPOCHS
    res["config"]["patience"] = L.PATIENCE
    for key, loader in (("metr_la", load_metr_la), ("pems_bay", load_pems_bay),
                        ("beijing_air", load_beijing_air)):
        t0 = time.time()
        L.run_dataset(key, loader, res)
        print(f"[{key}] 完成 ({time.time()-t0:.0f}s)", flush=True)
    print("done ->", L.OUT)


if __name__ == "__main__":
    main()
