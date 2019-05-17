CMSC 828U Project: Online Boosting for Binary Classification
=============================================================

A suite of boosting algorithms and weak learners for the online learning setting. This repo builds upon 
the [online boosting](https://github.com/crm416/online_boosting) repo of [Charlie Marsh ](https://github.com/crm416).

## Ensemblers

The repo includes implementations for the following online boosting algorithms (coded by Charlie Marsh):

1. Online AdaBoost (OzaBoost), from [Oza & Russell](http://ti.arc.nasa.gov/m/profile/oza/files/ozru01a.pdf).
2. Online GradientBoost (OGBoost), from [Leistner et al.](http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=5457451)
3. Online SmoothBoost (OSBoost), from [Chen et al.](http://ntur.lib.ntu.edu.tw/retrieve/188503/07.pdf)
4. OSBoost with Expert Advice (EXPBoost), again from [Chen et al.](http://ntur.lib.ntu.edu.tw/retrieve/188503/07.pdf)
5. OSBoost with Online Convex Programming (OCPBoost), again from [Chen et al.](http://ntur.lib.ntu.edu.tw/retrieve/188503/07.pdf)

Implementations of the more recent online boosting algorithm with theoretical guarantees are implemented:

`6.` Online Boost-By-Majority (OnlineBBM), from [Beygelzimer et al.](https://arxiv.org/pdf/1502.02651.pdf)

`7.` Online Adaptive Adaboost with logistic loss (AdaBoostOL), afain from [Beygelzimer et al.](https://arxiv.org/pdf/1502.02651.pdf)

The corresponding Python modules can be found in the _ensemblers_ folder, named as above.

## Weak Learners

The repo includes implementation of online weak learners from the [online boosting](https://github.com/crm416/online_boosting) repo.

## Dependencies

Dependencies can be found under the `requirements.txt` file .

## Experiments

Report and experimental results to be uploaded soon.
