# Signed_Graph_Metric_Learning
 source code for running experiments in paper：\
 @misc{yang2020signed,
    title={Signed Graph Metric Learning via Gershgorin Disc Alignment},
    author={Cheng Yang and Gene Cheung and Wei Hu},
    year={2020},
    eprint={2006.08816},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
1. run 'RUN_ME.m' for immediate experimental results.
2. you might consider using Gurobi Matlab interface instead of Matlab linprog for fast experiments:\
   s_k = linprog(net_gc,...\
        LP_A,LP_b,...\
        LP_Aeq,LP_beq,...\
        LP_lb,LP_ub,options);\
    %% ===Gurobi Matlab interface might be faster than Matlab linprog======\
    % you need to apply an Academic License (free) in order to use Gurobi Matlab\
    % interface: https://www.gurobi.com/downloads/end-user-license-agreement-academic/ \
    % once you have an Academic License and have Gurobi Optimizer\
    % installed, you should be able to run the following code by\
    % uncommenting them in the source code (see below, for example).\
    % s_k = gurobi_matlab_interface(net_gc,...\
    % LP_A,LP_b,...\
    % LP_Aeq,LP_beq,...\
    % LP_lb,LP_ub,options);
3. email me cheng DOT yang AT ieee DOT org for any questions.