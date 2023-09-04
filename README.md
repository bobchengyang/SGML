# Signed_Graph_Metric_Learning
 source code for running experiments in paper：\
 C. Yang, G. Cheung and W. Hu, "Signed Graph Metric Learning via Gershgorin Disc Perfect Alignment," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 10, pp. 7219-7234, 1 Oct. 2022, doi: 10.1109/TPAMI.2021.3091682.\
@ARTICLE{9463735,
  author={Yang, Cheng and Cheung, Gene and Hu, Wei},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Signed Graph Metric Learning via Gershgorin Disc Perfect Alignment}, 
  year={2022},
  volume={44},
  number={10},
  pages={7219-7234},
  doi={10.1109/TPAMI.2021.3091682}}
  
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
