%=================================================================
% Signed Graph Metric Learing (SGML) via Gershgorin Disc Alignment
% **get objective function value
%
% author: Cheng Yang
% email me any questions: cheng.yang@ieee.org
% date: June 16th, 2020
% please kindly cite the paper: 
% ['Signed Graph Metric Learning via Gershgorin Disc Alignment', 
% Cheng Yang, Gene Cheung, Wei Hu, 
% https://128.84.21.199/abs/2006.08816]
%=================================================================
function [f] = obj_value(...
    gamma,...
    M_previous,...
    t_M21_solution_previous,...
    M_lc,...
    x,...
    feature_N,...
    BCD,...
    remaining_idx,...
    zero_mask,...
    zz,...
    total_offdia,...
    partial_sample,...
    c,...
    dia_idx,...
    nv)
t_M21 = M_previous + gamma * t_M21_solution_previous;
t_M21 = t_M21.*zero_mask;
if nv==feature_N+(feature_N*(feature_N-1)/2) && BCD==0
    M_lc(zz)=t_M21(1:total_offdia);
    M_lc_t=M_lc';
    M_lc(zz')=M_lc_t(zz');
    M_lc(dia_idx)=t_M21(total_offdia+1:end);
elseif nv==feature_N-1+feature_N
    M_lc(BCD,remaining_idx)=t_M21(1:feature_N-1);
    M_lc(remaining_idx,BCD)=M_lc(BCD,remaining_idx);
    M_lc(dia_idx)=t_M21(feature_N-1+1:end);
elseif nv==feature_N
    M_lc(dia_idx)=t_M21;
else
    remaining_idx=1:feature_N;
    remaining_idx(BCD)=[];
    M_lc(remaining_idx,BCD)=t_M21;
    M_lc(BCD,remaining_idx)=M_lc(remaining_idx,BCD);
end

%=replace the following block if you run SGML on a different
        %objective function from GLR=======================================
[ L ] = graph_Laplacian( partial_sample, c, M_lc );% replace this if you need to run SGML on a different objective function
f = x' * L * x;% replace this if you need to run SGML on a different objective function
%==========================================================================
end

