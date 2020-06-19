%=================================================================
% Signed Graph Metric Learing (SGML) via Gershgorin Disc Alignment
% **get the first and second derivative of Q(M) w.r.t. gamma
%
% author: Cheng Yang
% email me any questions: cheng.yang@ieee.org
% date: June 16th, 2020
% please kindly cite the paper: 
% ['Signed Graph Metric Learning via Gershgorin Disc Alignment', 
% Cheng Yang, Gene Cheung, Wei Hu, 
% https://128.84.21.199/abs/2006.08816]
%=================================================================
function [G1,G2] = get_gamma_first_second_derivative(...
    M_previous,...
    gamma,...
    t_M21_solution_previous,...
    zero_mask,...
    remaining_idx,...
    BCD,...
    feature_N,...
    M_lc,...
    zz,...
    nv,...
    partial_sample,...
    c,...
    y,...
    dia_idx,...
    total_offdia)

t_M21 = M_previous + gamma * t_M21_solution_previous;
t_M21=t_M21.*zero_mask;

if nv==feature_N+(feature_N*(feature_N-1)/2) && BCD==0
    M_lc(zz)=t_M21(1:total_offdia);
    M_lc_t=M_lc';
    M_lc(zz')=M_lc_t(zz');
    M_lc(dia_idx)=t_M21(total_offdia+1:end);
else
    M_lc(remaining_idx,BCD)=t_M21(1:feature_N-1);
    M_lc(BCD,remaining_idx)=M_lc(remaining_idx,BCD);
    M_lc(dia_idx)=t_M21(feature_N-1+1:end);
end

% replace the following Newton-Raphson process if you need to run SGML on a
% different objective function from GLR
[ G1, G2 ] = gamma_first_second_derivative( partial_sample, feature_N, c, M_lc, y, nv, BCD, remaining_idx, t_M21_solution_previous.*zero_mask, zz );
G1=sum(G1(:));
G2=sum(G2(:));
end

