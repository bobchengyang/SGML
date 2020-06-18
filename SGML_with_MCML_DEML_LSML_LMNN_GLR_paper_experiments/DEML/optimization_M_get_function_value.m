function [f] = optimization_M_get_function_value(...
    gamma,...
    M_previous,...
    t_M21_solution_previous,...
    M_lc,...
    feature_N,...
    BCD,...
    zero_mask,...
    zz,...
    total_offdia,...
    remaining_idx,...
    D,...
    dia_idx)
%OPTIMIZATION_M_GET_FUNCTION_VALUE Summary of this function goes here
%   Detailed explanation goes here
t_M21 = M_previous + gamma * t_M21_solution_previous;
t_M21 = t_M21.*zero_mask;
if length(M_previous)==feature_N+(feature_N*(feature_N-1)/2) && BCD==0
%     current_slot=0;
%     for BCD_temp_i=1:feature_N-1
%         od_temp_length=length(M_lc(BCD_temp_i,BCD_temp_i+1:end));
%         M_lc(BCD_temp_i,BCD_temp_i+1:end)=t_M21(current_slot+1:current_slot+od_temp_length);
%         M_lc(BCD_temp_i+1:end,BCD_temp_i)=t_M21(current_slot+1:current_slot+od_temp_length);
%         current_slot = current_slot+od_temp_length;
%     end
    M_lc(zz)=t_M21(1:total_offdia);
    M_lc_t=M_lc';
    M_lc(zz')=M_lc_t(zz');
    %M_lc(logical(eye(feature_N)))=t_M21(total_offdia+1:end);
    M_lc(dia_idx)=t_M21(total_offdia+1:end);
elseif length(M_previous)==feature_N-1+feature_N
    M_lc(BCD,remaining_idx)=t_M21(1:feature_N-1);
    M_lc(remaining_idx,BCD)=M_lc(BCD,remaining_idx);
    %M_lc(logical(eye(feature_N)))=t_M21(feature_N-1+1:end);
    M_lc(dia_idx)=t_M21(feature_N-1+1:end);
elseif length(M_previous)==feature_N
    %M_lc(logical(eye(feature_N)))=t_M21;
    M_lc(dia_idx)=t_M21;
else
    M_lc(remaining_idx,BCD)=t_M21;
    M_lc(BCD,remaining_idx)=M_lc(remaining_idx,BCD);
end
f=dml_obj(M_lc, D);
end

