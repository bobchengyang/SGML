function [M_lc] = optimization_M_M_lc(...
    M_previous,...
    gamma,...
    t_M21_solution_previous,...
    zero_mask,...
    remaining_idx,...
    BCD,...
    feature_N,...
    M_lc)

t_M21 = M_previous + gamma * t_M21_solution_previous;
t_M21=t_M21.*zero_mask;

if length(t_M21_solution_previous)~=feature_N+(feature_N*(feature_N-1)/2)
    M_lc(remaining_idx,BCD)=t_M21(1:feature_N-1);
    M_lc(BCD,remaining_idx)=M_lc(remaining_idx,BCD);
    M_lc(logical(eye(feature_N)))=t_M21(feature_N-1+1:end);
else 
    current_slot=0;
    for BCD_temp_i=1:feature_N-1
        od_temp_length=length(M_lc(BCD_temp_i,BCD_temp_i+1:end));
        M_lc(BCD_temp_i,BCD_temp_i+1:end)=t_M21(current_slot+1:current_slot+od_temp_length);
        M_lc(BCD_temp_i+1:end,BCD_temp_i)=t_M21(current_slot+1:current_slot+od_temp_length);
        current_slot = current_slot+od_temp_length;
    end
    M_lc(logical(eye(feature_N)))=t_M21((feature_N*(feature_N-1)/2)+1:end);
end
end

