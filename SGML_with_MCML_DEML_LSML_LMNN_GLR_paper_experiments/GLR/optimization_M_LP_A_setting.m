function [LP_A_sparse_i,LP_A_sparse_j,LP_A_sparse_s,LP_b,LP_lb,LP_ub] = optimization_M_LP_A_setting(feature_N,rho)

LP_A_sparse_i=zeros(1,feature_N-1+1+2*(feature_N-1)+feature_N);
LP_A_sparse_j=LP_A_sparse_i;
LP_A_sparse_s=LP_A_sparse_i;

LP_b = zeros(1+feature_N,1);
%sign_vecdd = flip_number'*current_node*-1;

%LP_A(1,1:feature_N-1)=sign_vecdd.*abs(scaled_factors_h);
%LP_A(1,feature_N-1+BCD)=-1;
LP_A_sparse_i(1:feature_N-1)=1;
LP_A_sparse_j(1:feature_N-1)=1:feature_N-1;
%LP_A_sparse_s(1:feature_N-1)=sign_vecdd.*abs(scaled_factors_h);
LP_A_sparse_i(feature_N)=1;
%LP_A_sparse_j(feature_N)=feature_N-1+BCD;
LP_A_sparse_s(feature_N)=-1;

LP_b(1)=-rho;
LP_b(end)=feature_N;
%scaler_v = abs(scaled_factors_v);

%remaining_idx = 1:feature_N;
%remaining_idx(BCD)=[];
for LP_A_i=1:feature_N-1
    %LP_A(LP_A_i+1,LP_A_i)=sign_vecdd(1,LP_A_i)*scaler_v(LP_A_i);
    %LP_A(LP_A_i+1,feature_N-1+remaining_idx(LP_A_i))=-1;
    temp_index=feature_N+(LP_A_i-1)*2+1;
    temp_index1=feature_N+(LP_A_i-1)*2+2;
    LP_A_sparse_i(temp_index)=LP_A_i+1;
    LP_A_sparse_j(temp_index)=LP_A_i;
    %LP_A_sparse_s(temp_index)=sign_vecdd(1,LP_A_i)*scaler_v(LP_A_i);
    LP_A_sparse_i(temp_index1)=LP_A_i+1;
    %LP_A_sparse_j(temp_index1)=feature_N-1+remaining_idx(LP_A_i);
    LP_A_sparse_s(temp_index1)=-1;
    %LP_b(LP_A_i+1)=ddd(LP_A_i);
end
%LP_A(end,feature_N-1+1:end)=1;
temp_index=feature_N-1+1+2*(feature_N-1)+1;
LP_A_sparse_i(temp_index:end)=feature_N+1;
LP_A_sparse_j(temp_index:end)=feature_N-1+1:feature_N-1+feature_N;
LP_A_sparse_s(temp_index:end)=1;

LP_lb=zeros(feature_N-1+feature_N,1);
LP_ub=zeros(feature_N-1+feature_N,1);
LP_lb(feature_N-1+1:end)=rho;
LP_ub(feature_N-1+1:end)=Inf;

end

