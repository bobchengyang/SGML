function [M] = optimization_M_ns_cell_MLjournal_subgraph_FULL_gs_ws(league_vec,...
    scaled_factors,...
    feature_N,...
    G,...
    M,...
    rho,...
    tol_golden_search,...
    objective_previous,...
    S_upper,...
    nv,...
    zz,...
    G_ready,...
    targets_ind,...
    same_label,...
    options,...
    dia_idx,...
    tol_NR,...
    tol_GD,...
    GS_or_NR,...
    max_iter,...
    partial_feature,...
    partial_sample)

tol_full=Inf;
counter=0;
M_best_temp=M;
objective_previous_temp=objective_previous;

remaining_idx=0;
BCD=0;

%% LP settings that do not have to be set in every Frank-Wolfe iteration starts
scaled_factors__ = scaled_factors';
scaled_factors__(dia_idx)=[];
scaled_factors__ = reshape(scaled_factors__,feature_N-1,feature_N)';

total_offdia = sum(1:feature_N-1);

%LP_A = zeros(1+feature_N,total_offdia+feature_N);
%LP_A(1,total_offdia+1:end)= 1;
LP_A_sparse_i=zeros(1,feature_N^2+feature_N);
LP_A_sparse_j=LP_A_sparse_i;
LP_A_sparse_s=LP_A_sparse_i;

LP_A_sparse_i(1:feature_N)=1;
LP_A_sparse_j(1:feature_N)=total_offdia+1:total_offdia+feature_N;
LP_A_sparse_s(1:feature_N)=1;

LP_b = [S_upper;zeros(feature_N,1)-rho];

LP_lb=zeros(total_offdia+feature_N,1);
LP_ub=zeros(total_offdia+feature_N,1)+Inf;
LP_lb(total_offdia+1:end)=rho;

t_counter=0;
t_counter_c=0;
sign_vec=zeros(1,total_offdia);
sign_idx=zeros(feature_N);
scaled_factors_zero_idx = zeros(1,total_offdia);
for vec_i=1:feature_N
    for vec_j=1:feature_N
        if vec_j>vec_i
            t_counter=t_counter+1;
            if league_vec(vec_i)==league_vec(vec_j) % positive edge, negative m entry <0
                sign_vec(t_counter)=-1;
                LP_lb(t_counter)=-Inf;
                LP_ub(t_counter)=0;
            else
                sign_vec(t_counter)=1;
                LP_lb(t_counter)=0;
                LP_ub(t_counter)=Inf;
            end
            
            if scaled_factors(vec_i,vec_j)==0
                scaled_factors_zero_idx(t_counter)=1;
            end
            
        end
        t_counter_c=t_counter_c+1;
        sign_idx(vec_i,t_counter_c)=t_counter;
    end
    t_counter_c=0;
end

%% fixing the off-diagonals to be 0's
scaled_factors_zero_idx=logical(scaled_factors_zero_idx);
LP_lb(scaled_factors_zero_idx)=0;
LP_ub(scaled_factors_zero_idx)=0;

zero_mask=ones(total_offdia+feature_N,1);
zero_mask(scaled_factors_zero_idx)=0;

sign_idx=triu(sign_idx,1);
sign_idx=sign_idx+sign_idx';
sign_idx(dia_idx)=[];
sign_idx=reshape(sign_idx,feature_N-1,feature_N)';

for LP_i=1:feature_N
    %LP_A(1+LP_i,sign_idx(LP_i,:))=abs(scaled_factors__(LP_i,:)).*sign_vec(sign_idx(LP_i,:));
    %LP_A(1+LP_i,total_offdia+LP_i)=-1;
    temp_index=feature_N+(LP_i-1)*feature_N+1:feature_N+LP_i*feature_N-1;
    temp_index1=feature_N+LP_i*feature_N;
    LP_A_sparse_i(temp_index)=1+LP_i;
    LP_A_sparse_i(temp_index1)=1+LP_i;
    LP_A_sparse_j(temp_index)=sign_idx(LP_i,:);
    LP_A_sparse_j(temp_index1)=total_offdia+LP_i;
    LP_A_sparse_s(temp_index)=abs(scaled_factors__(LP_i,:)).*sign_vec(sign_idx(LP_i,:));
    LP_A_sparse_s(temp_index1)=-1;
end

LP_A=sparse(LP_A_sparse_i,LP_A_sparse_j,LP_A_sparse_s,1+feature_N,total_offdia+feature_N);

LP_Aeq = [];
LP_beq = [];

%% LP settings that do not have to be set in every Frank-Wolfe iteration ends
while tol_full>1e-5
    
    net_gc=[2*G(zz);diag(G)];
    if sum(abs(net_gc))==0
        M=M_best_temp;
        return
    end
    s_k = gurobi_test(net_gc,...
        LP_A,LP_b,...
        LP_Aeq,LP_beq,...
        LP_lb,LP_ub,options);
    
    while isempty(s_k) == 1
        disp('===trying with larger OptimalityTolerance===');
        options.OptimalityTolerance = options.OptimalityTolerance*10;
        s_k = gurobi_test(net_gc,...
            LP_A,LP_b,...
            LP_Aeq,LP_beq,...
            LP_lb,LP_ub,options);
    end
    
    %% proximal gradient to determine the Frank-Wolfe step size starts
    
    M_previous=[M_best_temp(zz);diag(M_best_temp)];
    
    t_M21_solution_previous=s_k - M_previous;
    
    if GS_or_NR==1
        %% GS starts
        [gamma] = optimization_M_golden_section_search(...
            0,...
            1,...
            M_previous,...
            t_M21_solution_previous,...
            M_best_temp,...
            partial_feature,...
            feature_N,...
            BCD,...
            remaining_idx,...
            tol_golden_search,...
            zero_mask,...
            zz,...
            total_offdia,...
            partial_sample,...
            targets_ind,...
            same_label,...
            dia_idx,...
            nv);
        if counter==0 && gamma==0
            return
        end
        %% GS ends
    else
        %% NR starts 
        [gamma] = optimization_M_gamma_GD(...
            M_previous,...
            t_M21_solution_previous,...
            zero_mask,...
            remaining_idx,...
            BCD,...
            feature_N,...
            M_best_temp,...
            zz,...
            nv,...
            dia_idx,...
            tol_GD,...
            counter,...
            partial_sample,...
            partial_feature,...
            targets_ind,...
            same_label,...
            G_ready);
        if counter==0 && gamma==0
            return
        end
        %% NR ends
    end
    
    t_M21 = M_previous + gamma * t_M21_solution_previous;
    t_M21 = t_M21.*zero_mask;
    M_updated=M_best_temp;
    M_updated(zz)=t_M21(1:total_offdia);
    M_updated_t=M_updated';
    M_updated(zz')=M_updated_t(zz');
    
    M_updated(dia_idx)=t_M21(total_offdia+1:end);
    
    %% evaluate the objective value
    min_objective = lmnn_obj(targets_ind,same_label,M_updated,partial_feature,partial_sample);
    if min_objective==0
        M=M_best_temp;
        return
    end
    
    if min_objective>=objective_previous_temp
        M=M_best_temp;
        return
    end
    
    M_best_temp = M_updated;
    
    %% choose the M_best_temp that has not been thresholded to compute the gradient
    
    [G] = lmnn_gradient(G_ready,targets_ind,...
        same_label,...
        M_best_temp,...
        partial_feature,...
        partial_sample,...
        feature_N,...
        zz,...
        0,...
        0,...
        nv);
    
    tol_full=norm(min_objective-objective_previous_temp);
    
    objective_previous_temp=min_objective;
    
    counter=counter+1;
    if counter==max_iter
        break
    end
end
M=M_best_temp;
end

