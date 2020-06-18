function [M,...
    scaled_M,...
    scaled_factors,...
    bins,...
    min_objective] = optimization_M_ns_cell_MLjournal_subgraph_FULL_gd(partial_feature,league_vec,...
    scaled_M,...
    bins,...
    scaled_factors,...
    feature_N,...
    G,...
    M,...
    x,...
    rho,...
    lobpcg_random_control,...
    objective_previous,...
    S_upper)

tol_full=Inf;
counter=0;
M_best_temp=M;
objective_previous_temp=objective_previous;
while tol_full>1e-5
    
    scaled_factors_ = scaled_factors';
    scaled_factors_(logical(eye(feature_N)))=[];
    scaled_factors__ = reshape(scaled_factors_,feature_N-1,feature_N)';
    
    %% linear constriants start
    total_offdia = sum(1:feature_N-1);
    
    LP_A = zeros(1+feature_N,total_offdia+feature_N);
    
    LP_A(1,total_offdia+1:end)= 1;
    
    LP_b = [S_upper;zeros(feature_N,1)-rho];
    %% linear constriants end
    
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
    LP_lb(logical(scaled_factors_zero_idx))=0;
    LP_ub(logical(scaled_factors_zero_idx))=0;
    
    zero_mask=ones(total_offdia+feature_N,1);
    zero_mask(logical(scaled_factors_zero_idx))=0;
    
    sign_idx=triu(sign_idx,1);
    sign_idx=sign_idx+sign_idx';
    sign_idx(logical(eye(feature_N)))=[];
    sign_idx=reshape(sign_idx,feature_N-1,feature_N)';
    
    for LP_i=1:feature_N
        LP_A(1+LP_i,sign_idx(LP_i,:))=abs(scaled_factors__(LP_i,:)).*sign_vec(sign_idx(LP_i,:));
        LP_A(1+LP_i,total_offdia+LP_i)=-1;
    end
    
    LP_Aeq = [];
    LP_beq = [];
    
    options = optimoptions('linprog','Display','none','Algorithm','interior-point');
    options.OptimalityTolerance = 1e-7;
    
    gg_list = zeros(sum(1:feature_N-1),1);
    tc=0;
    for gg = 1:feature_N-1
        gg_list(tc+1:tc+feature_N-gg) = 2*G(gg,gg+1:end)';
        tc=tc+feature_N-gg;
    end
    net_gc=[gg_list;diag(G)];
    s_k = linprog(net_gc,...
        LP_A,LP_b,...
        LP_Aeq,LP_beq,...
        LP_lb,LP_ub,options);
    
    while isempty(s_k) == 1
        disp('===trying with larger OptimalityTolerance===');
        options.OptimalityTolerance = options.OptimalityTolerance*10;
        s_k = linprog(net_gc,...
            LP_A,LP_b,...
            LP_Aeq,LP_beq,...
            LP_lb,LP_ub,options);
    end
    
    M_previous = zeros(total_offdia+feature_N,1);
    current_slot=0;
    for Mp_i=1:feature_N-1
        od_temp_content=M_best_temp(Mp_i,Mp_i+1:end);
        od_temp_length=length(od_temp_content);
        M_previous(current_slot+1:current_slot+od_temp_length)=od_temp_content;
        current_slot = current_slot+od_temp_length;
    end
    M_previous(total_offdia+1:end)=M_best_temp(logical(eye(feature_N)));
    t_M21_solution_previous=s_k - M_previous;
    M_lc=M_best_temp;
    remaining_idx=1:feature_N;
    BCD=0;
    %% examine the gradient at 0 and 1
    [G_0] = optimization_M_gamma_M_lc(...
        M_previous,...
        0,...
        t_M21_solution_previous,...
        zero_mask,...
        remaining_idx,...
        BCD,...
        feature_N,...
        partial_feature,...
        M_lc,...
        x);
    [G_1] = optimization_M_gamma_M_lc(...
        M_previous,...
        1,...
        t_M21_solution_previous,...
        zero_mask,...
        remaining_idx,...
        BCD,...
        feature_N,...
        partial_feature,...
        M_lc,...
        x);
    if G_0<=0 && G_1<=0 % fmin=f(gamma=1)
        gamma=1;
    elseif G_0>=0 && G_1>=0 % fmin=f(gamma=0)
        gamma=0;
    else % fmin=f(gamma\in[0,1])
        gL=0;
        gU=1;
        nn=0;
        while abs(gL-gU)>1e-5
            nn=nn+1;
            gamma=(gL+gU)/2;
            [G_gM] = optimization_M_gamma_M_lc(...
                M_previous,...
                gamma,...
                t_M21_solution_previous,...
                zero_mask,...
                remaining_idx,...
                BCD,...
                feature_N,...
                partial_feature,...
                M_lc,...
                x);
            if G_gM>0 % fmin=f(gamma\in[gL,gamma])
                gU=gamma;
                gamma=(gL+gU)/2;
                [G_gM] = optimization_M_gamma_M_lc(...
                    M_previous,...
                    gamma,...
                    t_M21_solution_previous,...
                    zero_mask,...
                    remaining_idx,...
                    BCD,...
                    feature_N,...
                    partial_feature,...
                    M_lc,...
                    x);
            elseif G_gM<0 % fmin=f(gamma\in[gamma,gU])
                gL=gamma;
                gamma=(gL+gU)/2;
                [G_gM] = optimization_M_gamma_M_lc(...
                    M_previous,...
                    gamma,...
                    t_M21_solution_previous,...
                    zero_mask,...
                    remaining_idx,...
                    BCD,...
                    feature_N,...
                    partial_feature,...
                    M_lc,...
                    x);
            else % G_gM=0
                gamma=gM;
                break
            end
        end
    end
    
    [M_updated] = optimization_M_M_lc(...
        M_previous,...
        gamma,...
        t_M21_solution_previous,...
        zero_mask,...
        remaining_idx,...
        BCD,...
        feature_N,...
        M_lc);
    
    
    %% evaluate the objective value
    [ L_c ] = optimization_M_set_L_Mahalanobis( partial_feature, M_updated );
    min_objective = x' * L_c * x;
    
    M_best_temp = M_updated;
    
    %% choose the M_best_temp that has not been thresholded to compute the gradient
    [ G ] = optimization_M_set_gradient( partial_feature, M_best_temp, x);
    
    tol_full=norm(min_objective-objective_previous_temp);
    
    objective_previous_temp=min_objective;
    
    counter=counter+1;
    
end

%     %% temporarily accept the result
%     [leftend_diff,...
%         lower_bounds,...
%         scaled_M_,...
%         scaled_factors_] = optimization_M_check_leftend(M_best_temp,...
%         feature_N,...
%         lobpcg_random_control,...
%         rho);
%
%     %% detect subgraphs if leftends are not aligned OR the lower bounds are too large
%     if leftend_diff>1e-8 %|| sum(lower_bounds)>S_upper % check leftends and lower_bounds
M_best_temp(abs(M_best_temp)<1e-5)=0;

%% detect subgraphs
ST=[];
for STi=1:feature_N
    for STj=1:feature_N
        if STi<STj
            if M_best_temp(STi,STj)~=0
                ST=[ST [STi;STj]];
            end
        end
    end
end
if isempty(ST)~=1
    G = graph(ST(1,:),ST(2,:),[],feature_N);
else
    G = graph([],[],[],feature_N);
end
bins_temp = conncomp(G);

%% evaluate the temporarily accepted result with temporary scaled_M and scaled_factors
scaled_M_=zeros(feature_N);
scaled_factors_=zeros(feature_N);

for bins_i = 1:length(unique(bins_temp))
    M_updated_current = M_best_temp(bins_temp==bins_i,bins_temp==bins_i);
    temp_dim = size(M_updated_current,1);
    if temp_dim~=1
        rng(lobpcg_random_control);
        M_updated_current_eigenvector = ...
            optimization_M_lobpcg(randn(temp_dim,1),M_updated_current,1e-12,200);
        scaling_matrix_0 = diag(1./M_updated_current_eigenvector(:,1));
        scaling_matrix_0_inv = diag(M_updated_current_eigenvector(:,1));
        scaled_M_0 = scaling_matrix_0 * M_updated_current * scaling_matrix_0_inv;
        scaled_factors_0 = scaling_matrix_0 * ones(temp_dim) * scaling_matrix_0_inv;
        
        scaled_M_(bins_temp==bins_i,bins_temp==bins_i)=scaled_M_0;
        scaled_factors_(bins_temp==bins_i,bins_temp==bins_i)=scaled_factors_0;
        
    else
        scaled_M_(bins_temp==bins_i,bins_temp==bins_i)=M_updated_current;
        scaled_factors_(bins_temp==bins_i,bins_temp==bins_i)=1;
        
    end
    
end

scaled_M__=scaled_M_;
scaled_M__(logical(eye(size(M_best_temp,1))))=0;
lower_bounds = sum(abs(scaled_M__),2) + rho;

%% reject the result if the lower_bounds are larger than S_upper
if sum(lower_bounds) > S_upper
    invalid_result=1;
    min_objective=objective_previous;
    %disp(['lower bounds sum:' num2str(sum(lower_bounds))]);
    %disp('========lower bounds sum larger than S_upper!!!========');
    return
end
%     end
bins=bins_temp;
M=M_best_temp;
scaled_M=scaled_M_;
scaled_factors=scaled_factors_;
end

