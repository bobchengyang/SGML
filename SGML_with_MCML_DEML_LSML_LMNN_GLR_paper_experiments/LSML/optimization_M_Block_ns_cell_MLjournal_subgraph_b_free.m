function [M,...
    scaled_M,...
    scaled_factors,...
    M_current_eigenvector,...
    min_objective,...
    bins,...
    invalid_result] = optimization_M_Block_ns_cell_MLjournal_subgraph_b_free(current_node,league_vec,league_vec_temp,flip_number,...
    Ms_off_diagonal,...
    scaled_factors_v,...
    scaled_factors_h,...
    feature_N,...
    G,...
    M21,...
    M,...
    BCD,...
    M_current_eigenvector,...
    partial_feature,...
    x,...
    rho,...
    S_upper,...
    scaled_M,...
    scaled_factors,...
    bins,...
    objective_previous,...
    lobpcg_random_control)

tol_offdia=Inf;
counter=0;
while tol_offdia>1e-5
    
    %% linear constraints
    ddd = (0 - rho - sum(abs(Ms_off_diagonal),2));
    
    LP_A = zeros(1+feature_N,feature_N-1+feature_N);
    LP_b = zeros(1+feature_N,1);
    sign_vecdd = flip_number'*current_node*-1;
    LP_A(1,1:feature_N-1)=sign_vecdd.*abs(scaled_factors_h);
    LP_A(1,feature_N-1+BCD)=-1;
    LP_b(1)=-rho;
    scaler_v = abs(scaled_factors_v);
    
    remaining_idx = 1:feature_N;
    remaining_idx(BCD)=[];
    for LP_A_i=1:feature_N-1
        LP_A(LP_A_i+1,LP_A_i)=sign_vecdd(1,LP_A_i)*scaler_v(LP_A_i);
        LP_A(LP_A_i+1,feature_N-1+remaining_idx(LP_A_i))=-1;
        LP_b(LP_A_i+1)=ddd(LP_A_i);
    end
    LP_A(end,feature_N-1+1:end)=1;
    LP_b(end)=S_upper;
    
    LP_lb=zeros(feature_N-1+feature_N,1);
    LP_ub=zeros(feature_N-1+feature_N,1);
    
    for LP_lb_i=1:feature_N-1
        if sign(sign_vecdd(LP_lb_i))==-1 %<0
            LP_lb(LP_lb_i)=-Inf;
            LP_ub(LP_lb_i)=0;
        else %>0
            LP_lb(LP_lb_i)=0;
            LP_ub(LP_lb_i)=Inf;
        end
    end
    
    zero_mask=ones(2*feature_N-1,1);
    
    for LP_lb_i=1:feature_N-1
        if scaler_v(LP_lb_i)==0
            LP_lb(LP_lb_i)=0;
            LP_ub(LP_lb_i)=0;
            zero_mask(LP_lb_i)=0;
        end
    end
    
    LP_lb(feature_N-1+1:end)=rho;
    LP_ub(feature_N-1+1:end)=Inf;
    
    LP_Aeq = [];
    LP_beq = [];
    
    options = optimoptions('linprog','Display','none','Algorithm','interior-point');
    options.OptimalityTolerance = 1e-7;
    remaining_idx=1:feature_N;
    remaining_idx(remaining_idx==BCD)=[];
    lp_co = [G(remaining_idx,BCD);diag(G)];
    
    s_k = linprog(lp_co,...
        LP_A,LP_b,...
        LP_Aeq,LP_beq,...
        LP_lb,LP_ub,options);
    
    %% set a step size
    if isequal(league_vec,league_vec_temp)==1
        gamma_initial=0;
        tol_step_size=Inf;
        M_lc=M;
        
        [ L ] = optimization_M_set_L_Mahalanobis( partial_feature, M );
        objective_previous_step_size = x' * L * x;
        
        M_previous=zeros(feature_N-1+feature_N,1);
        M_previous(1:feature_N-1)=M(BCD,remaining_idx);
        M_previous(feature_N-1+1:end)=M(logical(eye(feature_N)));
        
        nn=0;
        while tol_step_size>1e-5
            
            t_M21_solution_previous=s_k - M_previous;
            t_M21 = M_previous + gamma_initial * t_M21_solution_previous;
            t_M21=t_M21.*zero_mask;
            
            M_lc(remaining_idx,BCD)=t_M21(1:feature_N-1);
            M_lc(BCD,remaining_idx)=M_lc(remaining_idx,BCD);
            M_lc(logical(eye(feature_N)))=t_M21(feature_N-1+1:end);
            
            if nn==0
                [ G1,G2 ] = optimization_M_set_step_gradient_g1g2( partial_feature, M_lc, x, t_M21_solution_previous.*zero_mask, BCD );
                L_constant = sqrt(sum(G2(:))^2);
            else
                [ G1 ] = optimization_M_set_step_gradient( partial_feature, M_lc, x, t_M21_solution_previous.*zero_mask, BCD );
            end
            
            gamma = gamma_initial - (1/L_constant) * sum(G1(:));
            gamma(gamma<0||isnan(gamma)==1)=0;
            gamma(gamma>1)=1;
            
            t_M21 = M_previous + gamma * t_M21_solution_previous;
            t_M21=t_M21.*zero_mask;
            
            M_lc(remaining_idx,BCD)=t_M21(1:feature_N-1);
            M_lc(BCD,remaining_idx)=M_lc(remaining_idx,BCD);
            M_lc(logical(eye(feature_N)))=t_M21(feature_N-1+1:end);
            
            [ L ] = optimization_M_set_L_Mahalanobis( partial_feature, M_lc );
            objective_current_step_size = x' * L * x;
    
            tol_step_size = norm(objective_current_step_size - objective_previous_step_size);
            
            gamma_initial=gamma;
            objective_previous_step_size = objective_current_step_size;
            M_updated=M_lc;
            nn=nn+1;
        end
    else
        gamma = 1;
        M_previous=zeros(feature_N-1+feature_N,1);
        M_previous(1:feature_N-1)=M(BCD,remaining_idx);
        M_previous(feature_N-1+1:end)=M(logical(eye(feature_N)));
        t_M21_solution_previous=s_k - M_previous;
        M21_updated = M_previous + gamma * t_M21_solution_previous;
        M21_updated=M21_updated.*zero_mask;
        
        M_updated = M;
        M_updated(remaining_idx,BCD)=M21_updated(1:feature_N-1);
        M_updated(BCD,remaining_idx)=M_updated(remaining_idx,BCD);
        M_updated(logical(eye(feature_N)))=M21_updated(feature_N-1+1:end);
    end
    
    %% evaluate the objective value
    [ L_c ] = optimization_M_set_L_Mahalanobis( partial_feature, M_updated );
    min_objective = x' * L_c * x;
    
    %% reject the result if it is larger than previous
%     if min_objective>objective_previous
%         min_objective=objective_previous;
%         invalid_result=0;
%         return
%     end
    
    %% temporarily accept the result
    M_updated(abs(M_updated)<1e-5)=0;
    
    %% detect subgraphs
    ST=[];
    for STi=1:feature_N
        for STj=1:feature_N
            if STi<STj
                if M_updated(STi,STj)~=0
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
    bins = conncomp(G);
    
    %% evaluate the temporarily accepted result with temporary scaled_M and scaled_factors
    scaled_M_=zeros(feature_N);
    scaled_factors_=zeros(feature_N);
    
    for bins_i = 1:length(unique(bins))
        M_updated_current = M_updated(bins==bins_i,bins==bins_i);
        temp_dim = size(M_updated_current,1);
        if temp_dim~=1
            rng(lobpcg_random_control);
            M_updated_current_eigenvector = ...
                optimization_M_lobpcg(randn(temp_dim,1),M_updated_current,1e-12,200);
            scaling_matrix_0 = diag(1./M_updated_current_eigenvector(:,1));
            scaling_matrix_0_inv = diag(M_updated_current_eigenvector(:,1));
            scaled_M_0 = scaling_matrix_0 * M_updated_current * scaling_matrix_0_inv;
            scaled_factors_0 = scaling_matrix_0 * ones(temp_dim) * scaling_matrix_0_inv;
            
            scaled_M_(bins==bins_i,bins==bins_i)=scaled_M_0;
            scaled_factors_(bins==bins_i,bins==bins_i)=scaled_factors_0;
            
        else
            scaled_M_(bins==bins_i,bins==bins_i)=M_updated_current;
            scaled_factors_(bins==bins_i,bins==bins_i)=1;
            
        end
        
    end
    
    scaled_M__=scaled_M_;
    scaled_M__(logical(eye(size(M,1))))=0;
    lower_bounds = sum(abs(scaled_M__),2) + rho;
    
    %% reject the result if the lower_bounds are larger than S_upper
    if sum(lower_bounds) > S_upper
        invalid_result=1;
        min_objective=objective_previous;
        %disp(['lower bounds sum:' num2str(sum(lower_bounds))]);
        %disp('========lower bounds sum larger than S_upper!!!========');
        return
    end
    
    %% finally accept the result
    invalid_result=0;
    
    M = M_updated;
    
    scaled_M=scaled_M_;
    scaled_factors=scaled_factors_;
    Ms_off_diagonal = scaled_M(remaining_idx,remaining_idx);
    scaled_factors_v = scaled_factors(remaining_idx,BCD);
    scaled_factors_h = scaled_factors(BCD,remaining_idx);
    Ms_off_diagonal(logical(eye(feature_N-1))) = 0;
    [ G ] = optimization_M_set_gradient( partial_feature, M, x);
    
    tol_offdia=norm(min_objective-objective_previous);
    
    objective_previous=min_objective;
    
    counter=counter+1;
    
end

end

