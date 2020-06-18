function [ M ] = optimization_M_Block_CDdiaLP_subgraph(...
    partial_feature,...
    n_feature,...
    M,...
    x,...
    S_upper,...
    rho,...
    tol_diagonal,...
    bins,...
    lobpcg_random_control,...
    tol_golden_search)

tol = 1e4;

counter = 0;

[ L ] = optimization_M_set_L_Mahalanobis( partial_feature, M );

objective_previous = x' * L * x;

while tol > tol_diagonal
      
    scaled_M=zeros(n_feature);
    scaled_factors=zeros(n_feature);
    
    for bins_i = 1:length(unique(bins))
        M_current = M(bins==bins_i,bins==bins_i);
        temp_dim=size(M_current,1);
        if temp_dim~=1
            rng(lobpcg_random_control);
            [M_current_eigenvector] = ...
                optimization_M_lobpcg(randn(temp_dim,1),M_current,1e-12,200);
            
            scaling_matrix_0 = diag(1./M_current_eigenvector(:,1));
            scaling_matrix_0_inv = diag(M_current_eigenvector(:,1));
            scaled_M_0 = scaling_matrix_0 * M_current * scaling_matrix_0_inv;
            scaled_factors_0 = scaling_matrix_0 * ones(temp_dim) * scaling_matrix_0_inv;
            
            scaled_M(bins==bins_i,bins==bins_i)=scaled_M_0;
            scaled_factors(bins==bins_i,bins==bins_i)=scaled_factors_0;
            
        else
            scaled_M(bins==bins_i,bins==bins_i)=M_current;
            scaled_factors(bins==bins_i,bins==bins_i)=1;
            
        end
    end
    
    scaled_M_=scaled_M;
    scaled_M_(logical(eye(n_feature)))=0;
    lower_bounds = sum(abs(scaled_M_),2) + rho;
    
    if sum(lower_bounds) > S_upper
        disp(['lower bounds sum:' num2str(sum(lower_bounds))]);
        disp('========lower bounds sum larger than S_upper!!!========');
        return       
    end
    
    [ G ] = optimization_M_set_gradient( partial_feature, M, x);
        
    LP_A = zeros(1,n_feature)+1;
    
    LP_b = S_upper;
    
    LP_Aeq = [];
    LP_beq = [];
    LP_lb = lower_bounds';
    LP_ub = zeros(1,n_feature) + Inf;
    options = optimoptions('linprog','Display','none','Algorithm','interior-point');
    options.OptimalityTolerance = 1e-7;
    
    s_k = linprog(diag(G),...
        LP_A,LP_b,...
        LP_Aeq,LP_beq,...
        LP_lb,LP_ub,options);

    M_previous = diag(M);

    t_M21_solution_previous=s_k - M_previous;

    [gamma] = optimization_M_golden_section_search(...
    0,...
    1,...
    M_previous,...
    t_M21_solution_previous,...
    M,...
    partial_feature,...
    x,...
    n_feature,...
    0,...
    tol_golden_search);
    
    M_updated=M; 
    M_updated(logical(eye(n_feature))) = diag(M) + gamma * ( s_k - diag(M));
    [ L ] = optimization_M_set_L_Mahalanobis( partial_feature, M_updated );
    objective_current = x' * L * x;
    
    if objective_current>objective_previous
        return
    end
    
    %disp(['obj dia: ' num2str(objective_current)]);
    tol = norm(objective_current - objective_previous);  
    objective_previous = objective_current; 
    counter = counter + 1;
    
end

end

















