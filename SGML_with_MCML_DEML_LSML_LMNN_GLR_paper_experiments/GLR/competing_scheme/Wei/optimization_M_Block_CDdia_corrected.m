function [ M, lr ] = optimization_M_Block_CDdia_corrected(...
    partial_sample,...
    c,...
    y,...
    lr,...
    lower_bounds,...
    n_feature,...
    M,...
    x,...
    S_upper,...
    tol_set,...
    dia_idx)

tol = Inf;

counter = 0;

[ L ] = optimization_M_set_L_Mahalanobis( partial_sample, c, M );

objective_previous = x' * L * x;

while tol > tol_set
    
    [ G1 ] = optimization_M_set_gradient( partial_sample, n_feature, c, M, y, n_feature, 0, 0 );
    
    M_diagonal = diag(M) - lr * G1;
    
    M_diagonal=max([M_diagonal lower_bounds],[],2);
    
    if sum(M_diagonal)>S_upper
        [M_diagonal]=optimization_M_clipping(M_diagonal,lower_bounds,S_upper);
    end
    
    M(dia_idx) = real(M_diagonal);
    
    [ L ] = optimization_M_set_L_Mahalanobis( partial_sample, c, M );
    objective_current = x' * L * x;
    if objective_current>objective_previous
        lr=lr/2;
    else
        lr=lr*(1+1e-2);
    end
    
    tol = norm(objective_current - objective_previous);
    counter = counter + 1;
    objective_previous = objective_current;
    if counter>1e3%not converged
        break
    end
end

end
