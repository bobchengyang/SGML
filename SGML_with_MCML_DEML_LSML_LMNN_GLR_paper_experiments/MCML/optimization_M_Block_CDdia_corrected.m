function [ M, lr, Q, objective_previous ] = optimization_M_Block_CDdia_corrected(...
    partial_sample,...
    P,...
    lr,...
    lower_bounds,...
    n_feature,...
    M,...
    S_upper,...
    tol_set,...
    partial_feature,...
    dia_idx,...
    Q,...
    objective_previous)

tol = Inf;

counter = 0;

while tol > tol_set
    
    [G1] = mcml_gradient(...
        partial_feature, ...
        P, ...
        Q, ...
        partial_sample, ...
        n_feature, ...
        0, ...
        n_feature, ...
        0, ...
        0);
    
    M_diagonal = diag(M) - lr * G1;
    
    M_diagonal=max([M_diagonal lower_bounds],[],2);
    
    if sum(M_diagonal)>S_upper
        [M_diagonal]=optimization_M_clipping(M_diagonal,lower_bounds,S_upper);
    end
    
    M(dia_idx) = real(M_diagonal);
    
    [objective_current,Q] = mcml_obj(M, partial_feature, P, partial_sample);
    
    if objective_current>objective_previous
        lr=lr/2;%avoid any solutions that end up with a larger obj.
    else
        lr=lr*(1+1e-2);
    end
    tol = norm(objective_current - objective_previous);
    counter = counter + 1;
    objective_previous = objective_current;
    if counter>1e3 %not converged
        break
    end
end

end
