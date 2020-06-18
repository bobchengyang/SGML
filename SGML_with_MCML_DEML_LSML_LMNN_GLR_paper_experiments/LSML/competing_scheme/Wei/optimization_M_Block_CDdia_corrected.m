function [ M,lr ] = optimization_M_Block_CDdia_corrected(...
    S,...
    D,...
    lr,...
    lower_bounds,...
    n_feature,...
    M,...
    S_upper,...
    tol_set,...
    dia_idx)

tol = Inf;

counter = 0;

objective_previous = lsml_obj(M,S,D);

while tol > tol_set

    G1 = lsml_gradient(M,S,D,n_feature,n_feature,0,0);
    
    M_diagonal = diag(M) - lr * G1;
    
    M_diagonal=max([M_diagonal lower_bounds],[],2);
    
    if sum(M_diagonal)>S_upper
        [M_diagonal]=optimization_M_clipping(M_diagonal,lower_bounds,S_upper);
    end
    
    M(dia_idx) = real(M_diagonal);
    
    objective_current = lsml_obj(M,S,D);
    
    if objective_current==0
        return
    end
    
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
