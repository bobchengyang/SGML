function [ M,lr ] = optimization_M_Block_CDdia_corrected(...
    D,...
    n_feature,...
    M,...
    lr,...
    S_upper,...
    tol_set,...
    lower_bounds,...
    length_D,...
    dia_idx)

tol = Inf;

counter = 0;

objective_previous = dml_obj(M,D);

while tol > tol_set
    
    G1 = dml_gradient(...
        M, ...
        D, ...
        n_feature, ...
        n_feature, ...
        0, ...
        0,...
        length_D);
    
    M_diagonal = diag(M) + lr * G1;
    
    M_diagonal=max([M_diagonal lower_bounds],[],2);
    
    if sum(M_diagonal)>S_upper
        [M_diagonal]=optimization_M_clipping(M_diagonal,lower_bounds,S_upper);
    end
    
    M(dia_idx) = real(M_diagonal);
    
    objective_current = dml_obj(M,D);
    
    if objective_current>objective_previous
        lr=lr*(1+1e-2);
    else
        lr=lr/2;
    end
    
    %disp(['obj dia: ' num2str(objective_current)]);
    tol = norm(objective_current - objective_previous);
    counter = counter + 1;
    objective_previous = objective_current;
    if counter>1e3 % not converged
        break
    end
end

end