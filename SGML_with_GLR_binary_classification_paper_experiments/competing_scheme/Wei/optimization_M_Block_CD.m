function [ M,lr,lower_bounds ] = optimization_M_Block_CD(...
    partial_sample,...
    n_feature,...
    c,...
    y,...
    lr,...
    m11,...
    x_original,...
    M21,...
    M22_min_eig,...
    M,...
    BCD,...
    tol_set,...
    remaining_idx,...
    lower_bounds)


tol = Inf;
counter = 0;

[ L ] = optimization_M_set_L_Mahalanobis( partial_sample, c, M );
objective_previous = x_original' * L * x_original;

lower_bounds0=lower_bounds;
lower_bounds0(BCD)=0;
max_od=n_feature-sum(lower_bounds0);

while tol > tol_set
    
    [ G1 ] = optimization_M_set_gradient( partial_sample, n_feature, c, M, y, n_feature-1, BCD );
    
    if counter == 0
        M21_step = M21 - lr * G1;
    else
        M21_step = M21_updated - lr * G1;
    end
    
    if norm(M21_step) < sqrt(M22_min_eig*m11)
        M21_updated = M21_step;
    else
        M21_updated = (M21_step/norm(M21_step)) * sqrt(M22_min_eig*m11);
    end
    
    if sum(abs(M21_updated))>max_od % invalid result, not going to update M
        return
    end
    
    M(BCD,remaining_idx)=real(M21_updated');
    M(remaining_idx,BCD)=M(BCD,remaining_idx);
    
    [ L ] = optimization_M_set_L_Mahalanobis( partial_sample, c, M );
    objective_current = x_original' * L * x_original;
    
    if objective_current>objective_previous
        lr=lr/2;
    else
        lr=lr*(1+1e-2);
    end
    tol = norm(objective_current - objective_previous);
    counter = counter + 1;
    objective_previous = objective_current;
    if counter>1e3% not converged
        break
    end
end

lower_bounds(BCD)=sum(abs(M21_updated));

end