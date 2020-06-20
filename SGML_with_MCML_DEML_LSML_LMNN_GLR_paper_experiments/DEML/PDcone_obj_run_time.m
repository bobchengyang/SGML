function [obj_vec,time_vec] = ...
    PDcone_obj_run_time( feature_train_test, ...
    initial_label_index, ...
    class_train_test)

partial_feature = feature_train_test(initial_label_index,:);
partial_observation = class_train_test(initial_label_index);
[partial_sample,n_feature]=size(partial_feature);
S_upper=n_feature;

run_t=1;
time_vec=zeros(run_t,1);
obj_vec=zeros(run_t,1);
max_iter=1e3;

for time_i=1:run_t
    
    tStart=tic;
    A=optimization_M_initialization(n_feature,1);
    
    [D,length_D] = get_S_and_D(partial_sample, partial_feature, partial_observation);
    
    initial_objective=dml_obj(A, D);    
    
    disp(['initial = ' num2str(initial_objective)]);
    
    
    %% eric code starts
    d=n_feature;
    nv=d+(d*(d-1)/2);
    count=0;
    lr = 0.1/partial_sample;         % initial step size along gradient
    
    DML_tol=Inf;
    tol_c1c2=1e-5;
    total_grad=0;
    total_eig=0;
    while DML_tol>tol_c1c2
        
        %grad2 = fS1(X, S, A, N, d, nv, 0, 0);
        %grad1 = fD1(X, D, A, N, d, nv, 0, 0);
        %Eric_grad = grad_projection(grad1, grad2, d);
        tgrad=tic;
        Eric_grad = fD1(D, A, d, nv, 0, 0,length_D);
        total_grad=total_grad+toc(tgrad);
        A = A + lr*Eric_grad;
        A = (A + A')/2;
        
        teig=tic;
        [V, L] = eig(A); % eigen-decomposition of M
        ind=find(diag(L)>0);
        A=V(:,ind) * L(ind,ind) * V(:,ind)';
        total_eig=total_eig+toc(teig);
        
        % scale A if needed
        if sum(diag(A))>S_upper
            ft=sum(diag(A))/S_upper;
            A=A/ft;
        end
        
        objective_current=dml_obj(A, D);
        
        if objective_current>initial_objective || count==0
            lr=lr*(1+1e-2);
        else
            lr=lr/2;
        end
        
        DML_tol=norm(objective_current-initial_objective);
        %disp(['tol: ' num2str(DML_tol)]);
        
        initial_objective=objective_current;
        count=count+1;
        if count==max_iter
            break
        end
    end
    %% eric code ends
    
    current_objective=dml_obj(A, D);  
    disp(['converged = ' num2str(current_objective)]);
    
    time_vec(time_i)=toc(tStart);
    obj_vec(time_i)=current_objective;
    min(eig(A))
end

disp(['time_vec mean: ' num2str(mean(time_vec)) ' std:' num2str(std(time_vec))]);
disp(['obj_vec mean: ' num2str(mean(obj_vec)) ' std:' num2str(std(obj_vec))]);

end