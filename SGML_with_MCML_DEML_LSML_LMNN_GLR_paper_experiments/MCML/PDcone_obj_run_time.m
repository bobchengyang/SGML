function [obj_vec,time_vec] = ...
    PDcone_obj_run_time( feature_train_test, ...
    initial_label_index, ...
    class_train_test)

partial_feature = feature_train_test(initial_label_index,:);
partial_observation = class_train_test(initial_label_index);
[partial_sample,n_feature]=size(partial_feature);
S_upper=n_feature;
tol_main=1e-5;
run_t=1;
time_vec=zeros(run_t,1);
obj_vec=zeros(run_t,1);
time_eig=zeros(run_t,1);

max_iter=1e3;
for time_i=1:run_t
    
    tStart=tic;
    M=optimization_M_initialization(n_feature,1);
    [partial_feature,P] = mcml_get_variables_ready(partial_feature, partial_observation, partial_sample);
    lr=.1/partial_sample;
    
    counter_diag_nondiag = 0;
    
    tol_diag_nondiag = Inf;
    
    while tol_diag_nondiag > tol_main
        
        if counter_diag_nondiag == 0
            [initial_objective,Q] = mcml_obj(M, partial_feature, P, partial_sample);
            disp(['initial = ' num2str(initial_objective)]);
        end
        
        [ M,...
            time_eig] = optimization_M_PDcone_y(...
            partial_sample,...
            n_feature,...
            P,...
            M,...
            S_upper,...
            time_eig,...
            time_i,...
            lr,...
            partial_feature,...
            Q);
        
        [current_objective,Q] = mcml_obj(M, partial_feature, P, partial_sample);
        
        if current_objective>initial_objective
            lr=lr/2;
        else
            lr=lr*(1+1e-2);
        end
        
        tol_diag_nondiag = norm(current_objective - initial_objective);
        
        initial_objective = current_objective;
        
        counter_diag_nondiag = counter_diag_nondiag + 1;
        
        if counter_diag_nondiag==max_iter
            break
        end
    end
    disp(['converged = ' num2str(current_objective)]);
    
    time_vec(time_i)=toc(tStart);
    obj_vec(time_i)=current_objective;
end

disp(['time_vec mean: ' num2str(mean(time_vec)) ' std:' num2str(std(time_vec))]);
disp(['obj_vec mean: ' num2str(mean(obj_vec)) ' std:' num2str(std(obj_vec))]);
end