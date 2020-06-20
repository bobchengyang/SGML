function [obj_vec,time_vec] = ...
    HBNB_obj_run_time( feature_train_test, ...
    initial_label_index, ...
    class_train_test)

partial_feature = feature_train_test(initial_label_index,:);
partial_observation = class_train_test(initial_label_index);
[partial_sample,n_feature]=size(partial_feature);
tol_main=1e-5;
tol_diagonal=1e-3;
tol_offdiagonal=1e-3;
S_upper=n_feature;

run_t=1;
time_vec=zeros(run_t,1);
obj_vec=zeros(run_t,1);

for time_i=1:run_t

    tStart=tic;
    A=optimization_M_initialization(n_feature,1);
    [D,length_D] = get_S_and_D(partial_sample, partial_feature, partial_observation);
    
    initial_objective=dml_obj(A, D);
    
    disp(['initial = ' num2str(initial_objective)]);
    
    [A] = ...
        HBNB_core(...
        partial_sample,...
        n_feature,...
        A,...
        D,...
        S_upper,...
        tol_main,...
        tol_diagonal,...
        tol_offdiagonal,...
        length_D);
    
    current_objective=dml_obj(A, D);

    disp(['converged = ' num2str(current_objective)]);

    time_vec(time_i)=toc(tStart);
    obj_vec(time_i)=current_objective;
    min(eig(A))
end

disp(['time_vec mean: ' num2str(mean(time_vec)) ' std:' num2str(std(time_vec))]);
disp(['obj_vec mean: ' num2str(mean(obj_vec)) ' std:' num2str(std(obj_vec))]);
end