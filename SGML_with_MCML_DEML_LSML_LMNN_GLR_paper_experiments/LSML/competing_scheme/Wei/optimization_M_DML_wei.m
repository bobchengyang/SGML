function [ class_temp_binary, class_temp, ...
    GT_obj_all, obj_all, error_iter] = ...
    optimization_M_DML_wei( class_test, ...
    feature_train_test, ...
    initial_label, ...
    initial_label_index, ...
    class_train_test, ...
    class_i, ...
    class_j, ...
    S_upper,...
    rho,...
    epsilon,...
    proportion_factor,...
    proportion_threshold,...
    tol_set_prepro,...
    tol_main,...
    tol_diagonal,...
    tol_offdiagonal,...
    step_scale,...
    step_scale_od)

GT_obj_all = 0;
obj_all = 0;
error_iter = 0;

%% check error rate before metric learning starts

[n_sample, n_feature]= size(feature_train_test); %get the number of samples and the number of features

M=eye(n_feature);

[ L ] = optimization_M_set_L_Mahalanobis_tt( feature_train_test, M ); % full observation

% knn_size = 5;
% %========KNN classifier starts========
% fl = class_train_test(initial_label_index);
% fl(fl == -1) = 0;
% x = KNN(fl, feature_train_test(initial_label_index,:), sqrtm(M), knn_size, feature_train_test(~initial_label_index,:));
% x(x==0) = -1;
% x_valid = class_train_test;
% x_valid(~initial_label_index) = x;
% %=========KNN classifier ends=========

%=======Graph classifier starts=======
cvx_begin
variable x(n_sample,1);
minimize(x'*L*x)
subject to
x(initial_label_index) == class_train_test(initial_label_index);
cvx_end

x_valid = sign(x);
%========Graph classifier ends========

diff_label = x_valid - class_train_test;
error_classifier = size(find(diff_label~=0),1)*size(find(diff_label~=0),2)/size(class_test,1);

disp(['objective before metric learning : ' num2str(x_valid'*L*x_valid)]);
disp(['error rate before metric learning : ' num2str(error_classifier)]);

partial_feature = feature_train_test(initial_label_index,:);
partial_observation = class_train_test(initial_label_index);
partial_sample=length(partial_observation);

run_t=3;
time_vec=zeros(run_t,1);
obj_vec=zeros(run_t,1);

for time_i=1:run_t

    tStart=tic;
    A=optimization_M_initialization(n_feature,1);
    [D,length_D] = get_S_and_D(partial_sample, partial_feature, partial_observation);
    
    initial_objective=dml_obj(A, D);
    
    disp(['initial = ' num2str(initial_objective)]);
    
    [A] = ...
        optimization_M_classification_main_workflow_wei_partial(...
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
    min(eig(M))
end

disp(['time_vec mean: ' num2str(mean(time_vec)) ' std:' num2str(std(time_vec))]);
disp(['obj_vec mean: ' num2str(mean(obj_vec)) ' std:' num2str(std(obj_vec))]);

[ L ] = optimization_M_set_L_Mahalanobis_tt( feature_train_test, M ); % full observation

% knn_size = 5;
% %========KNN classifier starts========
% fl = class_train_test(initial_label_index);
% fl(fl == -1) = 0;
% x = KNN(fl, feature_train_test(initial_label_index,:), sqrtm(M), knn_size, feature_train_test(~initial_label_index,:));
% x(x==0) = -1;
% x_valid = class_train_test;
% x_valid(~initial_label_index) = x;
% %=========KNN classifier ends=========

%=======Graph classifier starts=======
cvx_begin
variable x(n_sample,1);
minimize(x'*L*x)
subject to
x(initial_label_index) == class_train_test(initial_label_index);
cvx_end

x_valid = sign(x);
%========Graph classifier ends========

diff_label = x_valid - class_train_test;
error_classifier = size(find(diff_label~=0),1)*size(find(diff_label~=0),2)/size(class_test,1);
disp(['objective after metric learning : ' num2str(x_valid'*L*x_valid)]);
disp(['error rate after metric learning : ' num2str(error_classifier)]);

class_temp_binary = sign(x_valid);
class_temp_binary(initial_label_index) = [];

class_temp = zeros(size(class_temp_binary,1),size(class_temp_binary,2));
class_temp(class_temp_binary==1) = class_i;
class_temp(class_temp_binary==-1) = class_j;

end

