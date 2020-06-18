function [ class_temp_binary, class_temp, ...
    GT_obj_all, obj_all, error_iter] = ...
    optimization_M_classification_main_workflow_gmml_partial( class_test, ...
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
    tol_offdiagonal)

%SDP_BINARY_GU_oao Summary of this function goes here
%   Detailed explanation goes here

%% set L (graph Laplacian (based on features that are drived from body part trajectories))

GT_obj_all = 0;
obj_all = 0;
error_iter = 0;

%% check error rate before metric learning starts

[n_sample, n_feature]= size(feature_train_test); %get the number of samples and the number of features

M = eye(n_feature);

[ L ] = optimization_M_set_L_Mahalanobis( feature_train_test, M ); % full observation

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

run_t=3;
time_vec=zeros(run_t,1);
obj_vec=zeros(run_t,1);
lobpcg_random_control=0;
for time_i=1:run_t
    
    M=optimization_M_initialization(n_feature,0);

    initial_objective = gmml_obj(M, partial_feature, partial_observation);
    disp(['initial = ' num2str(initial_objective)]);
    tic;
    %setting the parameters of the metric learning method
    t = 0.65;
    %planning 0.9
    %breast-cancer 0.9
    %heart 0.765
    %iris 0.765
    %seeds 0.95
    %wine 0.9
    %sonar 0.8
    %madelon 0.65
    %colon-cancer 0.6
    params.const_factor = 40;
    params.tuning_num_fold = 5;

    M = feval(@(partial_observation,partial_feature) MetricLearning(partial_observation, partial_feature,...
        t, params), partial_observation, partial_feature);
    
    if sum(diag(M))>S_upper
        factor_for_diag = sum(diag(M))/S_upper;
        M = M/factor_for_diag;
    end
    
    current_objective = gmml_obj(M, partial_feature, partial_observation);
    disp(['converged = ' num2str(current_objective)]);
    
    time_vec(time_i)=toc;
    obj_vec(time_i)=current_objective;
    min(eig(M))
end

disp(['time_vec mean: ' num2str(mean(time_vec)) ' std:' num2str(std(time_vec))]);
disp(['obj_vec mean: ' num2str(mean(obj_vec)) ' std:' num2str(std(obj_vec))]);

[ L ] = optimization_M_set_L_Mahalanobis( feature_train_test, M ); % full observation

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