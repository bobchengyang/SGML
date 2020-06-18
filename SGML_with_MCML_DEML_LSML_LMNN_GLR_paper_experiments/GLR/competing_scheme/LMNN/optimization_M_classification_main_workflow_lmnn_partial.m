function [ class_temp_binary, class_temp, ...
    GT_obj_all, obj_all, error_iter] = ...
    optimization_M_classification_main_workflow_lmnn_partial( class_test, ...
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

M = zeros(n_feature);

for diagonal_M_i = 1:n_feature
    
    M(diagonal_M_i,diagonal_M_i) = S_upper/n_feature;
    
end

[ Wf ] = optimization_M_set_Wf_Mahalanobis( feature_train_test );
[ L ] = optimization_M_set_L_Mahalanobis( n_sample, Wf, M ); % full observation

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

%% check error rate before metric learning ends

%% feature pre-selection starts

% [feature_train_test, ...
%     selected_indices] = ...
%     feature_pre_selection(feature_train_test,...
%     initial_label_index,...
%     class_train_test,...
%     proportion_threshold,...
%     tol_set_prepro);
%
% [n_sample, n_feature]= size(feature_train_test); %get the number of samples and the number of features
%
% S_upper = n_feature;

%% feature pre-selection ends

partial_feature = feature_train_test(initial_label_index,:);
partial_observation = class_train_test(initial_label_index);
partial_sample = length(partial_observation);

flag = 0;

counter_diag_nondiag = 0;

tol_diag_nondiag = 1e+4;

[M, L_, Y_, C_] = lmnn(partial_feature, partial_observation);

factor_for_diag = sum(diag(M))/S_upper;
M = M/factor_for_diag;

[ Wf ] = optimization_M_set_Wf_Mahalanobis( partial_feature );
[L] = optimization_M_set_L_Mahalanobis( partial_sample, Wf, M );

initial_objective = partial_observation' * L * partial_observation;
disp(['current objective = ' num2str(initial_objective)]);

[ Wf ] = optimization_M_set_Wf_Mahalanobis( feature_train_test );
[ L ] = optimization_M_set_L_Mahalanobis( n_sample, Wf, M ); % full observation

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

% figure(1);imagesc(M);axis square;axis off;
objective_interpolated = sign(x_valid)'*L*sign(x_valid);
objective_groundtruth = sign(class_train_test)'*L*sign(class_train_test);
% pause(1);

end