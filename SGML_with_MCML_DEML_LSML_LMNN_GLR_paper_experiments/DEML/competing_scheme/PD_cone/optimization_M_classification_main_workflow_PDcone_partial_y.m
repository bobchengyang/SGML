function [ class_temp_binary, class_temp, ...
    GT_obj_all, obj_all, error_iterjanitor] = ...
    optimization_M_classification_main_workflow_PDcone_partial_y( class_test, ...
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

%SDP_BINARY_GU_oao Summary of this function goes here
%   Detailed explanation goes here

%% set L (graph Laplacian (based on features that are drived from body part trajectories))

GT_obj_all = 0;
obj_all = 0;
error_iter = 0;

%% check error rate before metric learning starts

[n_sample, n_feature]= size(feature_train_test); %get the number of samples and the number of features

M = zeros(n_feature);

M(logical(eye(n_feature))) = S_upper/n_feature;

[ L ] = optimization_M_set_L_Mahalanobis_tt( feature_train_test, M ); % full observation

cvx_begin
variable x(n_sample,1);
minimize(x'*L*x)
subject to
x(initial_label_index) == class_train_test(initial_label_index);
cvx_end

x_valid = sign(x);

diff_label = x_valid - class_train_test;
error_classifier = size(find(diff_label~=0),1)*size(find(diff_label~=0),2)/size(class_test,1);

disp(['objective before metric learning : ' num2str(x_valid'*L*x_valid)]);
disp(['error rate before metric learning : ' num2str(error_classifier)]);

%% check error rate before metric learning ends

partial_feature = feature_train_test(initial_label_index,:);
partial_observation = class_train_test(initial_label_index);

run_t=3;
time_vec=zeros(run_t,1);
obj_vec=zeros(run_t,1);
lobpcg_random_control=0;
for time_i=1:run_t
    tic;
    
    counter_diag_nondiag = 0;
    
    tol_diag_nondiag = Inf;
    
    lr=.1/length(partial_observation);
    
    while tol_diag_nondiag > tol_main
        
        if counter_diag_nondiag == 0
            
            M=optimization_M_initialization(n_feature,0);
            initial_objective = gmml_obj(M,partial_feature,partial_observation);
            disp(['initial = ' num2str(initial_objective)]);
            
        end
        
        G1 = gmml_gradient(M,partial_feature,partial_observation);
        [ M_lc ] = optimization_M_PDcone_y(M,S_upper,lr,G1);
        current_objective = gmml_obj(M_lc,partial_feature,partial_observation);
   
        while current_objective>initial_objective
            if lr<1e-20
                break
            end
            lr=lr/2;
            [ M_lc ] = optimization_M_PDcone_y(M,S_upper,lr,G1);
            current_objective = gmml_obj(M_lc,partial_feature,partial_observation);
        end
        
        if current_objective>initial_objective
            current_objective=initial_objective;
            break
        end
        
        tol_diag_nondiag = norm(current_objective - initial_objective);
        %disp(['current = ' num2str(current_objective) ' tol: ' num2str(tol_diag_nondiag)]);
        initial_objective = current_objective;
        
        counter_diag_nondiag = counter_diag_nondiag + 1;
        
        M=M_lc;
    end
    disp(['converged = ' num2str(current_objective)]);
    
    time_vec(time_i)=toc;
    obj_vec(time_i)=current_objective;
end

disp(['time_vec mean: ' num2str(mean(time_vec)) ' std:' num2str(std(time_vec))]);
disp(['obj_vec mean: ' num2str(mean(obj_vec)) ' std:' num2str(std(obj_vec))]);

[ L ] = optimization_M_set_L_Mahalanobis_tt( feature_train_test, M ); % full observation

cvx_begin
variable x(n_sample,1);
minimize(x'*L*x)
subject to
x(initial_label_index) == class_train_test(initial_label_index);
cvx_end

x_valid = sign(x);

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