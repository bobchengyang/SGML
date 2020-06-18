function [ class_temp_binary, class_temp, ...
    GT_obj_all, obj_all, error_iter,obj_vec,time_vec] = ...
    optimization_M_classification_main_workflow_wei_partial( class_test, ...
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

% M = eye(n_feature);
%
% [ L ] = optimization_M_set_L_Mahalanobis_tt( feature_train_test, M ); % full observation
%
% % knn_size = 5;
% % %========KNN classifier starts========
% % fl = class_train_test(initial_label_index);
% % fl(fl == -1) = 0;
% % x = KNN(fl, feature_train_test(initial_label_index,:), sqrtm(M), knn_size, feature_train_test(~initial_label_index,:));
% % x(x==0) = -1;
% % x_valid = class_train_test;
% % x_valid(~initial_label_index) = x;
% % %=========KNN classifier ends=========
%
% %=======Graph classifier starts=======
% cvx_begin
% variable x(n_sample,1);
% minimize(x'*L*x)
% subject to
% x(initial_label_index) == class_train_test(initial_label_index);
% cvx_end
%
% x_valid = sign(x);
% %========Graph classifier ends========
%
% diff_label = x_valid - class_train_test;
% error_classifier = size(find(diff_label~=0),1)*size(find(diff_label~=0),2)/size(class_test,1);
%
% disp(['objective before metric learning : ' num2str(x_valid'*L*x_valid)]);
% disp(['error rate before metric learning : ' num2str(error_classifier)]);

partial_feature = feature_train_test(initial_label_index,:);
partial_observation = class_train_test(initial_label_index);
partial_sample=length(partial_observation);

run_t=1;
time_vec=zeros(run_t,1);
obj_vec=zeros(run_t,1);
lobpcg_random_control=0;
for time_i=1:run_t
    
    tStart=tic;
    M=optimization_M_initialization(n_feature,1);
    dia_idx=find(M==diag(M));
    lower_bounds=M;
    lower_bounds(dia_idx)=0;
    lower_bounds=sum(abs(lower_bounds),2);
    
    [partial_feature,P] = mcml_get_variables_ready(partial_feature, partial_observation, partial_sample);
    lr=.1/partial_sample;%start with a relatively large learning rate
    
    counter_diag_nondiag = 0;
    
    tol_diag_nondiag = Inf;
    
    nc=0;
    flag=0;
    while tol_diag_nondiag > tol_main
        
        if counter_diag_nondiag == 0
            [initial_objective,Q] = mcml_obj(M, partial_feature, P, partial_sample);
            disp(['inintial = ' num2str(initial_objective)]);
        end
        
        initial_objective0=initial_objective;
        
        [ M, lr, Q, initial_objective0] = optimization_M_Block_CDdia_corrected(...
            partial_sample,...
            P,...
            lr,...
            lower_bounds,...
            n_feature,...
            M,...
            S_upper,...
            tol_diagonal,...
            partial_feature,...
            dia_idx,...
            Q,...
            initial_objective0);
        
        for BCD = 1:n_feature
            remaining_idx=1:n_feature;
            remaining_idx(BCD)=[];
            m11=M(BCD,BCD);
            M12=M(BCD,remaining_idx);
            M22=M(remaining_idx,remaining_idx);
            M21 = M12';
            
        %if n_feature<300
        %    M22_min_eig_lanczos = min(eigs(M22,1,'sm')); % lanczos algorithm
        %    M22_prime = M22 - M22_min_eig_lanczos * eye(n_feature-1);
        %    M22_min_eig_negative = lambdaBoundSC(M22_prime);
        %    M22_min_eig = M22_min_eig_lanczos + M22_min_eig_negative;
        %else
            %M22_min_eig=min(eig(M22));
%             tmineig=tic;
%             if n_feature==2
%                 M22_min_eig=M22;
%             else
%                 M22_min_eig=eigs(M22,1,'smallestabs','FailureTreatment','keep');
%             end
%             toc(tmineig);
if n_feature==2
    M22_min_eig=M22;
else
    if flag==0
        flag=1;
        rng(lobpcg_random_control);
        [mce,~] = ...
            optimization_M_lobpcg(randn(n_feature-1,1),M22,1e-4,200);
    else
        [mce,~] = ...
            optimization_M_lobpcg(mce,M22,1e-4,200);
    end
    M22_min_eig=mce'*M22*mce;
end
            
            [ M,lr,lower_bounds,Q,initial_objective0] = optimization_M_Block_CD(...
                partial_sample,...
                n_feature,...
                P,...
                lr,...
                m11,...
                M21,...
                M22_min_eig,...
                M,...
                BCD,...
                tol_offdiagonal,...
                remaining_idx,...
                lower_bounds,...
                partial_feature,...
                Q,...
                initial_objective0,...
                dia_idx);
            
        end
        
        counter_diag_nondiag = counter_diag_nondiag + 1;
        [current_objective,Q] = mcml_obj(M, partial_feature, P, partial_sample);
        tol_diag_nondiag = norm(current_objective - initial_objective);
        
        initial_objective = current_objective;
        if counter_diag_nondiag>1e3 % not converged
            nc=1;
            break
        end
    end
    
    if nc==1
        disp(['not converged = ' num2str(current_objective)]);
    else
        disp(['converged = ' num2str(current_objective)]);
    end
    
    time_vec(time_i)=toc(tStart);
    obj_vec(time_i)=current_objective;
    min(eig(M))
    
end

disp(['time_vec mean: ' num2str(mean(time_vec)) ' std:' num2str(std(time_vec))]);
disp(['obj_vec mean: ' num2str(mean(obj_vec)) ' std:' num2str(std(obj_vec))]);

% [ L ] = optimization_M_set_L_Mahalanobis_tt( feature_train_test, M ); % full observation
%
% % knn_size = 5;
% % %========KNN classifier starts========
% % fl = class_train_test(initial_label_index);
% % fl(fl == -1) = 0;
% % x = KNN(fl, feature_train_test(initial_label_index,:), sqrtm(M), knn_size, feature_train_test(~initial_label_index,:));
% % x(x==0) = -1;
% % x_valid = class_train_test;
% % x_valid(~initial_label_index) = x;
% % %=========KNN classifier ends=========
%
% %=======Graph classifier starts=======
% cvx_begin
% variable x(n_sample,1);
% minimize(x'*L*x)
% subject to
% x(initial_label_index) == class_train_test(initial_label_index);
% cvx_end
%
% x_valid = sign(x);
% %========Graph classifier ends========
%
% diff_label = x_valid - class_train_test;
% error_classifier = size(find(diff_label~=0),1)*size(find(diff_label~=0),2)/size(class_test,1);
% disp(['objective after metric learning : ' num2str(x_valid'*L*x_valid)]);
% disp(['error rate after metric learning : ' num2str(error_classifier)]);
%
% class_temp_binary = sign(x_valid);
% class_temp_binary(initial_label_index) = [];
%
% class_temp = zeros(size(class_temp_binary,1),size(class_temp_binary,2));
% class_temp(class_temp_binary==1) = class_i;
% class_temp(class_temp_binary==-1) = class_j;

class_temp_binary=class_train_test;
class_temp_binary(initial_label_index) = [];
class_temp=class_temp_binary;
end