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
    nv=n_feature+(n_feature*(n_feature-1)/2);
    M=optimization_M_initialization(n_feature,1);
    [S,D] = lsml_S_D(partial_observation,partial_sample,partial_feature);
    lr=.1/partial_sample;%finding a good choice of a step size is out of the scope of the paper
    counter_diag_nondiag = 0;
    
    tol_diag_nondiag = Inf;
    
    while tol_diag_nondiag > tol_main
        
        if counter_diag_nondiag == 0
            
            initial_objective = lsml_obj(M,S,D);
            disp(['initial = ' num2str(initial_objective)]);
            
        end
        
        [ M,...
            time_eig] = optimization_M_PDcone_y(...
            S,...
            D,...
            M,...
            S_upper,...
            time_eig,...
            time_i,...
            lr,...
            n_feature,...
            nv);
        
        current_objective = lsml_obj(M,S,D);
        %disp(['obj = ' num2str(current_objective)]);
        if current_objective==0
            break
        end
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