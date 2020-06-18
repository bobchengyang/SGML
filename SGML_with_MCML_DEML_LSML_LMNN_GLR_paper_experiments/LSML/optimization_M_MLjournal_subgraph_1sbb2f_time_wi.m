function [ class_temp_binary, class_temp, ...
    GT_obj_all, obj_all, error_iter] = ...
    optimization_M_MLjournal_subgraph_1sbb2f_time_wi( class_test, ...
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
    M_wei)

%% proposed method: FULL + Diagonal and one row/column of Off-diagonal + FULL
GT_obj_all = 0;
obj_all = 0;
error_iter = 0;
%% check error rate before metric learning starts
[n_sample, n_feature]= size(feature_train_test); %get the number of samples and the number of features
[ L ] = optimization_M_set_L_Mahalanobis( feature_train_test, M_wei ); % full observation

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

%% lobpcg random control starts
lobpcg_random_control=0;
% planning = norm 6 non-norm 0
% breast-cancer = norm 1 non-norm 2
% heart = norm 7 non-norm 4
% iris = norm 8 non-norm 0
% seeds = norm 1 non-norm 0
% wine = norm 6 non-norm 1
%% lobpcg random control ends
tol_golden_search=1e-7;
run_t=1;
time_vec=zeros(run_t,1);
obj_vec=zeros(run_t,1);
for time_i=1:run_t
    tic;
    
    counter_diag_nondiag = 0;
    
    tol_diag_nondiag = 1e+4;
    
    while tol_diag_nondiag > tol_main
        
        if counter_diag_nondiag == 0
            
            %% initialize with Wei's result and make it balanced starts
            [M,league_vec,bins] = optimization_M_balancing_a_graph(M_wei,n_feature);
            %% initialize with Wei's result and make it balanced ends
            
            [ L ] = optimization_M_set_L_Mahalanobis( partial_feature, M );
            initial_objective = partial_observation' * L * partial_observation;
            disp(['initial = ' num2str(initial_objective)]);
            
            scaled_M=zeros(n_feature);
            scaled_factors=zeros(n_feature);
            
            for bins_i = 1:length(unique(bins))
                M_current = M(bins==bins_i,bins==bins_i);
                temp_dim=size(M_current,1);
                if temp_dim~=1
                    rng(lobpcg_random_control);
                    [M_current_eigenvector] = ...
                        optimization_M_lobpcg(randn(temp_dim,1),M_current,1e-12,200);
                    scaling_matrix_0 = diag(1./M_current_eigenvector(:,1));
                    scaling_matrix_0_inv = diag(M_current_eigenvector(:,1));
                    scaled_M_0 = scaling_matrix_0 * M_current * scaling_matrix_0_inv;
                    scaled_factors_0 = scaling_matrix_0 * ones(temp_dim) * scaling_matrix_0_inv;
                    
                    scaled_M(bins==bins_i,bins==bins_i)=scaled_M_0;
                    scaled_factors(bins==bins_i,bins==bins_i)=scaled_factors_0;
                    
                else
                    scaled_M(bins==bins_i,bins==bins_i)=M_current;
                    scaled_factors(bins==bins_i,bins==bins_i)=1;
                    
                end
            end
            
            [ M, scaled_M, scaled_factors, M_current_eigenvector,...
                league_vec,...
                bins] = ...
                optimization_M_MLjournal_subgraph_nocycle_after(partial_feature,...
                n_feature,...
                partial_observation,...
                M,...
                scaled_M,...
                bins,...
                scaled_factors,...
                rho,...
                tol_offdiagonal,...
                M_current_eigenvector,...
                league_vec,...
                lobpcg_random_control,...
                tol_golden_search);
            
        end
        
        for BCD = 1:n_feature
            
            if BCD == 1
                M12 = M(1,2:end);
            elseif BCD == n_feature
                M12 = M(end,1:end-1);
            else
                M12 = M(BCD,[1:BCD-1 BCD+1:end]);
            end
            M21 = M12';
            
            [ M_updated, scaled_M, scaled_factors, M_current_eigenvector, ...
                league_vec,...
                bins] = ...
                optimization_M_Block_CDLPt_blue_red_MLjournal_subgraph_1sbb2f(partial_feature,...
                n_feature,...
                partial_observation,...
                M21,...
                M,...
                BCD,...
                scaled_factors,...
                scaled_M,...
                rho,...
                tol_offdiagonal,...
                M_current_eigenvector,...
                league_vec,...
                S_upper,...
                bins,...
                lobpcg_random_control,...
                tol_golden_search);
            
            M = M_updated;
            
        end
        
        [ L ] = optimization_M_set_L_Mahalanobis( partial_feature, M );
        
        current_objective = partial_observation' * L * partial_observation;
        
        %disp(['current objective = ' num2str(current_objective)]);
        
        tol_diag_nondiag = norm(current_objective - initial_objective);
        
        initial_objective = current_objective;
        counter_diag_nondiag = counter_diag_nondiag + 1;
        
    end
    
    [ M, ~, ~, ~,...
        ~,...
        ~] = ...
        optimization_M_MLjournal_subgraph_nocycle_after(partial_feature,...
        n_feature,...
        partial_observation,...
        M,...
        scaled_M,...
        bins,...
        scaled_factors,...
        rho,...
        tol_offdiagonal,...
        M_current_eigenvector,...
        league_vec,...
        lobpcg_random_control,...
        tol_golden_search);
    
    [ L ] = optimization_M_set_L_Mahalanobis( partial_feature, M );
    current_objective = partial_observation' * L * partial_observation;
    disp(['converged = ' num2str(current_objective)]);
    
    time_vec(time_i)=toc;
    obj_vec(time_i)=current_objective;
    
end

disp(['time_vec mean: ' num2str(mean(time_vec)) ' std:' num2str(std(time_vec))]);
disp(['obj_vec mean: ' num2str(mean(obj_vec)) ' std:' num2str(std(obj_vec))]);

[ L ] = optimization_M_set_L_Mahalanobis( feature_train_test, M ); % full observation

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