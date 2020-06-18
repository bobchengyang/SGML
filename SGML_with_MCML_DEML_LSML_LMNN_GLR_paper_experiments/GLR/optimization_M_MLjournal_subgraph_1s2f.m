function [ class_temp_binary, class_temp, ...
    GT_obj_all, obj_all, error_iter] = ...
    optimization_M_MLjournal_subgraph_1s2f( class_test, ...
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

%% proposed method: FULL + alternately Diagonal and one row/column of Off-diagonal + FULL
GT_obj_all = 0;
obj_all = 0;
error_iter = 0;
%% check error rate before metric learning starts
[n_sample, n_feature]= size(feature_train_test); %get the number of samples and the number of features
M = zeros(n_feature);
M(logical(eye(n_feature)))=S_upper/n_feature;
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

disp(['objective before metric learning : ' num2str(x_valid'*L*x_valid)]);
disp(['error rate before metric learning : ' num2str(error_classifier)]);

%% check error rate before metric learning ends

partial_feature = feature_train_test(initial_label_index,:);
partial_observation = class_train_test(initial_label_index);

%% lobpcg random control starts
lobpcg_random_control=0;
% planning = norm 9 non-norm 0
% breast-cancer = norm 1 non-norm 2
% heart = norm 7 non-norm 4
% iris = norm 8 non-norm 0
% seeds = norm 6 non-norm 0
% wine = norm 2 non-norm 1
%% lobpcg random control ends
tol_golden_search=1e-6;
run_t=10;
time_vec=zeros(run_t,1);
obj_vec=zeros(run_t,1);
for time_i=1:run_t
    tic;
    
    counter_diag_nondiag = 0;
    
    tol_diag_nondiag = 1e+4;
    
    while tol_diag_nondiag > tol_main
        
        if counter_diag_nondiag == 0
            
            M = zeros(n_feature);
            M(logical(eye(n_feature))) = S_upper/n_feature;
                        to_be_defined = epsilon*S_upper/n_feature;
                        for di=1:n_feature
                            for dj=1:n_feature
                                if abs(di-dj)==1
                                    M(di,dj)=to_be_defined;
                                end
                            end
                        end
%             for di=1:n_feature
%                 for dj=1:n_feature
%                     if di<dj %&& abs(di-dj)==1
% %                         if league_vec(di)~=league_vec(dj)
% %                             M(di,dj)=abs(.1*randn(1));
% %                             %M(di,dj)=epsilon;
% %                             M(dj,di)=M(di,dj);
% %                         else
% %                             M(di,dj)=-abs(.1*randn(1));
% %                             %M(di,dj)=-epsilon;
% %                             M(dj,di)=M(di,dj);
% %                         end
%                         if mod(abs(di-dj),2)==1
%                             M(di,dj)=abs(.1*randn(1));
%                             %M(di,dj)=epsilon;
%                             M(dj,di)=M(di,dj);
%                         else
%                             M(di,dj)=-abs(.1*randn(1));
%                             %M(di,dj)=-epsilon;
%                             M(dj,di)=M(di,dj);
%                         end
%                     end
%                 end
%             end
            rng(lobpcg_random_control);
            [M_current_eigenvector] = ...
                optimization_M_lobpcg(randn(n_feature,1),M,1e-12,200);
            
            [ L ] = optimization_M_set_L_Mahalanobis( partial_feature, M );
            initial_objective = partial_observation' * L * partial_observation;
            disp(['initial = ' num2str(initial_objective)]);
            
            %% 19-NOV-2019 assign BLUE/RED league for the nodes starts
            
            % BLUE is 1
            % RED is -1
            % initialize the graph as blue-red-blue-red
            
            league_vec = ones(n_feature,1);
            for ni=2:2:length(league_vec)
                league_vec(ni) = league_vec(ni)*-1;
            end
            
            %% 19-NOV-2019 assign BLUE/RED league for the nodes ends
            
            bins=ones(1,n_feature); % connected graph
            
            scaled_M=zeros(n_feature);
            scaled_factors=zeros(n_feature);
            
            for bins_i = 1:length(unique(bins))
                M_current = M(bins==bins_i,bins==bins_i);
                temp_dim=size(M_current,1);
                if temp_dim~=1
                    [M_current_eigenvector] = ...
                        optimization_M_lobpcg(M_current_eigenvector,M_current,1e-12,200);
                    
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
            
            [ M, ~, ~, M_current_eigenvector,...
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
        
        [ M ] = ...
            optimization_M_Block_CDdiaLP_subgraph(...
            partial_feature,...
            n_feature,...
            M,...
            partial_observation,...
            S_upper,...
            rho,...
            tol_diagonal,...
            bins,...
            lobpcg_random_control,...
            tol_golden_search);
        
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
        
        for BCD = 1:n_feature
            remaining_idx=1:n_feature;
            remaining_idx(BCD)=[];
            m11=M(BCD,BCD);
            M21=M(remaining_idx,BCD);
                
            [ M, scaled_M, scaled_factors,...
                league_vec,...
                bins] = ...
                optimization_M_Block_CDLPt_blue_red_MLjournal_subgraph_c(partial_feature,...
                n_feature,...
                m11,...
                partial_observation,...
                M21,...
                M,...
                BCD,...
                scaled_factors,...
                scaled_M,...
                rho,...
                tol_offdiagonal,...
                league_vec,...
                bins,...
                lobpcg_random_control,...
                tol_golden_search);
            
        end

        [ L ] = optimization_M_set_L_Mahalanobis( partial_feature, M );
        
        counter_diag_nondiag = counter_diag_nondiag + 1;
        
        current_objective = partial_observation' * L * partial_observation;
        
        disp(['current objective = ' num2str(current_objective)]);
        
        tol_diag_nondiag = norm(current_objective - initial_objective);
        
        initial_objective = current_objective;
        
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