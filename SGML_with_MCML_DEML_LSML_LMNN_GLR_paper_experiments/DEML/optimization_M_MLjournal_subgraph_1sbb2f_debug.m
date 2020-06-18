function [ class_temp_binary, class_temp, ...
    GT_obj_all, obj_all, error_iter] = ...
    optimization_M_MLjournal_subgraph_1sbb2f_debug( class_test, ...
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
partial_sample = length(partial_observation);

flag = 0;

counter_diag_nondiag = 0;

tol_diag_nondiag = 1e+4;
obj_list = [];
while tol_diag_nondiag > tol_main
    
    if flag == 0
        flag = 1;
        
        M = zeros(n_feature);
        
        M(logical(eye(n_feature))) = S_upper/n_feature;
        
        current_location_actual = zeros(n_feature,1);
        current_location_ordered = zeros(n_feature,1);
        odub = zeros(n_feature);
        to_be_defined = epsilon*S_upper/n_feature;
        
        for di=1:n_feature
            for dj=1:n_feature
                if abs(di-dj)==1
                    M(di,dj)=to_be_defined;
                end
            end
        end
        
        %         M
        odub_initial = odub;
        
        [M_current_eigenvector,M_min_eig,failureflag,~,residual_norm_history0] = ...
            optimization_M_lobpcg(randn(size(M,1),1),M,1e-12,200);
        
        [ L ] = optimization_M_set_L_Mahalanobis( partial_feature, M ); % full observation        
        initial_objective = partial_observation' * L * partial_observation;
        disp(['current objective = ' num2str(initial_objective)]);
        
        %% 19-NOV-2019 assign BLUE/RED league for the nodes starts
        
        % BLUE is 1
        % RED is -1
        % set the non-first nodes to be in the BLue league
        
        league_vec = ones(n_feature,1);
        for ni=2:2:length(league_vec)
            league_vec(ni) = league_vec(ni)*-1;
        end
        %         % league_vec
        %         league_vec(2:end) = [1 1 1]; % BLUE is 1
        
        %% 19-NOV-2019 assign BLUE/RED league for the nodes ends
        
        %% check connectivity begins MAY-5-2020
        
        % thresholding
        bins=ones(1,n_feature);
        % toc;
        %% check connectivity ends MAY-5-2020
        
        leftEnd=zeros(size(M,1),1);
        scaled_M=zeros(size(M));
        scaled_factors=zeros(size(M));
        scaled_factors_check=zeros(n_feature);
        
        for bins_i = 1:length(unique(bins))
            M_current = M(bins==bins_i,bins==bins_i);
            temp_dim=size(M_current,1);
            if temp_dim~=1
                [M_current_eigenvector,M_min_eig,failureflag,~,residual_norm_history0] = ...
                    optimization_M_lobpcg(randn(size(M_current,1),1),M_current,1e-12,200);
                %[vvv,ddd] = eig(M);
                %             tic;
                %             scaling_matrix_0 = inv(diag(M_current_eigenvector(:,1)));
                %             scaled_M_0 = scaling_matrix_0 * M_current * inv(scaling_matrix_0);
                %             scaled_factors_0 = scaling_matrix_0 * ones(size(M_current)) * inv(scaling_matrix_0);
                scaling_matrix_0 = diag(1./M_current_eigenvector(:,1));
                scaling_matrix_0_inv = diag(M_current_eigenvector(:,1));
                scaled_M_0 = scaling_matrix_0 * M_current * scaling_matrix_0_inv;
                scaled_factors_0 = scaling_matrix_0 * ones(temp_dim) * scaling_matrix_0_inv;
                %             toc;
                scaled_M_offdia_0 = scaled_M_0;
                scaled_M_offdia_0(scaled_M_offdia_0==diag(scaled_M_offdia_0))=0;
                leftEnd_0 = diag(scaled_M_0) - sum(abs(scaled_M_offdia_0),2);
                
                leftEnd_diff = sum(abs(leftEnd_0 - mean(leftEnd_0)));
                if leftEnd_diff > 1e-3
                    %if not_aligned > 0
                    
                    disp('Off-diagonal left ends not aligned!!!!');
                    leftEnd_0
                    asdf = 1;
                    
                end
                scaled_M(bins==bins_i,bins==bins_i)=scaled_M_0;
                scaled_factors(bins==bins_i,bins==bins_i)=scaled_factors_0;
                scaled_factors_check(bins==bins_i,bins==bins_i)=1;
                
                leftEnd(bins==bins_i)=leftEnd_0;
            else
                scaled_M(bins==bins_i,bins==bins_i)=M_current;
                scaled_factors(bins==bins_i,bins==bins_i)=1;
                scaled_factors_check(bins==bins_i,bins==bins_i)=1;
                
                leftEnd(bins==bins_i)=rho;
            end
            
            %M(bins==bins_i,bins==bins_i)=M_current;
            
        end
        %     scaled_factors(~logical(scaled_factors_check))=1;
        scaled_M_=scaled_M;
        scaled_M_(logical(eye(size(M,1))))=0;
        lower_bounds = sum(abs(scaled_M_),2) + rho;
        
        if sum(lower_bounds) > S_upper
            disp(['lower bounds sum:' num2str(sum(lower_bounds))]);
            disp('========lower bounds sum larger than S_upper!!!========');
            return
            
        end
        
        % get a first pass of the full matrix optimization while fixing the initial graph starts
        PN_temp=[];
        [ M, scaled_M, scaled_factors, M_current_eigenvector, PN_temp, odub,...
            league_vec,...
            bins,...
            lower_bound] = ...
            optimization_M_MLjournal_subgraph_nocycle_after(partial_feature,...
            n_feature,partial_sample,...
            partial_observation,...
            M,...
            odub,...
            scaled_factors,...
            scaled_M,...
            counter_diag_nondiag,...
            rho,...
            tol_offdiagonal,...
            M_current_eigenvector,...
            PN_temp,...
            to_be_defined,...
            odub_initial,...
            current_location_actual,...
            current_location_ordered,...
            leftEnd,...
            league_vec,...
            S_upper);
        % get a first pass of the full matrix optimization while fixing the initial graph ends
        
    end
    
    %% block coordinate dcsent
    %% update diagonals AND one row/column of off-diagonals
    
    
    
    
    %% update off-diagonals
    
    for BCD = 1:n_feature
        
        if BCD == 1
            m11 = M(1,1);
            M12 = M(1,2:end);
            M22 = M(2:end,2:end);
        elseif BCD == n_feature
            m11 = M(end,end);
            M12 = M(end,1:end-1);
            M22 = M(1:end-1,1:end-1);
        else
            m11 = M(BCD,BCD);
            M12 = M(BCD,[1:BCD-1 BCD+1:end]);
            M22 = M([1:BCD-1 BCD+1:end],[1:BCD-1 BCD+1:end]);
        end
        M21 = M12';
        
        %% update off-diagonals
        
            PN_temp = [];
            
        %         tic;
        [ M_updated, scaled_M, scaled_factors, M_current_eigenvector, PN_temp, odub,...
            league_vec,...
            bins,...
            lower_bounds] = ...
            optimization_M_Block_CDLPt_blue_red_MLjournal_subgraph_1sbb2f(partial_feature,...
            n_feature,partial_sample,...
            m11,...
            partial_observation,...
            M21,...
            M22,...
            M,...
            BCD,...
            odub,...
            scaled_factors,...
            scaled_M,...
            counter_diag_nondiag,...
            rho,...
            tol_offdiagonal,...
            M_current_eigenvector,...
            PN_temp,...
            to_be_defined,...
            odub_initial,...
            current_location_actual,...
            current_location_ordered,...
            leftEnd,...
            league_vec,...
            S_upper,...
            bins,...
            lower_bounds);
        %         toc;
        
        
        M = M_updated;
        
        
    end
    %    toc;
    [ L ] = optimization_M_set_L_Mahalanobis( partial_feature, M );
    counter_diag_nondiag = counter_diag_nondiag + 1;
    
    current_objective = partial_observation' * L * partial_observation;
    
    disp(['current objective = ' num2str(current_objective)]);
    
    tol_diag_nondiag = norm(current_objective - initial_objective);
    
    initial_objective = current_objective;
    
    %     if counter_diag_nondiag >= 1
    %         obj_list = [obj_list;current_objective];
    %         if counter_diag_nondiag > 4
    %             obj_list(1)=[];
    %
    %         end
    %     end
    %
    %     if counter_diag_nondiag > 4
    %         %         abs(obj_list(4)-obj_list(2))
    %         if abs(obj_list(4)-obj_list(2))<tol_main
    %             break
    %         end
    %     end
    %     M
end

[ M, scaled_M, scaled_factors, M_current_eigenvector, PN_temp, odub,...
    league_vec,...
    bins,...
    lower_bound] = ...
    optimization_M_MLjournal_subgraph_nocycle_after(partial_feature,...
    n_feature,partial_sample,...
    partial_observation,...
    M,...
    odub,...
    scaled_factors,...
    scaled_M,...
    counter_diag_nondiag,...
    rho,...
    tol_offdiagonal,...
    M_current_eigenvector,...
    PN_temp,...
    to_be_defined,...
    odub_initial,...
    current_location_actual,...
    current_location_ordered,...
    leftEnd,...
    league_vec,...
    S_upper);
[ L ] = optimization_M_set_L_Mahalanobis( partial_feature, M );
current_objective = partial_observation' * L * partial_observation;
disp(['current objective = ' num2str(current_objective)]);
% toc;

[ Wf ] = optimization_M_set_Wf_Mahalanobis( feature_train_test );
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

% figure(1);imagesc(M);axis square;axis off;
objective_interpolated = sign(x_valid)'*L*sign(x_valid);
objective_groundtruth = sign(class_train_test)'*L*sign(class_train_test);

%pause(0.01);

end