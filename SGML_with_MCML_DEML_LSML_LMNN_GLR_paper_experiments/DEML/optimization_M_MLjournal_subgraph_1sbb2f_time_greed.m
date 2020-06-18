function [ league_vec,class_temp_binary, class_temp, ...
    GT_obj_all, obj_all, error_iter] = ...
    optimization_M_MLjournal_subgraph_1sbb2f_time_greed( class_test, ...
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

%% proposed method: FULL + Diagonal and one row/column of Off-diagonal + FULL
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

%partial_feature = feature_train_test(initial_label_index,:);
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

tol_golden_search=1e0;
run_t=1;
time_vec=zeros(run_t,1);
obj_vec=zeros(run_t,1);
for time_i=1:run_t
    tic;
    
    for fn=2:n_feature
        %fn
        partial_feature=feature_train_test(initial_label_index,1:fn);
        
        if fn==2
            league_vec=ones(fn,1);
            M=eye(fn);
            for i=1:fn
                for j=1:fn
                    if i<j
                        M(i,j)=-1/fn*abs(randn(1));
                        M(j,i)=M(i,j);
                    end
                end
            end
            bins=ones(1,fn);
        else
            league_vec=[league_vec;1];
            M=eye(fn);
            for di=1:fn
                for dj=1:fn
                    if di<dj %&& abs(di-dj)==1
                        if league_vec(di)~=league_vec(dj)
                            M(di,dj)=1/fn*abs(randn(1));
                            M(dj,di)=M(di,dj);
                        else
                            M(di,dj)=-1/fn*abs(randn(1));
                            M(dj,di)=M(di,dj);
                        end
                    end
                end
            end
            bins=ones(1,fn);
        end
        
        [M_blue,obj_blue] = optimization_M_blue_red(...
            fn,...
            partial_feature,...
            lobpcg_random_control,...
            epsilon,...
            fn,...
            partial_observation,...
            tol_offdiagonal,...
            rho,...
            league_vec,...
            tol_golden_search,...
            bins,...
            M);
        
        league_vec(fn)=-1;
        [M_red,obj_red] = optimization_M_blue_red(...
            fn,...
            partial_feature,...
            lobpcg_random_control,...
            epsilon,...
            fn,...
            partial_observation,...
            tol_offdiagonal,...
            rho,...
            league_vec,...
            tol_golden_search,...
            bins,...
            M);
        
        if obj_blue < obj_red
            league_vec(fn)=1;
            M=M_blue;
        else
            league_vec(fn)=-1;
            M=M_red;
        end
        
    end
    
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