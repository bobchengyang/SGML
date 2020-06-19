function [error_classifier] = ...
    SGML_binary_classification( class_test, ...
    feature_train_test, ...
    initial_label_index, ...
    class_train_test, ...
    classifier_i)

%% check error rate before metric learning starts
[~, n_feature]= size(feature_train_test); %get the number of samples and the number of features

M = eye(n_feature);

[ L ] = graph_Laplacian_train_test( feature_train_test, M ); % full observation

if classifier_i==1 % 3-NN classifier
 knn_size = 3;
%========KNN classifier starts========
fl = class_train_test(initial_label_index);
fl(fl == -1) = 0;
x = KNN(fl, feature_train_test(initial_label_index,:), sqrtm(full(M)), knn_size, feature_train_test(~initial_label_index,:));
x(x==0) = -1;
x_valid = class_train_test;
x_valid(~initial_label_index) = x;
%=========KNN classifier ends=========   
elseif classifier_i==2 % Mahalanobis classifier
%=======Mahalanobis classifier starts========
[m,X] = ...
    mahalanobis_classifier_variables(...
    feature_train_test,...
    class_train_test,...
    initial_label_index);
z=mahalanobis_classifier(m,M,X);
x_valid=class_train_test;
x_valid(~initial_label_index)=z;
%========Mahalanobis classifier ends=========    
else % GLR-based classifier
%=======Graph classifier starts=======
cvx_begin
variable x(n_sample,1);
minimize(x'*L*x)
subject to
x(initial_label_index) == class_train_test(initial_label_index);
cvx_end
x_valid = sign(x);
%========Graph classifier ends========    
end

diff_label = x_valid - class_train_test;
error_classifier = size(find(diff_label~=0),1)*size(find(diff_label~=0),2)/size(class_test,1);

disp(['objective before metric learning : ' num2str(x_valid'*L*x_valid)]);
disp(['error rate before metric learning : ' num2str(error_classifier)]);

%% check error rate before metric learning ends

data_feature = feature_train_test(initial_label_index,:);
data_label= class_train_test(initial_label_index);

[n_sample,n_feature]=size(data_feature); % number of samples, number of features

%% set parameters
%=main parameters for SGML=================================================
C=n_feature; % constraint of the trace of M
rho=1e-5; % tolerance to make sure M is positive definite during optimization
lobpcg_random_control=0; % random seed for computing the first eigenvector for the first time using LOBPCG
options = optimoptions('linprog','Display','none','Algorithm','interior-point'); % linear program (LP) setting for Frank-Wolfe algorithm
options.OptimalityTolerance = 1e-2; % LP optimality tolerance
options.ConstraintTolerance = 1e-4; % LP interior-point constraint tolerance
FW_dia_offdia_tol=1e-3; % Frank-Wolfe tolerance when optimizing the diagonals + one row/column of off-diagonals
FW_full_tol=1e-5; % Frank-Wolfe tolerance when optimizing the full metric matrix M
max_iter=1e3; % maximum number of iterations for each round of Frank-Wolfe optimization
%==========================================================================

%=parameters for Frank-Wolfe step size optimization========================
GS_or_NR=2; % Frank-Wolfe step size optimization using 1) golden section search or 2) Newton-Raphson method
tol_golden_search=5e-1; % tolerance of golden section search
tol_NR=5e-1; % Newton-Raphson tolerance
tol_GD=5e-1; % gradient descent bisection tolerance
%==========================================================================

%=other parameters=========================================================
nv_od=2*n_feature-1; % number of LP variables when optimizing the diagonals + one row/column of off-diagonals
nv_full=n_feature+(n_feature*(n_feature-1))/2; % number of LP variables when optimizing the full metric matrix M
zz=logical(tril(ones(n_feature),-1)); % indices of the lower triangular part of M
dia_idx=(1:n_feature+1:n_feature^2)'; % indices of the diagonals of M
num_list=1:n_feature; % indices of graph nodes of M where the nodes are currently connected
league_vec = ones(n_feature,1); % color set of graph nodes of M
league_vec(2:2:end)=-1; % set it like this so that the odd nodes are blue and even nodes are red
bins=ones(1,n_feature); % the number of the unique numbers of bins represents the number of subgraphs of M
%==========================================================================

%=Initialize M=============================================================
M0=initial_M(n_feature,2); % initial M as a [1-dense matrix] or [2-sparse matrix]
rng(lobpcg_random_control);
[fv1,~] = ...
    lobpcg_fv(randn(n_feature,1),M0,1e-12,200); % compute the first eigenvector for the first time using LOBPCG
scaled_M = (1./fv1) .* M0 .* fv1'; % compute the similarity-transformed M
scaled_factors = (1./fv1) .* ones(n_feature) .* fv1'; % get the scalars of M
%==========================================================================

[LP_A_sparse_i,...
    LP_A_sparse_j,...
    LP_A_sparse_s,...
    LP_b,...
    LP_lb,...
    LP_ub] = LP_setting(n_feature,rho); % get parts of the linear constraints ready for running Matlab linprog

%=run SGML=================================================================
disp('starting SGML.');
[M]=SGML_main(M0,...
    data_feature,...
    data_label,...
    n_sample,...
    n_feature,...
    tol_NR,...
    tol_GD,...
    rho,...
    max_iter,...
    GS_or_NR,...
    tol_golden_search,...
    options,...
    FW_dia_offdia_tol,...
    FW_full_tol,...
    C,...
    nv_od,...
    nv_full,...
    zz,...
    dia_idx,...
    num_list,...
    league_vec,...
    bins,...
    fv1,...
    scaled_M,...
    scaled_factors,...
    LP_A_sparse_i,...
    LP_A_sparse_j,...
    LP_A_sparse_s,...
    LP_b,...
    LP_lb,...
    LP_ub); % run SGML
disp('done with SGML.');
%==========================================================================

[ L ] = graph_Laplacian_train_test( feature_train_test, M ); % full observation

if classifier_i==1 % 3-NN classifier
 knn_size = 3;
%========KNN classifier starts========
fl = class_train_test(initial_label_index);
fl(fl == -1) = 0;
x = KNN(fl, feature_train_test(initial_label_index,:), sqrtm(full(M)), knn_size, feature_train_test(~initial_label_index,:));
x(x==0) = -1;
x_valid = class_train_test;
x_valid(~initial_label_index) = x;
%=========KNN classifier ends=========   
elseif classifier_i==2 % Mahalanobis classifier
%=======Mahalanobis classifier starts========
z=mahalanobis_classifier(m,M,X);
x_valid=class_train_test;
x_valid(~initial_label_index)=z;
%========Mahalanobis classifier ends=========    
else % GLR-based classifier
%=======Graph classifier starts=======
cvx_begin
variable x(n_sample,1);
minimize(x'*L*x)
subject to
x(initial_label_index) == class_train_test(initial_label_index);
cvx_end
x_valid = sign(x);
%========Graph classifier ends========    
end

diff_label = x_valid - class_train_test;
error_classifier = size(find(diff_label~=0),1)*size(find(diff_label~=0),2)/size(class_test,1);
disp(['objective after metric learning : ' num2str(x_valid'*L*x_valid)]);
disp(['error rate after metric learning : ' num2str(error_classifier)]);
end