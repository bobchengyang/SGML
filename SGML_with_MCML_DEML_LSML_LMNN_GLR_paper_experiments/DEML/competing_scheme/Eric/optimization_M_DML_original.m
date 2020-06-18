function [ class_temp_binary, class_temp, ...
    GT_obj_all, obj_all, error_iter] = ...
    optimization_M_DML_original( class_test, ...
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

GT_obj_all = 0;
obj_all = 0;
error_iter = 0;

%% check error rate before metric learning starts

[n_sample, n_feature]= size(feature_train_test); %get the number of samples and the number of features

M = zeros(n_feature);

M(logical(eye(n_feature))) = S_upper/n_feature;

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
partial_sample=length(partial_observation);

run_t=3;
time_vec=zeros(run_t,1);
obj_vec=zeros(run_t,1);

for time_i=1:run_t
    
    
    A=optimization_M_initialization(n_feature,1);
    [w,S,D] = get_S_and_D(A,partial_feature, partial_observation);
    t=n_feature;
    
    initial_objective=dml_obj(A, partial_feature, D);
    
    disp(['initial = ' num2str(initial_objective)]);
    tic;
    
    %% eric code starts
    % function [A, converged] = dml_main(X, S, D, A, w, t, maxiter)
    % ---------------------------------------------------------------------------
    % Input
    % X: data
    % S: similarity constraints (in the form of a pairwise-similarity matrix)
    % D: disimilarity constraints (in the form of a pairwise-disimilarity matrix)
    % A: initial distance metric matrix
    % w: a weight vector originated from similar data (see paper)
    % t: upper bound of constraint C1 (the sum of pairwise distance bound)
    % maxiter: maximum iterations
    %
    % Output
    % A: the solution of distance metric matrix
    % converged: indicator of convergence
    % iters: iterations passed until convergence
    % ---------------------------------------------------------------------------
    
    X = partial_feature;
    N = partial_sample;     % number of examples
    d = n_feature;     % dimensionality of examples
    nv=d+(d*(d-1)/2);
    
    threshold2 = 1e-5;% error-bound of main A-update iteration
    epsilon = 1e-2;   % error-bound of iterative projection on C1 and C2
    maxcount = Inf;
    maxiter = Inf;
    delta=Inf;

    w1 = w/norm(w);    % make 'w' a unit vector
    t1 = t/norm(w);    % distance from origin to w^T*x=t plane
    
    count=1;
    alpha = 0.1/partial_sample;         % initial step size along gradient
    
    grad1 = fS1(X, S, A, N, d, nv, 0, 0);   % gradient of similarity constraint function
    grad2 = fD1(X, D, A, N, d, nv, 0, 0);   % gradient of dissimilarity constraint func.
    Eric_grad = grad_projection(grad1, grad2, d); % gradient of fD1 orthognal to fS1
    
    A_last = A;        % initial A
    done = 0;
    
    while (~done)
        
        % projection of constrants C1 and C2 ______________________________
        % _________________________________________________________________
        %A_update_cycle=count;
        projection_iters = 0;
        satisfy=0;
        
        while projection_iters < maxiter && ~satisfy
            
            A0 = A;
            % _____________________________________________________________
            % first constraint:
            % f(A) = \sum_{i,j \in S} d_ij' A d_ij <= t              (1)
            % (1) can be rewritten as a linear constraint: w^T x = t,
            % where x is the unrolled matrix of A,
            % w is also an unroled matrix of W where
            % W_{kl}= \sum_{i,j \in S}d_ij^k * d_ij^l
            
            %x0= unroll(A0);
            x0=A0(:);
            if w' * x0 <= t
                A = A0;
            else
                x = x0 + (t1-w1'*x0)*w1;
                %A = packcolume(x, d, d);
                A=reshape(x,[d d]);
            end
            
            % __________________________________________________________________
            % second constraint:
            % PSD constraint A>=0
            % project A onto domain A>0
            
            A = (A + A')/2;  % enforce A to be symmetric
            [V,L] = eig(A);  % V is an othornomal matrix of A's eigenvectors,
            % L is the diagnal matrix of A's eigenvalues,
            L = max(L, 0);
            A = V*L*V';
%             if sum(diag(A))>S_upper
%                 ft=sum(diag(A))/S_upper;
%                 A=A/ft;
%             end
            fDC2 = w'*A(:);
            
            % __________________________________________________________________
            
            error2 = (fDC2-t)/t;
            projection_iters = projection_iters + 1;
            if projection_iters>1e4 % not converged
                satisfy=0;
                break
            end
            if error2 > epsilon
                satisfy=0;
            else
                satisfy=1;   % loop until constrait is not violated after both projections
            end
            
        end  % end projection on C1 and C2
        
        % __________________________________________________________________
        % third constraint: Gradient ascent
        % max: g(A)>=1
        % here we suppose g(A) = fD(A) = \sum_{I,J \in D} sqrt(d_ij' A d_ij)
        
        obj_previous = fD(A_last, X, D);           % g(A_old)
        obj = fD(A, X, D);                         % g(A): current A
        
        if  (obj > obj_previous || count == 1) && (satisfy ==1)
            
            % if projection of 1 and 2 is successful,
            % and such projection imprives objective function,
            % slightly increase learning rate, and updata from the current A
            
            alpha =  alpha * 1.01;  A_last = A; %obj;
            grad2 = fS1(X, S, A, N, d, nv, 0, 0);
            grad1 = fD1(X, D, A, N, d, nv, 0, 0);
            Eric_grad = grad_projection(grad1, grad2, d);
            A = A + alpha*Eric_grad;
            
        else
            % if projection of 1 and 2 failed,
            % or obj <= obj_previous due to projection of 1 and 2,
            % shrink learning rate, and re-updata from the previous A
            
            alpha = alpha/2;
            A = A_last + alpha*Eric_grad;
            
        end
        
%                 delta = norm(alpha*Eric_grad, 'fro')/norm(A_last, 'fro');
        if count>1
           delta = norm(obj-obj_previous);
            disp(['delta: ' num2str(delta) ' obj: ' num2str(obj)]);
        end
        count = count + 1;
        if count == maxcount || delta <threshold2
            done = 1;
        end
        
    end
    
    if delta > threshold2
        converged=0;
    else
        converged=1;
    end
    
    A = A_last;
    %% eric code ends
    
    M=A;
    
    current_objective=dml_obj(M, partial_feature, D);
    
    if converged==1
        disp(['converged = ' num2str(current_objective)]);
    else
        disp(['NOT converged = ' num2str(current_objective)]);
    end
    
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

