function [obj_vec,time_vec] = ...
    HBNB_obj_run_time( feature_train_test, ...
    initial_label_index, ...
    class_train_test)

tol_main=1e-5;
tol_diagonal=1e-3;
tol_offdiagonal=1e-3;

partial_feature = feature_train_test(initial_label_index,:);
partial_observation = class_train_test(initial_label_index);
[partial_sample,n_feature]=size(partial_feature);
S_upper=n_feature;


run_t=1;
time_vec=zeros(run_t,1);
obj_vec=zeros(run_t,1);
lobpcg_random_control=0;

for time_i=1:run_t
    
    tStart=tic;
    M=initial_M(n_feature,1);
    dia_idx=find(M==diag(M));
    lower_bounds=M;
    lower_bounds(dia_idx)=0;
    lower_bounds=sum(abs(lower_bounds),2);
    
    [c,y] = get_graph_Laplacian_variables_ready(partial_feature,partial_observation,partial_sample,n_feature);
    lr=.1/partial_sample;%start with a relatively large learning rate
    
    counter_diag_nondiag = 0;
    
    tol_diag_nondiag = Inf;
    nc=0;
    flag=0;
    while tol_diag_nondiag > tol_main
        
        if counter_diag_nondiag == 0
            [L] = graph_Laplacian( partial_sample, c, M );
            initial_objective = partial_observation' * L * partial_observation;
            disp(['inintial = ' num2str(initial_objective)]);        
        end

        [ M, lr ] = HBNB_dia(...
            partial_sample,...
            c,...
            y,...
            lr,...
            lower_bounds,...
            n_feature,...
            M,...
            partial_observation,...
            S_upper,...
            tol_diagonal,...
            dia_idx);
        
        for BCD = 1:n_feature
            remaining_idx=1:n_feature;
            remaining_idx(BCD)=[];
            m11=M(BCD,BCD);
            M12=M(BCD,remaining_idx);
            M22=M(remaining_idx,remaining_idx);
            M21 = M12';
            
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
            
            [ M,lr,lower_bounds ] = HBNB_offdia(...
                partial_sample,...
                n_feature,...
                c,...
                y,...
                lr,...
                m11,...
                partial_observation,...
                M21,...
                M22_min_eig,...
                M,...
                BCD,...
                tol_offdiagonal,...
                remaining_idx,...
                lower_bounds,...
                dia_idx);
            
        end
        
        [ L ] = graph_Laplacian( partial_sample, c, M );
        
        counter_diag_nondiag = counter_diag_nondiag + 1;
        
        current_objective = partial_observation' * L * partial_observation;
        
        tol_diag_nondiag = norm(current_objective - initial_objective);
        
        initial_objective = current_objective;
        if counter_diag_nondiag>1e3% not converged
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
end