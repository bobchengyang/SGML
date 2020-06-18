% optimization M on testing classification dataset
% Cheng 03-SEP-2019

clear;
clc;
close all;

addpath('D:\different_objectives\dataset\'); %dataset
addpath('competing_scheme\Wei\'); %Wei
addpath('competing_scheme\PD_cone\'); %PD cone
addpath('competing_scheme\GMML\'); %GMML
addpath('competing_scheme\LMNN\'); %LMNN
addpath('competing_scheme\MCML\'); %MCML

record_counter = 0;
acc_record=zeros(10,1);
for rngi = 0:9
    
    disp(['=====current random seed===== ' num2str(rngi)]);
    
    rng(rngi); % for re-producibility
    
    mat_name = ['iris_dataset_' num2str(rngi) '_spectral_clustering_try.mat']; % save results
    
    noc = 3; % number of classes
    
    read_data = importdata('iris.dat'); % read data
    
    %     rng(0);
    % %
    %     read_data_idx = randperm(size(read_data,1));
    % %
    %     feature = read_data(read_data_idx(1:100),1:end-1); % data features
    %     label = read_data(read_data_idx(1:100),end); % data labels
    
    feature = read_data(:,1:end-1); % data features
    
    label = read_data(:,end); % data labels
    
    oao_combo = noc*(noc-1)/2; % number of classifiers
    
    error_count_total = 0; % count the mis-classified samples
    
    confusion_matrix = zeros(noc); % set confusion matrix
    
    K = 2; % K-fold cross-validation
    
    rng(rngi); % for re-producibility
    
    indices = crossvalind('Kfold',label,K); % K-fold cross-validation
    
    %% ===PARAMETER SETTINGS===
    
    S_upper = size(feature,2); % C constant
    
    rho = 1e-6; % constant for PD property of M
    
    epsilon = 5e-1; % constant for neighouring diagonal of M
    
    proportion_factor = 1e-5; % constant for maximum difference of diagonal entries of M
    
    tol_main = 1e-5; % tol for the main loop
    
    tol_diagonal = 1e-5; % tol for the diagonal optimization
    
    tol_offdiagonal = 1e-5; % tol for the off-diagonal optimization
    
    %% ========================
    
    %% ===PARAMETER SETTINGS for preprocessing===
    
    proportion_threshold = 1e-1; % minimum proportion of the min feature weight / max feature weight
    
    tol_set_prepro = 1e-0; % tol for the main loop of the preprocessing step
    
    %% ===========================================
    
    %% ===LIPSCHITZ step scale===
    
    step_scale = 1; % step scaling factor for lipschitz step size diagonal
    
    step_scale_od = 2; % step scaling factor for lipschitz step size off-diagonal
    
    %% ==========================
    
    class_ratio = [];
    
    for class_ratio_i = 1:noc
        
        class_ratio = [ class_ratio 1];
        
    end
    
    test_count_total = 0;
    
    record_counter=record_counter+1;
    
    fold_i=1;
    
    %test = ((indices==1)|(indices==2)); % these are indices for training data
    test = (indices==1);
    
    train = ~test; % the remaining indices are for testing data
    
    % set voting matrices and error record
    [ class_result, ...
        error_matrix, ...
        weights_result, ...
        GT_obj, ...
        final_obj, ...
        error_result] = ...
        optimization_M_oao_vae( label, ...
        oao_combo, ...
        K );
    
    % classification
    [ class_result, ...
        error_matrix, ...
        weights_result, ...
        GT_obj, ...
        final_obj, ...
        error_result] = ...
        optimization_M_one_against_one_BCD( noc, ...
        fold_i, ...
        oao_combo, ...
        feature, ...
        label, ...
        train, ...
        test, ...
        class_result, ...
        error_matrix, ...
        weights_result, ...
        GT_obj, ...
        final_obj, ...
        error_result,...
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
        step_scale_od);
    
    % error
    [test_count, ...
        error_count_fold_i, ...
        confusion_matrix] = ...
        optimization_M_one_against_one_voting_1_and_2(noc, ...
        class_ratio, ...
        oao_combo, ...
        fold_i, ...
        label, ...
        test,...
        class_result, ...
        confusion_matrix);
    
    test_count_total = test_count_total + test_count;
    
    [error_count_total ] = ...
        optimization_M_oao_oaa_error(error_count_total,...
        error_count_fold_i);
    
    acc_record(record_counter)=1-error_matrix(record_counter);
    
    % K-fold cross-validation error rate
    
    [error_rate] = optimization_M_net_error(test_count_total, ...
        error_count_total);
    
    save(mat_name); % save results
    
end

