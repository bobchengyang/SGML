% optimization M on testing classification dataset
% Cheng 03-SEP-2019

clear;
clc;
close all;

profile on

addpath('D:\different_objectives_old_source\dataset\'); %dataset
% addpath('competing_scheme\GMML\'); %GMML
% addpath('competing_scheme\LMNN\'); %LMNN
% addpath('competing_scheme\MCML\'); %MCML
% addpath('competing_scheme\LSML\'); %LSML
% addpath('competing_scheme\Eric\'); %DML
addpath('competing_scheme\Wei\'); %wei
addpath('competing_scheme\PD_cone\'); %PDcone

addpath('D:\different_objectives_old_source\lp_solver_ws\'); %lpsolver with warm start

for rngi = 0:0
    
    disp(['=====current random seed===== ' num2str(rngi)]);
    
    rng(rngi); % for re-producibility
    
    mat_name = ['iris_dataset_' num2str(rngi) '_spectral_clustering_try.mat']; % save results
    
%     noc = 2; % number of classes
   
%% PAMI2019 dataset starts
noc=2;read_data = importdata('australian.csv');
% noc=2;read_data = importdata('breast-cancer.csv');
% noc=2;read_data = importdata('diabetes.csv');
% noc=2;read_data = importdata('fourclass.csv');
% noc=2;read_data = importdata('german.csv');
% noc=2;read_data = importdata('haberman.csv');
% noc=2;read_data = importdata('heart.dat');
% noc=2;read_data = importdata('Indian Liver Patient Dataset (ILPD).csv');
% noc=2;read_data = importdata('liver-disorders.csv');
% noc=2;read_data = importdata('monk1.csv');
% noc=2;read_data = importdata('diabetes_scale.csv'); % pima
% noc=2;read_data = importdata('planning.csv');
% noc=2;read_data = importdata('house-votes-84n.csv'); % voting
% noc=2;read_data = importdata('house-votes-84y.csv'); % voting
% noc=2;read_data = importdata('house-votes-840.csv'); % voting
% noc=2;read_data = importdata('WDBC.csv');
%% PAMI2019 dataset ends

%% additional large-feature datasets starts
% noc=2;read_data = importdata('sonar.csv');
% noc=2;read_data = importdata('madelon.csv'); %1:200
% noc=2;read_data = importdata('colon-cancer.csv');
% noc=2;read_data = importdata('leukemia.csv');
%% additional large-feature datasets ends

%     rng(0);
%     read_data_idx = randperm(size(read_data,1)); 
%     feature = read_data(read_data_idx(1:200),1:end-1); % data features
%     label = read_data(read_data_idx(1:200),end); % data labels

feature = read_data(:,1:end-1); % data features
feature(isnan(feature))=0;
rng(0);
feature = feature + 1e-12*randn(size(feature)); % to avoid NaN during normalization
label = read_data(:,end); % data labels

K=round(length(label)/4); % test 4 samples at a time
run_n=K;
%% old dataset starts
%     noc=2;K=50;read_data = importdata('planning.csv');
%     noc=2;K=50;read_data = importdata('breast-cancer.csv'); 
%     noc=2;K=30;read_data = importdata('heart.dat');
%     noc=3;K=30;read_data = importdata('iris.dat');
%     noc=3;K=50;read_data = importdata('seeds.mat');
%     noc=3;K=50;read_data = importdata('wine.dat');
%     noc=2;K=50;read_data = importdata('sonar.csv');
%     noc=2;K=50;read_data = importdata('madelon.csv'); %1:200
%     noc=2;K=10;read_data = importdata('colon-cancer.csv');
%     noc=2;K=21;read_data = importdata('leukemia.csv');
%% old dataset ends
    
    oao_combo = 1; % number of classifiers
    
    error_count_total = 0; % count the mis-classified samples
    
    confusion_matrix = zeros(noc); % set confusion matrix
    
    rng(rngi); % for re-producibility
    
    indices = crossvalind('Kfold',label,K); % K-fold cross-validation
    
    %% ===PARAMETER SETTINGS===
    
    S_upper = size(feature,2); % C constant
    
    rho = 1e-5; % constant for PD property of M
    
    epsilon = 5e-1; % constant for neighouring diagonal of M
    
    proportion_factor = 1e-5; % constant for maximum difference of diagonal entries of M
    
    tol_main = 1e-5; % tol for the main loop
    
    tol_diagonal = 1e-3; % tol for the diagonal optimization
     
    tol_offdiagonal = 1e-3; % tol for the off-diagonal optimization
    
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
        
        class_ratio = [ class_ratio length(find(label==class_ratio_i))];
        
    end
    
    test_count_total = 0;
    
    obj_temp=zeros(10,1);
    time_temp=zeros(10,1);
    obj_i=1;
    for fold_i = 1:run_n
        disp(['fold ' num2str(fold_i) ' of ' num2str(K)]);
        train = (indices == fold_i); % these are indices for training data
        
        test = ~train; % the remaining indices are for testing data
        
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
            error_result,obj_vec,time_vec] = ...
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
        obj_temp(obj_i)=obj_vec;
        time_temp(obj_i)=time_vec;
        obj_i=obj_i+1;
    end
    %obj_temp
    disp(['mean obj: ' num2str(mean(obj_temp)) ' mean time: ' num2str(mean(time_temp))]);
    % K-fold cross-validation error rate
    
    [error_rate] = optimization_M_net_error(test_count_total, ...
        error_count_total);
    
    save(mat_name); % save results
    
end

profile off
profile viewer
