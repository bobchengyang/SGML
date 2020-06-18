%=================================================================
% Signed Graph Metric Learing (SGML) via Gershgorin Disc Alignment
% **SGML with GLR binary classification
%
% author: Cheng Yang
% email me any questions: cheng.yang@ieee.org
% date: June 18th, 2020
% please kindly cite the paper:
% ['Signed Graph Metric Learning via Gershgorin Disc Alignment',
% Cheng Yang, Gene Cheung, Wei Hu,
% https://128.84.21.199/abs/2006.08816]
%=================================================================

clear;
clc;
close all;

addpath('F:\18-JUN-2020 SGML\Signed_Graph_Metric_Learning\datasets\'); %dataset

disp('1. Australian; 14 features.');
disp('2. Breast-cancer; 10 features.');
disp('3. Diabetes; 8 features.');
disp('4. Fourclass; 2 features.');
disp('5. German; 24 features.');
disp('6. Haberman; 3 features.');
disp('7. Heart; 13 features.');
disp('8. ILPD; 10 features.');
disp('9. Liver-disorders; 5 features.');
disp('10. Monk1; 6 features.');
disp('11. Pima; 8 features.');
disp('12. Planning; 12 features.');
disp('13. Voting; 16 features.');
disp('14. WDBC; 30 features.');
disp('15. Sonar; 60 features.');
disp('16. Madelon; 500 features.');
disp('17. Colon-cancer; 2000 features.');
dataset_i = eval(input('please enter number 1-17 (# of the above datasets) to run: ', 's'));

noc=2; % number of classes

if dataset_i==1
    read_data = importdata('australian.csv');
    
elseif dataset_i==2
    read_data = importdata('breast-cancer.csv'); 
elseif dataset_i==3
    read_data = importdata('diabetes.csv');
elseif dataset_i==4
    read_data = importdata('fourclass.csv');
elseif dataset_i==5
    read_data = importdata('german.csv');
elseif dataset_i==6
    read_data = importdata('haberman.csv');
elseif dataset_i==7
    read_data = importdata('heart.dat');
elseif dataset_i==8
    read_data = importdata('Indian Liver Patient Dataset (ILPD).csv');
elseif dataset_i==9
    read_data = importdata('liver-disorders.csv');
elseif dataset_i==10
    read_data = importdata('monk1.csv');
elseif dataset_i==11
    read_data = importdata('pima.csv');
elseif dataset_i==12
    read_data = importdata('planning.csv');
elseif dataset_i==13
    read_data = importdata('voting.csv');
elseif dataset_i==14
    read_data = importdata('WDBC.csv');
end

disp('1. 3-NN classifier.');
disp('2. Mahalanobis classifier.');
disp('3. GLR-based classifier.');
classifier_i = eval(input('please kindly choose 1 out of the above 3 classifiers to run:', 's'));

obj_i=0;
number_of_runs=10;
obj_temp=zeros(number_of_runs,1);
time_temp=zeros(number_of_runs,1);
accuracy_temp=zeros(number_of_runs,1);

for rngi = 0:9
    obj_i=obj_i+1;
    disp(['=====current random seed===== ' num2str(rngi)]);

    feature = read_data(:,1:end-1); % data features
    feature(isnan(feature))=0;
    
    label = read_data(:,end); % data labels
    
    K=5; % for classification 60% training 40% test
    
    rng(rngi); % for re-producibility
    
    indices = crossvalind('Kfold',label,K); % K-fold cross-validation
    
    %% ===PARAMETER SETTINGS===
    
    S_upper = size(feature,2); % C constant
    
    rho = 1e-5; % constant for PD property of M
           
    %% ========================
        
    %% ==========================
    
    for fold_i = 1:1
        for fold_j = 2:2
            if fold_i<fold_j
                disp('==========================================================================');
                disp(['classifier ' num2str(obj_i) '; folds ' num2str(fold_i) ' and ' num2str(fold_j)]);
                test = (indices == fold_i | indices == fold_j); % these are indices for test data
                
                train = ~test; % the remaining indices are for training data
       
                % binary classification
                [ obj_vec,time_vec,error_classifier] = ...
                    optimization_M_one_against_one_binary_R_Block_CD( ...
                    feature, ...
                    label, ...
                    train, ...
                    test, ...
                    1, ...
                    -1, ...
                    S_upper,...
                    rho,...
                    classifier_i);
                
                obj_temp(obj_i)=obj_vec;
                time_temp(obj_i)=time_vec;
                accuracy_temp(obj_i)=1-error_classifier;
                disp(['classifier ' num2str(obj_i) ' accuracy: ' num2str(accuracy_temp(obj_i)*100)]);
                
            end
        end
    end
    
    
end
disp(['acc: ' num2str(mean(accuracy_temp)*100,'%.2f') char(177) num2str(std(accuracy_temp)*100,'%.2f') ...
    ' mean obj: ' num2str(mean(obj_temp)) ' mean time: ' num2str(mean(time_temp))]);
