%=================================================================
% Signed Graph Metric Learing (SGML) via Gershgorin Disc Alignment
% **SGML with MCML objective value and running time experiments
%
% author: Cheng Yang
% email me any questions: cheng.yang@ieee.org
% date: June 19th, 2020
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

disp('1. PD-cone.');
disp('2. HBNB.');
disp('3. SGML.');
optimizationer = eval(input('please enter number 1-3 (# of the above optimization framework) to run: ', 's'));

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

feature = read_data(:,1:end-1); % data features
feature(isnan(feature))=0;
rng(0);
feature = feature + 1e-12*randn(size(feature)); % to avoid NaN during normalization

label = read_data(:,end); % data labels

K=round(length(label)/4); % test 4 samples at a time
run_n=K;
    
obj_temp=zeros(run_n,1);
time_temp=zeros(run_n,1);

rng(0); % for re-producibility
indices = crossvalind('Kfold',label,K); % K-fold cross-validation

obj_i=1;

profile on

for fold_i = 1:run_n
    disp(['fold ' num2str(fold_i) ' of ' num2str(K)]);
    train = (indices == fold_i); % these are indices for training data
    test=~train;
    
    % binary classification
    [obj_vec,time_vec] = ...
        binary_classification( ...
        feature, ...
        label, ...
        train, ...
        test, ...
        1, ...
        -1, ...
        optimizationer);
    obj_temp(obj_i)=obj_vec;
    time_temp(obj_i)=time_vec;
    obj_i=obj_i+1;
end

disp(['mean obj: ' num2str(mean(obj_temp)) ' mean time: ' num2str(mean(time_temp))]);

profile off
profile viewer
