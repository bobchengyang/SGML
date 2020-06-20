function [obj_vec,time_vec] = ...
    binary_classification( ...
    feature, ...
    class, ...
    train, ...
    test, ...
    class_i, ...
    class_j,...
    optimizationer)

class(class~=class_i) = class_j; % turn ground truth labels to a binary one

train_index = train;
test_index = test;

feature_train = feature(train_index,:);

mean_TRAIN_set_0 = mean(feature_train);
std_TRAIN_set_0 = std(feature_train);

mean_TRAIN_set = repmat(mean_TRAIN_set_0,size(feature_train,1),1);
std_TRAIN_set = repmat(std_TRAIN_set_0,size(feature_train,1),1);

feature_train = (feature_train - mean_TRAIN_set)./std_TRAIN_set;

if length(find(isnan(feature_train)))>0
    error('features have NaN(s)');
end

feature_train_l2=sqrt(sum(feature_train.^2,2));
for i=1:size(feature_train,1)
    feature_train(i,:)=feature_train(i,:)/feature_train_l2(i);
end

feature_test = feature(test_index,:);

mean_TEST_set = repmat(mean_TRAIN_set_0,size(feature_test,1),1);
std_TEST_set = repmat(std_TRAIN_set_0,size(feature_test,1),1);

feature_test = (feature_test - mean_TEST_set)./std_TEST_set;

feature_test_l2=sqrt(sum(feature_test.^2,2));
for i=1:size(feature_test,1)
    feature_test(i,:)=feature_test(i,:)/feature_test_l2(i);
end

feature_REFORM = feature;

feature_REFORM(train_index,:) = feature_train;
feature_REFORM(test_index,:) = feature_test;
feature_REFORM(~(train_index|test_index),:) = [];

feature_train_test = feature_REFORM;

class_train_test = class(train_index|test_index);
class_train_test(class_train_test==class_i) = 1;
class_train_test(class_train_test==class_j) = -1;

initial_label = zeros(size(class,1),1);
initial_label(train_index&class==class_i) = 1;
initial_label(train_index&class==class_j) = -1;
initial_label(~train_index&~test_index) = [];
initial_label_index = initial_label ~= 0;

if optimizationer==1 % PD-cone
    [obj_vec,time_vec] = ...
        PDcone_obj_run_time( feature_train_test, ...
        initial_label_index, ...
        class_train_test);
elseif optimizationer==2 % HBNB
    [obj_vec,time_vec] = ...
        HBNB_obj_run_time( feature_train_test, ...
        initial_label_index, ...
        class_train_test);
else % SGML
    [obj_vec,time_vec] = ...
        SGML_obj_run_time( feature_train_test, ...
        initial_label_index, ...
        class_train_test);
end

end

