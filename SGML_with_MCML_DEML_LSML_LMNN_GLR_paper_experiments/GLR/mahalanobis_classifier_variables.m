function [m,X] = ...
    mahalanobis_classifier_variables(...
    feature_train_test,...
    class_train_test,...
    initial_label_index)

% feature_train_class_1=feature_train_test(class_train_test==1,:);
% feature_train_class_2=feature_train_test(class_train_test==-1,:);

feature_train=feature_train_test(initial_label_index,:);
class_train=class_train_test(initial_label_index);
feature_train_class_1=feature_train(class_train==1,:);
feature_train_class_2=feature_train(class_train==-1,:);

m=[mean(feature_train_class_1);mean(feature_train_class_2)];
X=feature_train_test(~initial_label_index,:);

m=m';
X=X';

end

