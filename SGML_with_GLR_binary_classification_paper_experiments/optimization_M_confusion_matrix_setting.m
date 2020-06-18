function [ confusion_matrix ] = optimization_M_confusion_matrix_setting( confusion_matrix, label_predicted, label_GT )
%CONFUSION_MATRIX_SETTING Summary of this function goes here
%   Detailed explanation goes here

%% Sample i that is predicted as Class X is in fact Class Y
for i = 1:size(label_predicted,1)
    if label_predicted(i) ~= 0
       confusion_matrix(label_GT(i),label_predicted(i)) = confusion_matrix(label_GT(i),label_predicted(i)) + 1;
    end
end

end

