function [voted_labels] = optimization_M_one_against_one_voted_label(voting_matrix,class_ratio)
%ONE_AGAINST_ONE_VOTED_LABEL Summary of this function goes here
%   Detailed explanation goes here
[voted_count, voted_label_i] = max(voting_matrix,[],2);
for i = 1:size(voting_matrix,1)
    voting_index = find(voting_matrix(i,:)==voted_count(i));
    if length(voting_index) > 1
       voting_matrix(i,voting_index) = voting_matrix(i,voting_index).*class_ratio(voting_index); 
    end
end
[voted_count, voted_labels] = max(voting_matrix,[],2);
end

