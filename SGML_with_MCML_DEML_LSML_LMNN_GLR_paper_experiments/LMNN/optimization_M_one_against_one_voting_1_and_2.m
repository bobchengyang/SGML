function [test_count, ...
 error_count_linear_fold_i, ...
 confusion_matrix_linear] = ...
 optimization_M_one_against_one_voting_1_and_2(noc, ...
 class_ratio, ...
 oao_combo, ...
 fold_i, ...
 class, ...
 test,...
 class_linear, ...
 confusion_matrix_linear)
%ONE_AGAINST_ONE_VOTING Summary of this function goes here
%   Detailed explanation goes here

class_combo = zeros(noc*(noc-1)/2,2);
temp_counter = 0;
for c_i = 1:noc-1
    for c_j = 2:noc
        if c_i < c_j
           temp_counter = temp_counter + 1;
           class_combo(temp_counter,1) = c_i;
           class_combo(temp_counter,2) = c_j;
        end
    end
end
    
   class_test = class(test);
   test_count = size(class_test,1);
   
   voting_linear_fold_i = zeros(size(class_linear,1),noc);

for order_number = 1:oao_combo
    for subject_number = 1:size(class_linear,1)
        for current_class_number_seq = 1:oao_combo
        if class_linear(subject_number,order_number,fold_i) == current_class_number_seq
           if current_class_number_seq <= noc
           voting_linear_fold_i(subject_number,current_class_number_seq) = voting_linear_fold_i(subject_number,current_class_number_seq) + 1;
           else
           to_be_added_c_i = class_combo(current_class_number_seq-noc,1);
           to_be_added_c_j = class_combo(current_class_number_seq-noc,2);
           voting_linear_fold_i(subject_number,to_be_added_c_i) = voting_linear_fold_i(subject_number,to_be_added_c_i) + 1;
           voting_linear_fold_i(subject_number,to_be_added_c_j) = voting_linear_fold_i(subject_number,to_be_added_c_j) + 1;
           end
        end
        end
    end
end
[voted_label_linear_fold_i] = optimization_M_one_against_one_voted_label(voting_linear_fold_i(test,:),class_ratio);
diff_label = voted_label_linear_fold_i - class_test;
error_count_linear_fold_i = size(find(diff_label~=0),1)*size(find(diff_label~=0),2);

confusion_matrix_linear = optimization_M_confusion_matrix_setting( confusion_matrix_linear, voted_label_linear_fold_i, class_test );

end

