function [ class_result, ...
           error_matrix, ...
           weights_result, ...
           GT_obj, ...
           final_obj, ...
           error_result] = ...
           optimization_M_oao_vae( class_gt, ...
                                oao_oaa_combo, ...
                                K )
%one_against_one_vae Summary of this function goes here
%   Detailed explanation goes here
    
%% create class matrix for voting

  subject_number = size(class_gt,1);

  class_result = zeros(subject_number,oao_oaa_combo,K);
  
%% create error matrix for record

  error_matrix = zeros(K,oao_oaa_combo);

%% create weights | GT_obj | final_obj | error cells for record

  weights_result = cell(K,oao_oaa_combo);
  GT_obj = cell(K,oao_oaa_combo);
  final_obj = cell(K,oao_oaa_combo);
  error_result = cell(K,oao_oaa_combo);

end

