function [ Wf ] = optimization_M_set_Wf_Mahalanobis( feature )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

N = size(feature,1);

Wf = cell(N);

for i = 1:N
    for j = 1:N
        
        f_i = feature(i,:);
        
        f_j = feature(j,:);
        
        f_i_j = f_i - f_j;
        
        Wf{i,j} =  f_i_j';
        
    end
end

end

