function [dml] = dml_obj(A, D)
%DML_OBJ Summary of this function goes here
%   Detailed explanation goes here
dml=sum(sum(D*A.*D,2).^(1/2));
%dml=fD(A, D);
end

