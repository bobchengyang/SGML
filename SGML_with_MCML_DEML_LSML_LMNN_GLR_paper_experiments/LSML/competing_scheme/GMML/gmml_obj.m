function [f] = gmml_obj(A, S, D)
f=trace(A*S)+trace(A\D);
end

