function [G_eric] = dml_step_size_gradient(A, D, d, nv, BCD, remaining_idx,s_k,zz,length_D)
%grad2 = fS1(X, S, A, N, d, nv, BCD, remaining_idx);
%grad1 = fD1(X, D, A, N, d, nv, BCD, remaining_idx);
%G_eric = grad_projection_combine(grad1, grad2, d, nv);
G_eric = fD1(D, A, d, nv, BCD, remaining_idx,length_D);
if nv==d+(d*(d-1)/2)
   G_eric=[2*G_eric(zz);diag(G_eric)]; 
end
if nv==d+d-1
   G_eric(1:d-1)=G_eric(1:d-1).*2; 
end
if nv==d-1
   G_eric=G_eric.*2; 
end
G_eric=G_eric.*s_k;
end

