function [G_eric1,G_eric2] = dml_step_size_NR(A, D, d, nv, BCD, remaining_idx,s_k,zz,length_D)
%grad2 = fS1_second(X, S, A, N, d, nv, BCD, remaining_idx);
%grad1 = fD1_second(X, D, A, N, d, nv, BCD, remaining_idx);
%G_eric = grad_projection_combine(grad1, grad2, d, nv);
[G_eric1,G_eric2] = fD1_second(D, A, d, nv, BCD, remaining_idx,length_D);
if nv==d+(d*(d-1)/2)
   G_eric1=[2*G_eric1(zz);diag(G_eric1)]; 
   G_eric2=[2*G_eric2(zz);diag(G_eric2)]; 
end
if nv==d+d-1
   G_eric1(1:d-1)=G_eric1(1:d-1).*2; 
   G_eric2(1:d-1)=G_eric2(1:d-1).*2; 
end
if nv==d-1
   G_eric1=G_eric1.*2; 
   G_eric2=G_eric2.*2; 
end
G_eric1=G_eric1.*s_k;
G_eric2=G_eric2.*s_k.*s_k;
end

