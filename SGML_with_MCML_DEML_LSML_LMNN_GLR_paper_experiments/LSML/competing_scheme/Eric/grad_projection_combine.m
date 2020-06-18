function gradProj = grad_projection_combine(grad1, grad2, d, nv)

% the compoment of g1 that is perpendicular to g2
%g1 = unroll(grad1);
%g2 = unroll(grad2);
g1=grad1(:);
g2=grad2(:);
g2 = g2/norm(g2, 2);
gtemp = g1 - (g2'*g1)*g2;
gtemp = gtemp/norm(gtemp, 2);   % normalize
%gradProj = packcolume(gtemp, d, d);
if nv==d+(d*(d-1)/2)
    gradProj=reshape(gtemp,[d d]);
else
    gradProj=gtemp;
end
