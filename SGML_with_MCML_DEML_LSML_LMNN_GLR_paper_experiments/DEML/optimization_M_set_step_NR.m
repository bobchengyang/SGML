function [ G1, G2 ] = optimization_M_set_step_NR( S, D, X, d, M, nv, zz, BCD, remaining_idx, s_k )
if nv==d+(d*(d-1)/2) % full
    G1=zeros(d);
    G2=zeros(d);
elseif nv==d+d-1 % diagonals + one row/column of off-diagonals
    G1=zeros(d+d-1,1);
    G2=zeros(d+d-1,1);
elseif nv==d % diagonals
    G1=zeros(d,1);
    G2=zeros(d,1);
else % one row/column of off-diagonals
    G1=zeros(d-1,1);
    G2=zeros(d-1,1);
end        

for i=1:size(S,1)
    for j=1:size(D,1)
        if nv==d+(d*(d-1)/2) % full
            d_ab=X(S(i,1),:)-X(S(i,2),:);
            d_cd=X(D(j,1),:)-X(D(j,2),:);
            ds=sqrt(d_ab*M*d_ab');
            dd=sqrt(d_cd*M*d_cd');
            if ds>dd
                dabab=d_ab'*d_ab;
                dcdcd=d_cd'*d_cd;
                G1=G1+...
                    dabab+...
                    dcdcd-...
                    (dabab*dd+dcdcd*ds)/(ds*dd);
                G2=G2+...
                    (1/2)*(dabab.*dabab*dd)/(ds^3)-...
                    (1/2)*(dabab.*dcdcd)/(ds*dd)+...
                    (1/2)*(dcdcd.*dcdcd*ds)/(dd^3)-...
                    (1/2)*(dcdcd.*dabab)/(dd*ds);
            end
        elseif nv==d+d-1 % diagonals + one row/column of off-diagonals
            dabab=[2*(X(S(i,1),BCD)-X(S(i,2),BCD))'.*(X(S(i,1),remaining_idx)-X(S(i,2),remaining_idx))';(X(S(i,1),:)-X(S(i,2),:))'.^2];
            dcdcd=[2*(X(D(j,1),BCD)-X(D(j,2),BCD))'.*(X(D(j,1),remaining_idx)-X(D(j,2),remaining_idx))';(X(D(j,1),:)-X(D(j,2),:))'.^2];            
            d_ab=X(S(i,1),:)-X(S(i,2),:);
            d_cd=X(D(j,1),:)-X(D(j,2),:);
            ds=sqrt(d_ab*M*d_ab');
            dd=sqrt(d_cd*M*d_cd');
            if ds>dd
                G1=G1+...
                    dabab+...
                    dcdcd-...
                    (dabab*dd+dcdcd*ds)/(ds*dd);
                G2=G2+...
                    (1/2)*(dabab.*dabab*dd)/(ds^3)-...
                    (1/2)*(dabab.*dcdcd)/(ds*dd)+...
                    (1/2)*(dcdcd.*dcdcd*ds)/(dd^3)-...
                    (1/2)*(dcdcd.*dabab)/(dd*ds);
            end          
        elseif nv==d % diagonals
            dabab=(X(S(i,1),:)-X(S(i,2),:))'.^2;
            dcdcd=(X(D(j,1),:)-X(D(j,2),:))'.^2;            
            d_ab=X(S(i,1),:)-X(S(i,2),:);
            d_cd=X(D(j,1),:)-X(D(j,2),:);
            ds=sqrt(d_ab*M*d_ab');
            dd=sqrt(d_cd*M*d_cd');
            if ds>dd
                G1=G1+...
                    dabab+...
                    dcdcd-...
                    (dabab*dd+dcdcd*ds)/(ds*dd);
                G2=G2+...
                    (1/2)*(dabab.*dabab*dd)/(ds^3)-...
                    (1/2)*(dabab.*dcdcd)/(ds*dd)+...
                    (1/2)*(dcdcd.*dcdcd*ds)/(dd^3)-...
                    (1/2)*(dcdcd.*dabab)/(dd*ds);
            end            
        else % one row/column of off-diagonals
            dabab=2*(X(S(i,1),BCD)-X(S(i,2),BCD))'.*(X(S(i,1),remaining_idx)-X(S(i,2),remaining_idx))';
            dcdcd=2*(X(D(j,1),BCD)-X(D(j,2),BCD))'.*(X(D(j,1),remaining_idx)-X(D(j,2),remaining_idx))';            
            d_ab=X(S(i,1),:)-X(S(i,2),:);
            d_cd=X(D(j,1),:)-X(D(j,2),:);
            ds=sqrt(d_ab*M*d_ab');
            dd=sqrt(d_cd*M*d_cd');
            if ds>dd
                G1=G1+...
                    dabab+...
                    dcdcd-...
                    (dabab*dd+dcdcd*ds)/(ds*dd);
                G2=G2+...
                    (1/2)*((1/2)*dabab.*dabab*dd)/(ds^3)-...
                    (1/2)*((1/2)*dabab.*dcdcd)/(ds*dd)+...
                    (1/2)*((1/2)*dcdcd.*dcdcd*ds)/(dd^3)-...
                    (1/2)*((1/2)*dcdcd.*dabab)/(dd*ds);
            end           
        end
    end
end

if nv==d+(d*(d-1)/2) % full
    G1=[2*G1(zz);diag(G1)];
    G2=[2*G2(zz);diag(G2)];
end

if nv==d+d-1 % diagonals + one row/column of off-diagonals
    G2(1:d-1)=G2(1:d-1)/2;
end

G1=G1.*s_k; % first derivative wrt gamma
G2=G2.*s_k.*s_k; % second derivative wrt gamma

end

