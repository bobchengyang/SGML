function [ G ] = optimization_M_set_step_gradient( S, D, X, d, M, nv, zz, BCD, remaining_idx, s_k )

if nv==d+(d*(d-1)/2) % full
    G=zeros(d);
elseif nv==d+d-1 % diagonals + one row/column of off-diagonals
    G=zeros(d+d-1,1);
elseif nv==d % diagonals
    G=zeros(d,1);
else % one row/column of off-diagonals
    G=zeros(d-1,1);
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
                G=G+...
                    dabab+...
                    dcdcd-...
                    (dabab*dd+dcdcd*ds)/(ds*dd);
            end
        elseif nv==d+d-1 % diagonals + one row/column of off-diagonals
            dabab=[2*(X(S(i,1),BCD)-X(S(i,2),BCD))'.*(X(S(i,1),remaining_idx)-X(S(i,2),remaining_idx))';(X(S(i,1),:)-X(S(i,2),:))'.^2];
            dcdcd=[2*(X(D(j,1),BCD)-X(D(j,2),BCD))'.*(X(D(j,1),remaining_idx)-X(D(j,2),remaining_idx))';(X(D(j,1),:)-X(D(j,2),:))'.^2];            
            d_ab=X(S(i,1),:)-X(S(i,2),:);
            d_cd=X(D(j,1),:)-X(D(j,2),:);
            ds=sqrt(d_ab*M*d_ab');
            dd=sqrt(d_cd*M*d_cd');
            if ds>dd
                G=G+...
                    dabab+...
                    dcdcd-...
                    (dabab*dd+dcdcd*ds)/(ds*dd);
            end          
        elseif nv==d % diagonals
            dabab=(X(S(i,1),:)-X(S(i,2),:))'.^2;
            dcdcd=(X(D(j,1),:)-X(D(j,2),:))'.^2;            
            d_ab=X(S(i,1),:)-X(S(i,2),:);
            d_cd=X(D(j,1),:)-X(D(j,2),:);
            ds=sqrt(d_ab*M*d_ab');
            dd=sqrt(d_cd*M*d_cd');
            if ds>dd
                G=G+...
                    dabab+...
                    dcdcd-...
                    (dabab*dd+dcdcd*ds)/(ds*dd);
            end            
        else % one row/column of off-diagonals
            dabab=2*(X(S(i,1),BCD)-X(S(i,2),BCD))'.*(X(S(i,1),remaining_idx)-X(S(i,2),remaining_idx))';
            dcdcd=2*(X(D(j,1),BCD)-X(D(j,2),BCD))'.*(X(D(j,1),remaining_idx)-X(D(j,2),remaining_idx))';            
            d_ab=X(S(i,1),:)-X(S(i,2),:);
            d_cd=X(D(j,1),:)-X(D(j,2),:);
            ds=sqrt(d_ab*M*d_ab');
            dd=sqrt(d_cd*M*d_cd');
            if ds>dd
                G=G+...
                    dabab+...
                    dcdcd-...
                    (dabab*dd+dcdcd*ds)/(ds*dd);
            end           
        end
    end
end

if nv==d+(d*(d-1)/2) % full
    G=[2*G(zz);diag(G)];
end

G=G.*s_k;

end

