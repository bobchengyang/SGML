function [ G ] = optimization_M_set_step_gradient( S, D, d, M, nv, zz, BCD, remaining_idx, s_k )

ds=sum(S*M.*S,2).^(1/2); % Ns x 1
dd=sum(D*M.*D,2).^(1/2); % Nd x 1
ds_dd=ds-dd'; % Ns x Nd
[I,J]=find(ds_dd>0); % row and column of ds>dd
length_IJ=length(I);

ds_I=reshape(ds(I),[1 1 length_IJ]); % 1 x 1 x length_IJ
dd_J=reshape(dd(J),[1 1 length_IJ]); % 1 x 1 x length_IJ

dabab=S(I,:); % length_IJ x d
dabab=reshape(dabab',[d 1 length_IJ]); % d x 1 x length_IJ
dcdcd=D(J,:); % length_IJ x d
dcdcd=reshape(dcdcd',[d 1 length_IJ]); % d x 1 x length_IJ
    
if nv==d+(d*(d-1)/2) && BCD==0 % full   

    dabab=dabab.*permute(dabab,[2 1 3]); % d x d x length_IJ
    dcdcd=dcdcd.*permute(dcdcd,[2 1 3]); % d x d x length_IJ
    
%     G=sum(dabab+...
%         dcdcd-...
%         (dabab.*dd_J+dcdcd.*ds_I)./(ds_I.*dd_J),3); % d x d
      
%     G=dabab+...
%         dcdcd-...
%         (dabab.*dd+dcdcd.*ds)./(ds*dd);

elseif nv==d+d-1 % diagonals + one row/column of off-diagonals
    
    dabab=[dabab(BCD,:,:).*dabab(remaining_idx,:,:); dabab.*dabab]; % (d+d-1) x 1 x length_IJ
    dcdcd=[dcdcd(BCD,:,:).*dcdcd(remaining_idx,:,:); dcdcd.*dcdcd]; % (d+d-1) x 1 x length_IJ
    
%     G=sum(dabab+...
%         dcdcd-...
%         (dabab.*dd_J+dcdcd.*ds_I)./(ds_I.*dd_J),3); % (d+d-1) x 1  
    
elseif nv==d % diagonals
    
    dabab=dabab.*dabab; % d x 1 x length_IJ
    dcdcd=dcdcd.*dcdcd; % d x 1 x length_IJ
    
%     G=sum(dabab+...
%         dcdcd-...
%         (dabab.*dd_J+dcdcd.*ds_I)./(ds_I.*dd_J),3); % d x 1    
    
else % one row/column of off-diagonals
    
    dabab=dabab(BCD,:,:).*dabab(remaining_idx,:,:); % (d-1) x 1 x length_IJ
    dcdcd=dcdcd(BCD,:,:).*dcdcd(remaining_idx,:,:); % (d-1) x 1 x length_IJ
    
%     G=sum(dabab+...
%         dcdcd-...
%         (dabab.*dd_J+dcdcd.*ds_I)./(ds_I.*dd_J),3); % (d-1) x 1  
    
end  

G=sum(dabab+...
    dcdcd-...
    (dabab.*dd_J+dcdcd.*ds_I)./(ds_I.*dd_J),3);

if nv==d+(d*(d-1)/2) && BCD==0 % full
    G=[2*G(zz);diag(G)];
elseif nv==d+d-1 % diagonals + one row/column of off-diagonals
    G(1:d-1)=G(1:d-1).*2;
end

if nv==d-1 % one row/column of off-diagonals
    G=G.*2;
end

% if nv==d+(d*(d-1)/2) % full
%     G=zeros(d);
% elseif nv==d+d-1 % diagonals + one row/column of off-diagonals
%     G=zeros(d+d-1,1);
% elseif nv==d % diagonals
%     G=zeros(d,1);
% else % one row/column of off-diagonals
%     G=zeros(d-1,1);
% end        
% 
% for i=1:size(S,1)
%     for j=1:size(D,1)
%         if nv==d+(d*(d-1)/2) % full
%             d_ab=X(S(i,1),:)-X(S(i,2),:);
%             d_cd=X(D(j,1),:)-X(D(j,2),:);
%             ds=sqrt(d_ab*M*d_ab');
%             dd=sqrt(d_cd*M*d_cd');
%             if ds>dd
%                 dabab=d_ab'*d_ab;
%                 dcdcd=d_cd'*d_cd;
%                 G=G+...
%                     dabab+...
%                     dcdcd-...
%                     (dabab*dd+dcdcd*ds)/(ds*dd);
%             end
%         elseif nv==d+d-1 % diagonals + one row/column of off-diagonals
%             dabab=[2*(X(S(i,1),BCD)-X(S(i,2),BCD))'.*(X(S(i,1),remaining_idx)-X(S(i,2),remaining_idx))';(X(S(i,1),:)-X(S(i,2),:))'.^2];
%             dcdcd=[2*(X(D(j,1),BCD)-X(D(j,2),BCD))'.*(X(D(j,1),remaining_idx)-X(D(j,2),remaining_idx))';(X(D(j,1),:)-X(D(j,2),:))'.^2];            
%             d_ab=X(S(i,1),:)-X(S(i,2),:);
%             d_cd=X(D(j,1),:)-X(D(j,2),:);
%             ds=sqrt(d_ab*M*d_ab');
%             dd=sqrt(d_cd*M*d_cd');
%             if ds>dd
%                 G=G+...
%                     dabab+...
%                     dcdcd-...
%                     (dabab*dd+dcdcd*ds)/(ds*dd);
%             end          
%         elseif nv==d % diagonals
%             dabab=(X(S(i,1),:)-X(S(i,2),:))'.^2;
%             dcdcd=(X(D(j,1),:)-X(D(j,2),:))'.^2;            
%             d_ab=X(S(i,1),:)-X(S(i,2),:);
%             d_cd=X(D(j,1),:)-X(D(j,2),:);
%             ds=sqrt(d_ab*M*d_ab');
%             dd=sqrt(d_cd*M*d_cd');
%             if ds>dd
%                 G=G+...
%                     dabab+...
%                     dcdcd-...
%                     (dabab*dd+dcdcd*ds)/(ds*dd);
%             end            
%         else % one row/column of off-diagonals
%             dabab=2*(X(S(i,1),BCD)-X(S(i,2),BCD))'.*(X(S(i,1),remaining_idx)-X(S(i,2),remaining_idx))';
%             dcdcd=2*(X(D(j,1),BCD)-X(D(j,2),BCD))'.*(X(D(j,1),remaining_idx)-X(D(j,2),remaining_idx))';            
%             d_ab=X(S(i,1),:)-X(S(i,2),:);
%             d_cd=X(D(j,1),:)-X(D(j,2),:);
%             ds=sqrt(d_ab*M*d_ab');
%             dd=sqrt(d_cd*M*d_cd');
%             if ds>dd
%                 G=G+...
%                     dabab+...
%                     dcdcd-...
%                     (dabab*dd+dcdcd*ds)/(ds*dd);
%             end           
%         end
%     end
% end

G=G.*s_k;

end

