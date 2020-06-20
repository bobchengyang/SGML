function [C] = lsml_obj(M,S,D)

ds=sum(S*M.*S,2).^(1/2); % Ns x 1
dd=sum(D*M.*D,2).^(1/2); % Nd x 1

ds_dd=ds-dd'; % Ns x Nd

ds_dd_idx=ds_dd>0; % indices of ds>dd

C=sum(ds_dd(ds_dd_idx).^2); % add up (ds-dd)^2

% C=0;
% for i=1:size(S,1)
%     for j=1:size(D,1)
%         d_ab=X(S(i,1),:)-X(S(i,2),:);
%         d_cd=X(D(j,1),:)-X(D(j,2),:);
%         ds=sqrt(d_ab*M*d_ab');
%         dd=sqrt(d_cd*M*d_cd');
%         if ds>dd
%             C=C+(ds-dd)^2;
%         end
%     end
% end

% C=C+trace(M)-log(det(M)); % comment this if one does not want
% regularization

end

