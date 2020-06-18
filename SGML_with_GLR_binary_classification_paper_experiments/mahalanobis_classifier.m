function z=mahalanobis_classifier(m,S,X)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION
%   [z]=mahalanobis_classifier(m,S,X)
% Mahalanobis classifier for c classes.
%
% INPUT ARGUMENTS:
%   m:  lxc matrix, whose i-th column corresponds to the
%       mean of the i-th class
%   S:  lxl matrix which corresponds to the matrix
%       involved in the Mahalanobis distance (when the classes have
%       the same covariance matrix, S equals to this common covariance
%       matrix).
%   X:  lxN matrix, whose columns are the data vectors to be classified.
%
% OUTPUT ARGUMENTS:
%   z:  N-dimensional vector whose i-th component contains the label
%       of the class where the i-th data vector has been assigned.
%
% (c) 2010 S. Theodoridis, A. Pikrakis, K. Koutroumbas, D. Cavouras
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[~,c]=size(m);
[~,N]=size(X);
z=zeros(N,1);
for i=1:N
    dm=zeros(c,1);
    for j=1:c
        %dm(j)=sqrt((X(:,i)-m(:,j))'*inv(S)*(X(:,i)-m(:,j)));
        dm(j)=sqrt((X(:,i)-m(:,j))'*S*(X(:,i)-m(:,j)));
    end
    [~,z(i)]=min(dm);
end
z(z==2)=-1;
end