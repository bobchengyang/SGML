function [D,length_D] = get_S_and_D(n, X, y)

D=[];

for i=1:n
    for j=1:n
        if y(i)~=y(j) && i<j
            D=[D;i j];
            %S(i,j)=1;
            %d_ij = X(i,:) - X(j,:);  
            %w=w+d_ij'*d_ij;
        %else
            %D(i,j)=1;
        end
    end
end

D=X(D(:,1),:)-X(D(:,2),:);
length_D=size(D,1);
%w=w(:);
%t = w' * A(:)/100;
end

