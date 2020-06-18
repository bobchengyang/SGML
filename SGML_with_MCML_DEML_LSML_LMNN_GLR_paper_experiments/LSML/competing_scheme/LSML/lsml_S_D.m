function [S,D] = lsml_S_D(y,N,X)

S=[];
D=[];

for i=1:N
    for j=1:N
        if i<j
            if y(i)==y(j)
                S=[S;i j];
            else
                D=[D;i j];
            end
        end
    end
end

S=X(S(:,1),:)-X(S(:,2),:);
D=X(D(:,1),:)-X(D(:,2),:);

end

