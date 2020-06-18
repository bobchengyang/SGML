function [M,league_vec,bins] = optimization_M_balancing_a_graph(M,n_feature)

league_vec=zeros(n_feature,1);
league_vec(1)=1;%color Node 1 as blue

for i=1:n_feature
    for j=1:n_feature
        if i<j
            M(j,i)=M(i,j);
            if i==1 % color every nodes
                if M(i,j)<0 % positive edge
                    league_vec(j)=1;
                else % negative edge
                    league_vec(j)=-1;
                end
            else % remove edges
                if M(i,j)<0 % positive edge
                    if league_vec(i)~=league_vec(j) % different color
                        M(i,j)=0;  % remove edge
                        M(j,i)=M(i,j);
                    end
                else % negative edge
                    if league_vec(i)==league_vec(j) % different color
                        M(i,j)=0;  % remove edge
                        M(j,i)=M(i,j);
                    end
                end
            end
        end
    end
end

M(abs(M)<1e-5)=0;
ST=[];
for STi=1:n_feature
    for STj=1:n_feature
        if STi<STj
            if M(STi,STj)~=0
                ST=[ST [STi;STj]];
            end
        end
    end
end
if isempty(ST)~=1
    G = graph(ST(1,:),ST(2,:),[],n_feature);
else
    G = graph([],[],[],n_feature);
end
bins = conncomp(G);

end

