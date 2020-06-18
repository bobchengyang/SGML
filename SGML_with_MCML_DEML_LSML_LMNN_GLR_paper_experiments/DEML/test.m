clear;

time_vec=zeros(10,1);
for test_i=1:10
%     clearvars -except time_vec test_i
tic;
rng(0);
a=randn(500,10);
rng(0);
x=randn(500,1);
y=(x-x.').^2;
[N,n]=size(a);
y=reshape(y,[N^2 1]);
a=reshape(a,[N 1 n]);
c=reshape(a-permute(a,[2 1 3]),[N^2 n]);
rng(0);
M=randn(n);
d=c*M.*c;
e=exp(-sum(d,2));
% G1=zeros(n^2,1);
% tc=0;
% for i=1:n
%     for j=1:n
%         tc=tc+1;
%         h=e.*y.*(-c(:,i).*c(:,j));%there is a problem here
%         G1(tc)=sum(h);
%     end
% end
c=reshape(c', [n 1 N^2]);
G1=sum(e.*y.*-reshape(c.*reshape(c,[1 n N^2]), [n^2 N^2])',1);
e=reshape(e,[N N]);
e(logical(eye(n))) = 0;
L1 = diag(sum(e))-e;
G1 = reshape(G1, [n n]);
time_vec(test_i)=toc;
end
disp(['time_vec: mean ' num2str(mean(time_vec)) ' || std ' num2str(std(time_vec))]);

time_naive=zeros(10,1);
for test_i=1:10
tic;
rng(0);
a=randn(500,10);
rng(0);
x=randn(500,1);
[N,n]=size(a);
Wf = cell(N);
for i = 1:N
    for j = 1:N   
        f_i = a(i,:);     
        f_j = a(j,:);       
        f_i_j = f_i - f_j;      
        Wf{i,j} =  f_i_j';       
    end
end
rng(0);
M=randn(n);
W = zeros(N);
g = zeros(N,N,n^2);
for i = 1:N
    for j = 1:N
        W(i,j) = Wf{i,j}' * M * Wf{i,j};
        tc=0;
        for k = 1:n
            for l = 1:n
                tc=tc+1;
                g(i,j,tc)=-Wf{i,j}(k)*Wf{i,j}(l);
            end
        end
    end
end
W = exp(-W);
W(W == diag(W)) = 0;
L2 = diag(sum(W))-W;
G2=zeros(n^2,1);
for i=1:n^2
G2(i) = sum(sum(W.*((x-x.').^2).*g(:,:,i)));
end
G2 = reshape(G2, [n n]);
time_naive(test_i)=toc;
end
disp(['time_naive: mean ' num2str(mean(time_naive)) ' || std ' num2str(std(time_naive))]);
disp(['speedup: ' num2str(mean(time_naive)/mean(time_vec))]);