function [ L ] = graph_Laplacian_train_test( feature, M )

[N,n]=size(feature);
feature=reshape(feature,[N 1 n]);
c=reshape(feature-permute(feature,[2 1 3]),[N^2 n]);
W=exp(-sum(c*M.*c,2));
W=reshape(W, [N N]);
W(1:N+1:end) = 0;
L = diag(sum(W))-W;
end

