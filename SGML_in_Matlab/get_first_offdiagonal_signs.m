function [sign_vec] = get_first_offdiagonal_signs(M)
dim=size(M,1);
sign_vec=zeros(dim-1,1);
counter=0;
for x=1:dim
    for y=1:dim
        if x-y==1
            counter=counter+1;
            sign_vec(counter)=sign(M(x,y));
        end
    end
end
end

