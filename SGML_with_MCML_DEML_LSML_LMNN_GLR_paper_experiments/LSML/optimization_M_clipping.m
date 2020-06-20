function [M_diagonal]=optimization_M_clipping(M_diagonal,lower_bound,S_upper)

alpha_c=M_diagonal-lower_bound;% 'pre-clipping'
[alpha_c_sorted,alpha_c_sorted_idx]=sort(alpha_c);% ascending order
alpha_c_0idx=alpha_c_sorted<=0;% these <=0 ones are already clipped by lower_bound
alpha_c_sorted(alpha_c_0idx)=[];% remove these <=0 ones (values)
alpha_c_sorted_idx(alpha_c_0idx)=[];% remove these <=0 ones (indices)

alpha_c_idx=alpha_c<=0;

num_i=length(alpha_c_sorted_idx);

for i=1:num_i
    if i==1
        alpha_temp = (sum(M_diagonal(alpha_c_sorted_idx))...
            +sum(lower_bound(alpha_c_idx))...
            -S_upper)/num_i;

        if alpha_temp>0 && alpha_temp<=alpha_c_sorted(i)
            alpha_final = alpha_temp;
            break
        end
    else
        alpha_temp = (sum([lower_bound(alpha_c_sorted_idx(1:i-1)); M_diagonal(alpha_c_sorted_idx(i:end))])...
            +sum(lower_bound(alpha_c_idx))...
            -S_upper)/(num_i-i+1);
        if alpha_temp>=alpha_c_sorted(i-1) && alpha_temp<=alpha_c_sorted(i)
            alpha_final = alpha_temp;
            break
        end   
        %disp([num2str(alpha_temp) ' ' num2str(alpha_c_sorted(i-1)) ' ' num2str(alpha_c_sorted(i))]);
    end
    
end

try
    M_diagonal = max([M_diagonal-alpha_final lower_bound],[],2);
catch
    for i=1:num_i
        M_diagonal0 = max([M_diagonal-alpha_c_sorted(i) lower_bound],[],2);
        if sum(M_diagonal0)-S_upper<=0 % numerical fix
            M_diagonal = M_diagonal0;
            break
        end
    end
end

end



