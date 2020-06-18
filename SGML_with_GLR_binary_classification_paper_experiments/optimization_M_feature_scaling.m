function [fin] = optimization_M_feature_scaling(fin,fsl,fsu)

N=size(fin,2);

for i=1:N
    fin_i=fin(:,i);
    fin(:,i)=fsl+...
        (fsu-fsl)*...
        (fin_i-min(fin_i))/(max(fin_i)-min(fin_i));
end

end

