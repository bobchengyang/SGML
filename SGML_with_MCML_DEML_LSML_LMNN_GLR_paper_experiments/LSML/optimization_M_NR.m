function [gamma] = optimization_M_NR(...
    M_previous,...
    t_M21_solution_previous,...
    zero_mask,...
    remaining_idx,...
    BCD,...
    feature_N,...
    M_lc,...
    zz,...
    nv,...
    S,...
    D,...
    counter,...
    dia_idx,...
    tol_NR,...
    tol_GD)
%% examine the gradient at 0 and 1
[G_0] = optimization_M_gamma_M_lc(...
    M_previous,...
    0,...
    t_M21_solution_previous,...
    zero_mask,...
    remaining_idx,...
    BCD,...
    feature_N,...
    M_lc,...
    zz,...
    nv,...
    S,...
    D,...
    dia_idx);
[G_1] = optimization_M_gamma_M_lc(...
    M_previous,...
    1,...
    t_M21_solution_previous,...
    zero_mask,...
    remaining_idx,...
    BCD,...
    feature_N,...
    M_lc,...
    zz,...
    nv,...
    S,...
    D,...
    dia_idx);
if G_0>0 && G_1>0 || G_0==0 % fmin=f(gamma=0)
    gamma=0;
    %disp('early 0 early stop');
    if counter==0
        return
    end
elseif G_0<0 && G_1<0 || G_1==0 % fmin=f(gamma=1)
    gamma=1;
    %disp('early 1');
else
    gamma_NR_tol=Inf;
    gamma_NR_0=1/2;
    cNR=0;
    while gamma_NR_tol>tol_NR
        [Gnr1,Gnr2] = optimization_M_gamma_M_lc_NR(...
            M_previous,...
            gamma_NR_0,...
            t_M21_solution_previous,...
            zero_mask,...
            remaining_idx,...
            BCD,...
            feature_N,...
            M_lc,...
            zz,...
            nv,...
            S,...
            D,...
            dia_idx);
        gamma=gamma_NR_0-Gnr1/Gnr2;
        cNR=cNR+1;
        if gamma>1 || gamma<0 || isnan(gamma)==1 || cNR>20
            %disp(['gamma: ' num2str(gamma) ' cNR: ' num2str(cNR)]);
            [gamma] = optimization_M_gamma_GD(...
                M_previous,...
                t_M21_solution_previous,...
                zero_mask,...
                remaining_idx,...
                BCD,...
                feature_N,...
                M_lc,...
                zz,...
                nv,...
                S,...
                D,...
                dia_idx,...
                tol_GD);
            break
        end
        gamma_NR_tol=norm(gamma-gamma_NR_0);
        gamma_NR_0=gamma;
    end
end
end

