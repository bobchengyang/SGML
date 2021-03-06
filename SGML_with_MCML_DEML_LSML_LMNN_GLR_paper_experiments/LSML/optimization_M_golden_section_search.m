function [gamma] = optimization_M_golden_section_search(...
    xL,...
    xU,...
    M_previous,...
    t_M21_solution_previous,...
    M_lc,...
    feature_N,...
    BCD,...
    tol_golden_search,...
    zero_mask,...
    zz,...
    total_offdia,...
    remaining_idx,...
    S,...
    D,...
    dia_idx)

% fL=optimization_M_get_function_value(...
%     xL,...
%     M_previous,...
%     t_M21_solution_previous,...
%     M_lc,...
%     partial_feature,...
%     x,...
%     feature_N,...
%     BCD);

% fU=optimization_M_get_function_value(...
%     xU,...
%     M_previous,...
%     t_M21_solution_previous,...
%     M_lc,...
%     partial_feature,...
%     x,...
%     feature_N,...
%     BCD);

R=0.5*(sqrt(5)-1);
d=R*(xU-xL);
x1=xU-d;
x2=xL+d;

f1=optimization_M_get_function_value(...
    x1,...
    M_previous,...
    t_M21_solution_previous,...
    M_lc,...
    feature_N,...
    BCD,...
    zero_mask,...
    zz,...
    total_offdia,...
    remaining_idx,...
    S,...
    D,...
    dia_idx);

f2=optimization_M_get_function_value(...
    x2,...
    M_previous,...
    t_M21_solution_previous,...
    M_lc,...
    feature_N,...
    BCD,...
    zero_mask,...
    zz,...
    total_offdia,...
    remaining_idx,...
    S,...
    D,...
    dia_idx);

err=Inf;
ni=0;
while err>tol_golden_search
    ni=ni+1;
    if f1<f2
        xU=x2;
        %fU=f2;
        x2=x1;
        f2=f1;
        d=R*(xU-xL);
        x1=xU-d;
        f1=optimization_M_get_function_value(...
            x1,...
            M_previous,...
            t_M21_solution_previous,...
            M_lc,...
            feature_N,...
            BCD,...
            zero_mask,...
            zz,...
            total_offdia,...
            remaining_idx,...
            S,...
            D,...
            dia_idx);
        
    elseif f1>f2
        xL=x1;
        %fL=f1;
        x1=x2;
        f1=f2;
        d=R*(xU-xL);
        x2=xL+d;
        f2=optimization_M_get_function_value(...
            x2,...
            M_previous,...
            t_M21_solution_previous,...
            M_lc,...
            feature_N,...
            BCD,...
            zero_mask,...
            zz,...
            total_offdia,...
            remaining_idx,...
            S,...
            D,...
            dia_idx);
    else
        xL=(x1+x2)/2;
        xU=xL;
    end
    err=2*abs(xU-xL)/(xU+xL);
end
gamma=(x1+x2)/2;
%disp([num2str(ni) ' golden section search iterations']);
end

