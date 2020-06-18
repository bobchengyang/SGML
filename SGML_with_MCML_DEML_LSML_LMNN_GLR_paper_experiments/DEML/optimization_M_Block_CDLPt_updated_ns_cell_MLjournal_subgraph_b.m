function [M,...
    scaled_M,...
    scaled_factors,...
    leftEnd,...
    PN_temp,...
    M_current_eigenvector,...
    M21,...
    min_objective,...
    bins,...
    lower_bounds,...
    invalid_result] = optimization_M_Block_CDLPt_updated_ns_cell_MLjournal_subgraph_b(N,current_node,league_vec,league_vec_temp,flip_number,...
    Ms_diagonal,...
    Ms_off_diagonal,...
    scaled_factors_v,...
    scaled_factors_h,...
    scaled_M,...
    scaled_factors,...
    M21_updated_ub___,...
    M21_updated_ub,...
    feature_N,...
    G,...
    counter,...
    M21,...
    m11,...
    M22,...
    M,...
    BCD,...
    M_current_eigenvector,...
    Wf,...
    x,...
    rho,...
    S_upper,...
    lower_bounds,...
    partial_feature,...
    bins)
%OPTIMIZATION_M_BLOCK_CDLPT_UPDATED_NS_CELL Summary of this function goes here
%   Detailed explanation goes here

% number_which = feature_N-1;
PN_temp = zeros(feature_N-1,1);

%% linear constriants start
%     ddd = (Ms_diagonal - rho - sum(abs(Ms_off_diagonal),2))./abs(scaled_factors_v);

ddd = (0 - rho - sum(abs(Ms_off_diagonal),2));
% ddd(ddd<0)=0;%this rarely happens

%     ddd(ddd<0)=0;
%     ddd(ddd==Inf)=0;
%     ddd(ddd==-Inf)=0;
%     ddd(isnan(ddd))=0;

LP_A = zeros(1+feature_N,feature_N-1+feature_N);
LP_b = zeros(1+feature_N,1);
sign_vecdd = flip_number'*current_node*-1;
LP_A(1,1:feature_N-1)=sign_vecdd.*abs(scaled_factors_h);
LP_A(1,feature_N-1+BCD)=-1;
LP_b(1)=-rho;
scaler_v = abs(scaled_factors_v);

remaining_idx = 1:feature_N;
remaining_idx(BCD)=[];
for LP_A_i=1:feature_N-1
    LP_A(LP_A_i+1,LP_A_i)=sign_vecdd(1,LP_A_i)*scaler_v(LP_A_i);
    LP_A(LP_A_i+1,feature_N-1+remaining_idx(LP_A_i))=-1;
    LP_b(LP_A_i+1)=ddd(LP_A_i);
end
LP_A(end,feature_N-1+1:end)=1;
LP_b(end)=S_upper;

% %% LP_lb LP_ub setting starts
LP_lb=zeros(feature_N-1+feature_N,1);
LP_ub=zeros(feature_N-1+feature_N,1);

for LP_lb_i=1:feature_N-1
    if sign(sign_vecdd(LP_lb_i))==-1 %<0
        LP_lb(LP_lb_i)=-Inf;
        LP_ub(LP_lb_i)=0;
    else %>0
        LP_lb(LP_lb_i)=0;
        LP_ub(LP_lb_i)=Inf;
    end
end


for LP_lb_i=1:feature_N-1
    if scaler_v(LP_lb_i)==0
        LP_lb(LP_lb_i)=0;
        LP_ub(LP_lb_i)=0;
    end
end

LP_lb(feature_N-1+1:end)=rho;
LP_ub(feature_N-1+1:end)=Inf;

LP_Aeq = [];
LP_beq = [];

options = optimoptions('linprog','Display','none','Algorithm','interior-point');
options.OptimalityTolerance = 1e-7;
remaining_idx=1:feature_N;
remaining_idx(BCD)=[];
lp_co = [G(BCD,remaining_idx)';diag(G)];
s_k = linprog(lp_co,...
    LP_A,LP_b,...
    LP_Aeq,LP_beq,...
    LP_lb,LP_ub,options);

% else
if length(LP_lb(LP_lb==0))==length(LP_lb) && length(LP_ub(LP_ub==0))==length(LP_ub)
    s_k = zeros(feature_N-1,1);
end
% end

while length(s_k) == 0
    
    disp('===trying with larger OptimalityTolerance===');
%     invalid_result=0;
%     min_objective = mcml_obj(M,partial_feature,x);
%     leftEnd=[];
%     return
    options.OptimalityTolerance = options.OptimalityTolerance*10;
    s_k = linprog(lp_co,...
        LP_A,LP_b,...
        LP_Aeq,LP_beq,...
        LP_lb,LP_ub,options);
    
end

if isequal(league_vec,league_vec_temp)==1
    gamma_initial=0;
    tol_step_size=1e4;
    M_lc=M;
    
    [ L ] = optimization_M_set_L_Mahalanobis( N, Wf, M );
    
    objective_previous=mcml_obj(M_lc,partial_feature,x);
    objective_previous_step_size=objective_previous;
    
    M_previous=zeros(feature_N-1+feature_N,1);
    M_previous(1:feature_N-1)=M21;
    M_previous(feature_N-1+1:end)=M(logical(eye(feature_N)));
    
    while tol_step_size>1e-5
        
        t_M21_solution_previous=s_k - M_previous;
        t_M21 = M_previous + gamma_initial * t_M21_solution_previous;
        if BCD == 1
            M_lc(1,2:end) = t_M21(1:feature_N-1)';
            M_lc(2:end,1) = t_M21(1:feature_N-1);
        elseif BCD == feature_N
            M_lc(end,1:end-1) = t_M21(1:feature_N-1)';
            M_lc(1:end-1,end) = t_M21(1:feature_N-1);
        else
            M_lc(BCD,[1:BCD-1 BCD+1:end]) = t_M21(1:feature_N-1)';
            M_lc([1:BCD-1 BCD+1:end],BCD) = t_M21(1:feature_N-1);
        end
        
        G = lmnn_gradient(M_lc,partial_feature,x);
        %     grad_g_c = 0;
        %     grad_grad_g_c = 0;
        %
        %     %% without trace(D)
        %     for i = 1:N
        %         for j = 1:N
        %             if BCD == 1
        %                 grad_g_c_unit = (-2)*t_M21_solution_previous(1:feature_N-1)' * (Wf{i,j}(1) * Wf{i,j}(2:end));
        %             elseif BCD == feature_N
        %                 grad_g_c_unit = (-2)*t_M21_solution_previous(1:feature_N-1)' * (Wf{i,j}(end) * Wf{i,j}(1:end-1));
        %             else
        %                 grad_g_c_unit = (-2)*t_M21_solution_previous(1:feature_N-1)' * (Wf{i,j}(BCD) * Wf{i,j}([1:BCD-1 BCD+1:end]));
        %             end
        %             grad_g_c_unit=grad_g_c_unit+(-1)*t_M21_solution_previous(feature_N-1+1:end)'*(Wf{i,j}.^2);
        %
        %             core_unit = exp(-Wf{i,j}'*M_lc*Wf{i,j})*((x(i)-x(j))^2);
        %             grad_g_c = grad_g_c + grad_g_c_unit * core_unit;
        %             grad_grad_g_c = grad_grad_g_c + grad_g_c_unit * grad_g_c_unit' * core_unit;
        %         end
        %     end
        grad_g_c=[G(BCD,remaining_idx)'.*t_M21_solution_previous(1:feature_N-1);diag(G).*t_M21_solution_previous(feature_N-1+1:end)];
        grad_g_c=sum(grad_g_c);
        grad_grad_g_c=grad_g_c^2;
        
        L_constant = sqrt(grad_grad_g_c^2);
        
        gamma = gamma_initial - (1/L_constant) * sum(grad_g_c);
        if gamma>1 || gamma<0 || isnan(gamma)==1
            gamma=gamma_initial;
            break
        end
        
        t_M21 = M_previous + gamma * t_M21_solution_previous;
        if BCD == 1
            M_lc(1,2:end) = t_M21(1:feature_N-1)';
            M_lc(2:end,1) = t_M21(1:feature_N-1);
        elseif BCD == feature_N
            M_lc(end,1:end-1) = t_M21(1:feature_N-1)';
            M_lc(1:end-1,end) = t_M21(1:feature_N-1);
        else
            M_lc(BCD,[1:BCD-1 BCD+1:end]) = t_M21(1:feature_N-1)';
            M_lc([1:BCD-1 BCD+1:end],BCD) = t_M21(1:feature_N-1);
        end
        M_lc(logical(eye(feature_N)))=t_M21(feature_N-1+1:end);
        
        [ L ] = optimization_M_set_L_Mahalanobis( N, Wf, M_lc );
        
        objective_current_step_size=mcml_obj(M_lc,partial_feature,x);
        while objective_current_step_size>objective_previous_step_size
            gamma=gamma/2;
            if gamma==0
                gamma=gamma_initial;
                objective_current_step_size=objective_previous_step_size;
                break
            end
            t_M21 = M_previous + gamma * t_M21_solution_previous;
            if BCD == 1
                M_lc(1,2:end) = t_M21(1:feature_N-1)';
                M_lc(2:end,1) = t_M21(1:feature_N-1);
            elseif BCD == feature_N
                M_lc(end,1:end-1) = t_M21(1:feature_N-1)';
                M_lc(1:end-1,end) = t_M21(1:feature_N-1);
            else
                M_lc(BCD,[1:BCD-1 BCD+1:end]) = t_M21(1:feature_N-1)';
                M_lc([1:BCD-1 BCD+1:end],BCD) = t_M21(1:feature_N-1);
            end
            M_lc(logical(eye(feature_N)))=t_M21(feature_N-1+1:end);
            objective_current_step_size=mcml_obj(M_lc,partial_feature,x);
        end
        
        disp(['step size (offdia) optimization obj: ' num2str(objective_current_step_size)]);
        tol_step_size = norm(objective_current_step_size - objective_previous_step_size);
        
        gamma_initial=gamma;
        objective_previous_step_size = objective_current_step_size;
    end
    
else
    gamma = 1;
    M_previous=zeros(feature_N-1+feature_N,1);
    M_previous(1:feature_N-1)=M21;
    M_previous(feature_N-1+1:end)=M(logical(eye(feature_N)));
    t_M21_solution_previous=s_k - M_previous;
end

M21_updated = M_previous + gamma * t_M21_solution_previous;

% M21_updated = s_k(1:feature_N-1);
M_updated_c = zeros(feature_N);
if BCD == 1
    M_updated_c(1,1) = m11;
    M_updated_c(1,2:end) = M21_updated(1:feature_N-1)';
    M_updated_c(2:end,1) = M21_updated(1:feature_N-1);
    M_updated_c(2:end,2:end) = M22;
elseif BCD == feature_N
    M_updated_c(end,end) = m11;
    M_updated_c(end,1:end-1) = M21_updated(1:feature_N-1)';
    M_updated_c(1:end-1,end) = M21_updated(1:feature_N-1);
    M_updated_c(1:end-1,1:end-1) = M22;
else
    M_updated_c(BCD,BCD) = m11;
    M_updated_c(BCD,[1:BCD-1 BCD+1:end]) = M21_updated(1:feature_N-1)';
    M_updated_c([1:BCD-1 BCD+1:end],BCD) = M21_updated(1:feature_N-1);
    M_updated_c([1:BCD-1 BCD+1:end],[1:BCD-1 BCD+1:end]) = M22;
end
M_updated_c(logical(eye(feature_N)))=M21_updated(feature_N-1+1:end);

[ L_c ] = optimization_M_set_L_Mahalanobis( length(x), Wf, M_updated_c );
min_objective=mcml_obj(M_updated_c,partial_feature,x);
%M21_updated = 0.5 * s_k(1:feature_N-1) + 0.5 * ( M21 + gamma * ( s_k(1:feature_N-1) - M21) ) ;

%M21_updated = s_k(1:feature_N-1);

%     M21_updated = (1 - gamma) * M21 + ...
%         gamma * s_k(1:feature_N-1);
M_updated=M_updated_c;


%% check min eig
% min_eig_M = min(eig(M_updated));
%
% if min_eig_M < 0
%     counter
%     disp(['min eig M = ' num2str(min_eig_M)]);
%     asdf = 1;
% end

%% check connectivity begins MAY-5-2020

% thresholding

% M_before_thresholding=M_updated;
M_updated(abs(M_updated)<1e-5)=0;

ST=[];
for STi=1:feature_N
    for STj=1:feature_N
        if STi<STj
            if M_updated(STi,STj)~=0
                ST=[ST [STi;STj]];
            end
        end
    end
end
if length(ST)~=0
    G = graph(ST(1,:),ST(2,:),[],feature_N);
else
    G = graph([],[],[],feature_N);
end
bins = conncomp(G);
leftEnd=zeros(feature_N,1);
scaled_M=zeros(feature_N);
scaled_factors=zeros(feature_N);
scaled_factors_check=zeros(feature_N);
% if length(unique(bins))>1
%     ttasdflk = 1;
%
% end
for bins_i = 1:length(unique(bins))
    M_updated_current = M_updated(bins==bins_i,bins==bins_i);
    temp_dim = size(M_updated_current,1);
    if temp_dim~=1
        [M_updated_current_eigenvector,M_min_eig,failureflag,~,residual_norm_history0] = ...
            optimization_M_lobpcg(randn(size(M_updated_current,1),1),M_updated_current,1e-12,200);
        scaling_matrix_0 = diag(1./M_updated_current_eigenvector(:,1));
        scaling_matrix_0_inv = diag(M_updated_current_eigenvector(:,1));
        scaled_M_0 = scaling_matrix_0 * M_updated_current * scaling_matrix_0_inv;
        scaled_factors_0 = scaling_matrix_0 * ones(temp_dim) * scaling_matrix_0_inv;
        scaled_M_offdia_0 = scaled_M_0;
        scaled_M_offdia_0(scaled_M_offdia_0==diag(scaled_M_offdia_0))=0;
        leftEnd_0 = diag(scaled_M_0) - sum(abs(scaled_M_offdia_0),2);
        
        leftEnd_diff = sum(abs(leftEnd_0 - mean(leftEnd_0)));
        if leftEnd_diff > 1e-3
            %if not_aligned > 0
            
            disp('Off-diagonal left ends not aligned!!!!');
            leftEnd_0
            asdf = 1;
            
        end
        scaled_M(bins==bins_i,bins==bins_i)=scaled_M_0;
        
        scaled_factors(bins==bins_i,bins==bins_i)=scaled_factors_0;
        scaled_factors_check(bins==bins_i,bins==bins_i)=1;
        
        leftEnd(bins==bins_i)=leftEnd_0;
    else
        scaled_M(bins==bins_i,bins==bins_i)=M_updated_current;
        
        
        %         scaled_M(bins==bins_i,bins~=bins_i)=Inf;
        %         scaled_M(bins~=bins_i,bins==bins_i)=Inf;
        
        scaled_factors(bins==bins_i,bins==bins_i)=1;
        scaled_factors_check(bins==bins_i,bins==bins_i)=1;
        
    end
    
    %     M(bins==bins_i,bins==bins_i)=M_updated_current;
    
end
scaled_M_=scaled_M;
scaled_M_(logical(eye(size(M,1))))=0;
lower_bounds = sum(abs(scaled_M_),2) + rho;

if sum(lower_bounds) > S_upper
    invalid_result=1;
    disp(['lower bounds sum:' num2str(sum(lower_bounds))]);
    disp('========lower bounds sum larger than S_upper!!!========');
    return
    
end
invalid_result=0;
%     if sum(diag(M_updated))>S_upper
%         factor_for_diag = sum(diag(M))/S_upper;
%         M = M/factor_for_diag;
%     end
% scaled_factors(~logical(scaled_factors_check))=1;

% scaled_factors
%bins
%% check connectivity ends MAY-5-2020

M = M_updated;

M21 = M21_updated(1:feature_N-1);

end

