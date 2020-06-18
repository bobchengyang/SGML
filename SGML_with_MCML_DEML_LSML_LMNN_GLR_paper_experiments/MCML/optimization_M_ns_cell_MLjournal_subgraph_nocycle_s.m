function [M,...
    scaled_M,...
    scaled_factors,...
    leftEnd,...
    PN_temp,...
    M_current_eigenvector,...
    min_objective,...
    bins,...
    lower_bound] = optimization_M_ns_cell_MLjournal_subgraph_nocycle_s(N,league_vec,...
    scaled_factors,...
    feature_N,...
    counter,...
    M,...
    M_current_eigenvector,...
    Wf,...
    x,...
    rho,...
    S_upper,...
    G,...
    partial_feature,...
    bins,...
    scaled_M)
%OPTIMIZATION_M_BLOCK_CDLPT_UPDATED_NS_CELL Summary of this function goes here
%   Detailed explanation goes here

% number_which = feature_N-1;
PN_temp = zeros(feature_N-1,1);

scaled_factors_ = scaled_factors';
scaled_factors_(logical(eye(feature_N)))=[];
scaled_factors__ = reshape(scaled_factors_,feature_N-1,feature_N)';

%% linear constriants start
total_offdia = sum(1:feature_N-1);

LP_A = zeros(1+feature_N,total_offdia+feature_N);
LP_A(1,:)=[zeros(1,total_offdia) ones(1,feature_N)];
LP_b = [S_upper;zeros(feature_N,1)-rho];
%% linear constriants end

LP_lb=zeros(total_offdia+feature_N,1);
LP_ub=zeros(total_offdia+feature_N,1)+Inf;
LP_lb(total_offdia+1:10)=rho;

t_counter=0;
t_counter_c=0;
sign_vec=zeros(1,total_offdia);
sign_idx=zeros(feature_N);
scaled_factors_zero_idx = zeros(1,total_offdia);
for vec_i=1:feature_N
    for vec_j=1:feature_N
        if vec_j>vec_i
            t_counter=t_counter+1;
            if league_vec(vec_i)==league_vec(vec_j) % positive edge, negative m entry <0
                sign_vec(t_counter)=-1;
                LP_lb(t_counter)=-Inf;
                LP_ub(t_counter)=0;
            else
                sign_vec(t_counter)=1;
                LP_lb(t_counter)=0;
                LP_ub(t_counter)=Inf;
            end
            
            % get indices of scalers with 0 values
            if scaled_factors(vec_i,vec_j)==0
                scaled_factors_zero_idx(t_counter)=1;
            end
            
        end
        t_counter_c=t_counter_c+1;
        sign_idx(vec_i,t_counter_c)=t_counter;
    end
    t_counter_c=0;
end

%% fixing the off-diagonals to be 0's
LP_lb(logical(scaled_factors_zero_idx))=0;
LP_ub(logical(scaled_factors_zero_idx))=0;

sign_idx=triu(sign_idx,1);
sign_idx=sign_idx+sign_idx';
sign_idx(logical(eye(feature_N)))=[];
sign_idx=reshape(sign_idx,feature_N-1,feature_N)';

for LP_i=1:feature_N
    LP_A(LP_i+1,sign_idx(LP_i,:))=abs(scaled_factors__(LP_i,:)).*sign_vec(sign_idx(LP_i,:));
    LP_A(LP_i+1,total_offdia+LP_i)=-1;
end

LP_Aeq = [];
LP_beq = [];


%options = optimoptions('linprog','Display','none','Algorithm','interior-point');
options = optimoptions('linprog','Display','none','Algorithm','interior-point');
options.OptimalityTolerance = 1e-7;
%
% grad_g_c_dia_reorder = grad_g_c_dia;
% grad_g_c_dia_reorder(BCD)=[];
gg_list = [];
for gg = 1:feature_N-1
    gg_list = [gg_list;G(gg,gg+1:end)'];
end
net_gc=[gg_list;diag(G)];
s_k = linprog(net_gc,...
    LP_A,LP_b,...
    LP_Aeq,LP_beq,...
    LP_lb,LP_ub,options);

% else
% if length(LP_lb(LP_lb==0))==length(LP_lb) && length(LP_ub(LP_ub==0))==length(LP_ub)
%     s_k = zeros(feature_N-1,1);
% end
% end
if length(s_k) == 0
    %scaled_M=[];
    %scaled_factors=[];
    leftEnd=[];
    min_objective=mcml_obj(M,partial_feature,x);
    lower_bound=[];
    return
end
while length(s_k) == 0
    
    disp('===trying with larger OptimalityTolerance===');
    options.OptimalityTolerance = options.OptimalityTolerance*10;
    s_k = linprog(net_gc,...
        LP_A,LP_b,...
        LP_Aeq,LP_beq,...
        LP_lb,LP_ub,options);
    
end

%% proximal gradient to determine the Frank-Wolfe step size starts

M_previous = zeros(total_offdia+feature_N,1);
current_slot=0;
for Mp_i=1:feature_N-1
    od_temp_content=M(Mp_i,Mp_i+1:end);
    od_temp_length=length(od_temp_content);
    M_previous(current_slot+1:current_slot+od_temp_length)=od_temp_content;
    current_slot = current_slot+od_temp_length;
end
M_previous(total_offdia+1:end)=M(logical(eye(feature_N)));

gamma_initial=0;
tol_step_size=1e4;
M_lc=M;

objective_previous = mcml_obj(M,partial_feature,x);

objective_previous_step_size=objective_previous;
t_M21_solution_previous=s_k - M_previous;
while tol_step_size>1e-5
    
    
    t_M21 = M_previous + gamma_initial * t_M21_solution_previous;
    
    current_slot=0;
    for BCD_temp_i=1:feature_N-1
        od_temp_length=length(M_lc(BCD_temp_i,BCD_temp_i+1:end));
        M_lc(BCD_temp_i,BCD_temp_i+1:end)=t_M21(current_slot+1:current_slot+od_temp_length);
        M_lc(BCD_temp_i+1:end,BCD_temp_i)=t_M21(current_slot+1:current_slot+od_temp_length);
        current_slot = current_slot+od_temp_length;
    end
    M_lc(logical(eye(feature_N)))=t_M21(total_offdia+1:end);
    
    %     grad_g_c = 0;
    %
    %     grad_grad_g_c = 0;
    
    %% without trace(D)
    G = lmnn_gradient(M_lc,partial_feature,x);
    c_idx=0;
    for gra_idx=1:feature_N-1
        numberlength=feature_N-gra_idx;
        G(gra_idx,gra_idx+1:end)=G(gra_idx,gra_idx+1:end).*t_M21_solution_previous(c_idx+1:c_idx+numberlength)';
        G(gra_idx+1:end,gra_idx)=G(gra_idx,gra_idx+1:end);
        c_idx=c_idx+numberlength;
    end
    G(logical(eye(feature_N)))=G(logical(eye(feature_N))).*t_M21_solution_previous(total_offdia+1:end);
    grad_g_c=sum(sum(G));
    grad_grad_g_c=grad_g_c^2;
    
    L_constant = sqrt(grad_grad_g_c^2);
    
    gamma = gamma_initial - (1/L_constant) * sum(grad_g_c);
    %
    if gamma>1 || gamma<0
        gamma=gamma_initial;
        break
    end
    
    t_M21 = M_previous + gamma * t_M21_solution_previous;
    current_slot=0;
    for BCD_temp_i=1:feature_N-1
        od_temp_length=length(M_lc(BCD_temp_i,BCD_temp_i+1:end));
        M_lc(BCD_temp_i,BCD_temp_i+1:end)=t_M21(current_slot+1:current_slot+od_temp_length);
        M_lc(BCD_temp_i+1:end,BCD_temp_i)=t_M21(current_slot+1:current_slot+od_temp_length);
        current_slot = current_slot+od_temp_length;
    end
    M_lc(logical(eye(feature_N)))=t_M21(total_offdia+1:end); 
  
    objective_current_step_size = mcml_obj(M_lc,partial_feature,x);
    
    if isnan(objective_current_step_size) == 1
        asdf=0;
    end
    
    while objective_current_step_size>objective_previous_step_size
        gamma=gamma/2;
        if gamma==0
            gamma=gamma_initial;
            objective_current_step_size=objective_previous_step_size;
            break
        end
        t_M21 = M_previous + gamma * t_M21_solution_previous;
        current_slot=0;
        for BCD_temp_i=1:feature_N-1
            od_temp_length=length(M_lc(BCD_temp_i,BCD_temp_i+1:end));
            M_lc(BCD_temp_i,BCD_temp_i+1:end)=t_M21(current_slot+1:current_slot+od_temp_length);
            M_lc(BCD_temp_i+1:end,BCD_temp_i)=t_M21(current_slot+1:current_slot+od_temp_length);
            current_slot = current_slot+od_temp_length;
        end
        M_lc(logical(eye(feature_N)))=t_M21(total_offdia+1:end);
        objective_current_step_size = mcml_obj(M_lc,partial_feature,x);
    end
    disp(['step size (full) optimization obj: ' num2str(objective_current_step_size)]);
    tol_step_size = norm(objective_current_step_size - objective_previous_step_size);
    
    gamma_initial=gamma;
    objective_previous_step_size = objective_current_step_size;
end

%% proximal gradient to determine the Frank-Wolfe step size ends

M_updated_c = zeros(feature_N);

t_M21 = M_previous + gamma * t_M21_solution_previous;
current_slot=0;
for BCD_temp_i=1:feature_N-1
    od_temp_length=length(M_updated_c(BCD_temp_i,BCD_temp_i+1:end));
    M_updated_c(BCD_temp_i,BCD_temp_i+1:end)=t_M21(current_slot+1:current_slot+od_temp_length);
    M_updated_c(BCD_temp_i+1:end,BCD_temp_i)=t_M21(current_slot+1:current_slot+od_temp_length);
    current_slot = current_slot+od_temp_length;
end
M_updated_c(logical(eye(feature_N)))=t_M21(total_offdia+1:end);

%% diagonal

min_objective = mcml_obj(M_updated_c,partial_feature,x);

M_updated=M_updated_c;


%% check min eig
min_eig_M = min(eig(M_updated));

if min_eig_M < 0
    counter
    disp(['min eig M = ' num2str(min_eig_M)]);
    asdf = 1;
end

%% check connectivity begins MAY-5-2020

% thresholding

% M_check_connectivity=M_updated;
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
% bins
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
        %[vvv,ddd] = eig(M);
        %         scaling_matrix_0 = inv(diag(M_updated_current_eigenvector(:,1)));
        %         scaled_M_0 = scaling_matrix_0 * M_updated_current * inv(scaling_matrix_0);
        %         scaled_factors_0 = scaling_matrix_0 * ones(size(M_updated_current)) * inv(scaling_matrix_0);
        %             tic;
        %             scaling_matrix_0 = inv(diag(M_current_eigenvector(:,1)));
        %             scaled_M_0 = scaling_matrix_0 * M_current * inv(scaling_matrix_0);
        %             scaled_factors_0 = scaling_matrix_0 * ones(size(M_current)) * inv(scaling_matrix_0);
        scaling_matrix_0 = diag(1./M_updated_current_eigenvector(:,1));
        scaling_matrix_0_inv = diag(M_updated_current_eigenvector(:,1));
        scaled_M_0 = scaling_matrix_0 * M_updated_current * scaling_matrix_0_inv;
        scaled_factors_0 = scaling_matrix_0 * ones(temp_dim) * scaling_matrix_0_inv;
        %             toc;
        scaled_M_offdia_0 = scaled_M_0;
        scaled_M_offdia_0(scaled_M_offdia_0==diag(scaled_M_offdia_0))=0;
        leftEnd_0 = diag(scaled_M_0) - sum(abs(scaled_M_offdia_0),2);
        
        %         leftEnd_diff = sum(abs(leftEnd_0 - mean(leftEnd_0)));
        %         if leftEnd_diff > 1e-3
        %             %if not_aligned > 0
        %
        %             disp('Off-diagonal left ends not aligned!!!!');
        %             leftEnd_0
        %             asdf = 1;
        %
        %         end
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
% scaled_factors(~logical(scaled_factors_check))=1;
scaled_M_=scaled_M;
scaled_M_(logical(eye(size(M,1))))=0;
lower_bound = sum(abs(scaled_M_),2)+rho;
% lower_bound
% scaled_factors
%bins
%% check connectivity ends MAY-5-2020

M = M_updated;

% M21 = M21_updated;

end

