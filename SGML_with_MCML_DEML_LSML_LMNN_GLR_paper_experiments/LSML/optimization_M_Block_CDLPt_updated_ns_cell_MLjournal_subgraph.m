function [M,...
    scaled_M,...
    scaled_factors,...
    min_objective,...
    bins] = optimization_M_Block_CDLPt_updated_ns_cell_MLjournal_subgraph(current_node,league_vec,league_vec_temp,flip_number,...
    Ms_diagonal,...
    Ms_off_diagonal,...
    scaled_factors_v,...
    scaled_factors_h,...
    feature_N,...
    G,...
    M21,...
    m11,...
    M,...
    BCD,...
    x,...
    rho,...
    lobpcg_random_control,...
    tol_golden_search,...
    partial_feature,...
    objective_previous,...
    scaled_M,...
    scaled_factors,...
    bins)

ddd = (Ms_diagonal - rho - sum(abs(Ms_off_diagonal),2));
ddd(ddd<0)=0;%this rarely happens

LP_A = zeros(feature_N,feature_N-1);
LP_b = zeros(feature_N,1);

sign_vecdd = flip_number'*current_node*-1;
LP_A(1,:)=sign_vecdd.*abs(scaled_factors_h);
LP_b(1)=m11-rho;

scaler_v = abs(scaled_factors_v);
for LP_A_i=1:feature_N-1
    LP_A(LP_A_i+1,LP_A_i)=sign_vecdd(1,LP_A_i)*scaler_v(LP_A_i);
    LP_b(LP_A_i+1)=ddd(LP_A_i);
end

LP_lb=zeros(feature_N-1,1);
LP_ub=zeros(feature_N-1,1);

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

LP_Aeq = [];
LP_beq = [];

options = optimoptions('linprog','Display','none','Algorithm','interior-point');
options.OptimalityTolerance = 1e-7;

remaining_idx=1:feature_N;
remaining_idx(BCD)=[];

s_k = linprog(2*G(remaining_idx,BCD),...
    LP_A,LP_b,...
    LP_Aeq,LP_beq,...
    LP_lb,LP_ub,options);

if length(LP_lb(LP_lb==0))==length(LP_lb) && length(LP_ub(LP_ub==0))==length(LP_ub)
    s_k = zeros(feature_N-1,1);
end

while length(s_k) == 0
    
    disp('===trying with larger OptimalityTolerance===');
    options.OptimalityTolerance = options.OptimalityTolerance*10;
    s_k = linprog(grad_g_c,...
        LP_A,LP_b,...
        LP_Aeq,LP_beq,...
        LP_lb,LP_ub,options);
end

if isequal(league_vec,league_vec_temp)==1
    
    M_previous = M(remaining_idx,BCD);
    
    t_M21_solution_previous=s_k - M_previous;
    
    [gamma] = optimization_M_golden_section_search(...
        0,...
        1,...
        M_previous,...
        t_M21_solution_previous,...
        M,...
        partial_feature,...
        x,...
        feature_N,...
        BCD,...
        tol_golden_search);
    
else
    gamma = 1;
end

M21_updated = M21 + gamma * (s_k - M21);

M_updated=M;
M_updated(remaining_idx,BCD)=M21_updated;
M_updated(BCD,remaining_idx)=M_updated(remaining_idx,BCD);

[ L_c ] = optimization_M_set_L_Mahalanobis( partial_feature, M_updated );
min_objective = x' * L_c * x;

if min_objective>objective_previous
    min_objective=objective_previous;
    return
end

M=M_updated;
M(abs(M)<1e-5)=0;

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

scaled_M=zeros(feature_N);
scaled_factors=zeros(feature_N);

for bins_i = 1:length(unique(bins))
    M_updated_current = M_updated(bins==bins_i,bins==bins_i);
    temp_dim = size(M_updated_current,1);
    if temp_dim~=1
        rng(lobpcg_random_control);
        [M_updated_current_eigenvector] = ...
            optimization_M_lobpcg(randn(temp_dim,1),M_updated_current,1e-12,200);
        scaling_matrix_0 = diag(1./M_updated_current_eigenvector(:,1));
        scaling_matrix_0_inv = diag(M_updated_current_eigenvector(:,1));
        scaled_M_0 = scaling_matrix_0 * M_updated_current * scaling_matrix_0_inv;
        scaled_factors_0 = scaling_matrix_0 * ones(temp_dim) * scaling_matrix_0_inv;
        
        scaled_M(bins==bins_i,bins==bins_i)=scaled_M_0;
        scaled_factors(bins==bins_i,bins==bins_i)=scaled_factors_0;
        
    else
        scaled_M(bins==bins_i,bins==bins_i)=M_updated_current;
        scaled_factors(bins==bins_i,bins==bins_i)=1;
        
    end
    
end

end

