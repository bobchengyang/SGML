function [M,league_vec] = determine_initial_M_appropriate_colors(data_cov,...
    n_feature,...
    partial_feature,...
    partial_observation,...
    partial_sample,...
    mode)
if mode==4 || mode==5 || mode==6
   data_cov=inv(data_cov);
end

[first_offdia_sign] = get_first_offdiagonal_signs(data_cov);

M=eye(n_feature);

if mode==1 || mode==4 % line-graph via cov

    M(2:n_feature+1:end)=-first_offdia_sign*0.1;
    M(n_feature+1:n_feature+1:end)=-first_offdia_sign*0.1;
    [league_vec] = initialize_league_vec(first_offdia_sign);
    
elseif mode==2 || mode==5 % tree-graph via cov

    % assign node colors
    league_vec=zeros(n_feature,1);
    
    selected_node=[];
    rng(0);
    pick_one=randperm(n_feature);
    pick_one=pick_one(1);
%      disp(['iniailly picked node: ' num2str(pick_one)])
     
    league_vec(pick_one)=1;
    previous_node_color=1; % remember the last node color
    
    selected_node=[selected_node pick_one];
    remaining_node=1:n_feature;
    remaining_node(remaining_node==pick_one)=[];
    
    while isempty(remaining_node)==0
        sn_dim=length(selected_node);
        amp=zeros(sn_dim,1); % amplitude
        row=zeros(sn_dim,1); % row
        col=zeros(sn_dim,1); % col
        sign_=zeros(sn_dim,1);% sign
        for check_max_obo=1:sn_dim
            [clo_v,clo_i]=max(abs(data_cov(selected_node(check_max_obo),remaining_node))); % index of the remaining_node
            amp(check_max_obo)=clo_v;
            row(check_max_obo)=selected_node(check_max_obo);
            col(check_max_obo)=remaining_node(clo_i);
            sign_(check_max_obo)=sign(data_cov(row(check_max_obo),col(check_max_obo)));
        end
        [~,max_i]=max(amp);
        max_i=max_i(1);
        row_max=row(max_i);
        col_max=col(max_i);
        
        sign_max=sign_(max_i);
%         disp(['later picked node: ' num2str(col_max) ' | edge sign: ' num2str(sign_max)])
        if previous_node_color==1
            if sign_max==1
                league_vec(col_max)=1;
            else
                league_vec(col_max)=-1;
            end
        else
            if sign_max==1
                league_vec(col_max)=-1;
            else
                league_vec(col_max)=1;
            end            
        end
        
        previous_node_color=league_vec(col_max); % update the last node color
        
        M(row_max,col_max)=-sign_max*(1/n_feature);
        M(col_max,row_max)=-sign_max*(1/n_feature);
        
        pick_one=col_max;
        
        selected_node=[selected_node pick_one];
        remaining_node(remaining_node==pick_one)=[];
    end
elseif mode==3 || mode==6 % full-graph via cov and then balance it
    data_cov(1:n_feature+1:end)=0;
    [BG,league_vec]=BFS_Balanced(data_cov);
    M=M-sign(BG)*1e-5;
end

end

