function [simdata] = simBM_v02(params,data,model)
% a revised BM with specification & code written by Wei Ji Ma, 2021
% code written into overarching model-fitting framework by Sarah Master,
% 2021
beta = 100; forget = 0; epsilon = 0;
if sum(contains(model,'beta'))
    beta = params(contains(model,'beta'));
end
if sum(contains(model,'forget'))
    forget = params(contains(model,'forget'));
end
if sum(contains(model,'epsilon'))
    epsilon = params(contains(model,'epsilon'));
end

% if there are more free parameters, they are the p_reward_believed for
% each probability condition, p_false_believed
% i.e. p_reward_believed

K = sum(unique(data.resp)>0); %in other code, this variable is called na
rewards_sim = []; resp_sim = []; cor_sim = [];
for b = 1:length(unique(data.block))
    stims = data.stim(data.block==b); %stimuli
    rewards = data.rew(data.block==b); %rewards presented to subjects
    key_vec = data.coract(data.block==b); %key, correct answers
    p_reward_true = unique(data.prew(data.block==b));
    p_reward_believed = p_reward_true;
    p_0 = (1-p_reward_believed)./(1-p_reward_believed + 1); %probability of getting 0 given you chose correctly 
    p_false = 0; %probability of getting reward given you chose incorrectly
    ns = length(unique(stims)); 
    if ns == 6
        NperK = ns/K;
        % this code works for ns6, written by Wei Ji
        oneC = repmat(1:K, [1 NperK]);
        % false assignment for simulation purposes
        allC = unique(perms(oneC),'rows');
        numC = size(allC,1);
    else %for ns3, smaller possible hypothesis space
        allsets = nchoosek([1 2 3 1 2 3],3);
        allC=zeros(0,K);
        for i=1:size(allsets,1) %shuffle output of nchoosek
            pi = perms(allsets(i,:));
            allC = unique([allC; pi],'rows');
        end
        numC = length(allC);
    end

    prior = ones(numC, 1)/numC;
    posterior = prior;

    ntrials = sum(data.block==b);
    resp_vec = []; cor_vec = []; %for storing responses by block
    for t = 1:ntrials
        i = stims(t);  % index of stimulus, between 1 and N
        C_true = key_vec(t);

        % Decision
        marginal = NaN(1,K);
        for j = 1:K % loops over possible choices
            marginal(j) = sum(posterior(allC(:,i)==j));
        end

        p_resp = marginal.^beta;
        p_resp = epsilon/K + (1-epsilon)*(p_resp/sum(p_resp));
        C_hat = randsample(1:K,1,true,p_resp);
        % Correctness
        resp_vec(t,:) = C_hat;
        cor_vec(t,:) = C_hat==C_true;

        % Reward
        if cor_vec(t,:)
            r = rand<p_reward_true;
        else
            r = 0;
        end
        rewards(t,:) = r;
        
        % Updating the posterior
        idx = find(allC(:,i)==C_hat);
        if r == 1
%           like = p_false * ones(numC, 1); %p_false is probability of a wrong choice being rewarded
%           like(idx) = p_reward_believed;
            like = p_false./2 * ones(numC,1);
            like(idx) = 1-p_false; %100 percent chance you chose correctly if r = 1
        else
            %like = (1-p_false) * ones(numC,1);
            %like(idx) = 1-p_reward_believed;
            like = (1-p_0)./2 * ones(numC,1);
            like(idx) = p_0; %p_0 chance you chose correctly if r = 0
        end

        posterior = posterior .* like;
        posterior = posterior/sum(posterior);
        
        % Forgetting
        posterior = (1-forget)* posterior + forget* prior;
        
    end % of each trial
    % close 3
    %store information for analysis of the simulation behavior later on
    rewards_sim = [rewards_sim; rewards]; resp_sim = [resp_sim; resp_vec];
    cor_sim = [cor_sim; cor_vec];
    
end % of each block

% save all in output, simdata
simdata = data; %everything else was the same from real data except responses/rewards
simdata.rew = rewards_sim; simdata.resp = resp_sim; simdata.cor = cor_sim;

end

