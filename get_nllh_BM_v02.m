function [nllh] = get_nllh_BM_v02(params)
% a revised BM with specification & code written by Wei Ji Ma, 2021
% code written into overarching model-fitting framework by Sarah Master,
% 2021
% the parameters of simplest version are forget rate, and decision noise (beta)
beta = params(1)*10;
forget = params(2);
% if there are more free parameters, they are the p_reward_believed for
% each probability condition
% i.e. p_reward_believed

llh = 0;
global onesubj
data = onesubj;
% this is NOT the simulation code
% this track p(action) across param values to return negative log
% likelihood for those specific param values

K = sum(unique(data.resp)>0); %in other code, this variable is called na
rewards_sim = []; resp_sim = []; cor_sim = [];
for b = 1:length(unique(data.block))
    stims = data.stim(data.block==b); %stimuli
    rewards = data.rew(data.block==b); %rewards presented to subjects
    key_vec = data.coract(data.block==b); %key, correct answers
    cor_vec = data.cor(data.block==b);
    resp_vec = data.resp(data.block==b);
    p_reward_true = unique(data.prew(data.block==b));
    p_reward_believed = p_reward_true;
    p_false = 0; %probability of getting 1 given you chose incorrectly
    p_0 = (1-p_reward_believed)./(1-p_reward_believed + 1); %probability of getting 0 given you chose correctly is probability 
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
    for t = 1:ntrials
        i = stims(t);  % index of stimulus, between 1 and N

        % Decision
        marginal = NaN(1,K);
        for j = 1:K % loops over possible choices
            marginal(j) = sum(posterior(allC(:,i)==j));
        end

        p_resp = marginal.^beta;
        p_resp = p_resp/sum(p_resp);
        %C_hat = randsample(1:K,1,true,p_resp);
        C_hat = resp_vec(t);
        if C_hat > 0 %not a trial with non-response
            llh = llh + log(p_resp(C_hat));
        end

        % Reward
        r = rewards(t);
        
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
        posterior = (1-forget)*posterior + forget*prior;
    end % of each trial
end % of each block

nllh = -llh; %flip sign for final output

end

