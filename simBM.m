function [simdata] = simBM(params,data,model)
% simBM
% sim WMP data with BM model
% This version of the BI model has a hard capacity (k) limit for how much
% information can fit in storage. 
% Posterior is entirely re-computed every trial based on the likelihood
% (evidence/reward) information still held in memory
% U_action = U_correct (1) * p(C=a|x)
% Where x is all the reward information you have for that stimulus

% p(C|x) = U_C*p(C)*p(x|C)
% p(C|x_vec) = U_C*p(C)*p(x_1|C)*p(x_2|C)*...*p(x_t|C)
% uninformative prior, re-update posterior every time, don't just learn

global k
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

K = sum(unique(data.resp)>0);
rewards_sim = []; resp_sim = []; cor_sim = [];
for b = 1:length(unique(data.block))
    stims = data.stim(data.block==b); %stimuli
    rewards = data.rew(data.block==b); %rewards presented to subjects
    key_vec = data.coract(data.block==b); %key, correct answers
    p_reward = unique(data.prew(data.block==b));
    p_0 = (1-p_reward)./(1-p_reward + 1); %probability of getting 0 given you chose correctly 
    ns = length(unique(stims));

    prior = ones(ns,K)./K; p = prior;
     %initialize with equal values at start of each block
    resp_vec = []; cor_vec = []; %for storing responses by block
    for trial = 1:sum(data.block==b) %ntrials
        stim = stims(trial); %which stimulus is it?
        % run softmax, choose action
        p_softmax = epsilon/K + (1-epsilon)*(exp(p(stim,:).*beta)./sum(exp(p(stim,:).*beta))); %get probabilities 
        resp = randsample(1:K,1,true,p_softmax);
        resp_vec(trial,:) = resp;
        
        %get reward from sequence
        rew = rewards(trial);
        if key_vec(trial) == resp %correct response, give pre-determined feedback
            r = rew;
            cor_vec(trial,:) = 1;
        else %they were wrongs
            r = 0;
            cor_vec(trial,:) = 0;
        end
        rewards(trial) = r; %overwrite with rewards this simulated dude is getting
        % learn!
        %likelihood changes
        x = ones(size(p)); %starts out totally uninformative
        if r == 1
            x(stim,resp) = 1;
            notaction = [1 2 3]; notaction(resp) = [];
            x(stim,notaction) = 0;
        elseif r == 0
            x(stim,resp) = p_0; %probably of not getting rewarded this block
            notaction = [1 2 3]; notaction(resp) = [];
            x(stim,notaction) = (1-p_0)/2;
        end
        
        if k > 0 %if there is a capacity, k, then do forgetting this way
            % given data in x so far, update p of each category for each
            % stimulus
            p = ones(ns,K)./K; %initialize with flat prior before inference for each trial
            for tt = 1:size(x,3) %loop over REMEMBERED trials
               p = p.*x(:,:,tt); %update w likelihood of x if category
               p = p./sum(p,2); %normalize rows
            end
            % posterior gets completely re-updated every trial, since some info
            % gets corrupted
        else % just update posterior from previous one
            p = p.*x;
            p = p./sum(p,2);
        end
        
        %decay to priors with forget parameter
        p = (1-forget)*p + forget*prior;
        % ^ this is replacing the hard capacity limits

        if k > 0; x(:,:,k+1:end) = []; end %delete all info outside capacity
      
    end %end of trial by trial loop
    rewards_sim = [rewards_sim; rewards]; resp_sim = [resp_sim; resp_vec];
    cor_sim = [cor_sim; cor_vec];
end %end of block by block loop

simdata = data; %everything else was the same from real data except responses/rewards
simdata.rew = rewards_sim; simdata.resp = resp_sim; simdata.cor = cor_sim;

end

