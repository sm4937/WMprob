function [simdata] = simBM(params,data)
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
beta = params(1); %higher beta -> more greediness
forget = params(2);
metalearning = false;

na = sum(unique(data.resp)>0);
rewards_sim = []; resp_sim = []; cor_sim = [];
for b = 1:length(unique(data.block))
    stims = data.stim(data.block==b); %stimuli
    rewards = data.rew(data.block==b); %rewards presented to subjects
    key_vec = data.coract(data.block==b); %key, correct answers
    p_reward = unique(data.prew(data.block==b));
    p_0 = (1-p_reward)./(1-p_reward + 1); %probability of getting 0 given you chose correctly is probability 
    ns = length(unique(stims));

    p = ones(ns,na)./na; %initialize with equal values at start of each block
    x = ones(ns,na); % no evidence, no priors
    resp_vec = []; cor_vec = []; %for storing responses by block
    for trial = 1:sum(data.block==b) %ntrials
        stim = stims(trial); %which stimulus is it?
        % run softmax, choose action
        p_softmax = (exp(p(stim,:).*beta)./sum(exp(p(stim,:).*beta))); %get probabilities 
        randnum = rand;
%         counts = histc(randnum,[0 cumsum(p_softmax)]); % Setting up bins of prob intervals, which one is rand number "x" in?
%         resp = find(counts==1);
        cum_dist = cumsum(p_softmax);
        if randnum < cum_dist(1)
            resp = 1;
        elseif randnum < cum_dist(2)
            resp = 2;
        else
            resp = 3;
        end
        resp_vec(trial,:) = resp;
        
        %get reward from sequence
        rew = rewards(trial);
        if key_vec(trial) == resp %correct response, give pre-determined feedback
            r = rew;
            cor_vec(trial,:) = 1;
        else %they were wrong
            r = 0;
            cor_vec(trial,:) = 0;
        end
        rewards(trial) = r; %overwrite with rewards this simulated dude is getting
        % learn!
        %likelihood changes, starts out totally uninformative
        x = cat(3,ones(size(p)),x);
        if r == 1
            x(stim,resp,1) = 1;
            notaction = [1 2 3]; notaction(resp) = [];
            x(stim,notaction,1) = 0;
        elseif r == 0
            x(stim,resp,1) = p_0; %probably of not getting rewarded this block
            notaction = [1 2 3]; notaction(resp) = [];
            x(stim,notaction,1) = (1-p_0)/2;
        end
        
        % given data in x so far, update p of each category for each
        % stimulus
        p = ones(ns,na)./na; %initialize with flat prior before inference for each trial
        for tt = 1:size(x,3) %loop over REMEMBERED trials
            p = p.*x(:,:,tt); %update w likelihood of x if category
            p = p./sum(p,2); %normalize rows
        end
        % posterior gets completely re-updated every trial, since some info
        % gets corrupted
        
        %decay to priors with forget parameter
        if length(params)>1 %then there is a forget parameter
            prior = ones(ns,na)./na;
            p = (1-forget)*p + forget*prior;
        end
        
        if ns == 6 & metalearning
            % on set size 6 blocks, do some meta-learning in terms of class
            % priors
            prior = ones(ns,na)./na;
            occurrences = sum(p==1);
            prior(:,occurrences==2) = 0;
            solved = sum(p==1,2)==1; %make solved a boolean index
            prior(solved,:) = 1;
            proto = p.*prior;
            p = proto./sum(proto,2);
        end

        % update memory, wipe some
        %according to k, what's susceptible to distortion?
        %vulnerable = 1:size(x,3) > k; 
        %flip = find(rand(sum(vulnerable),1)>0.5); %above capacity, drop with 50% probability
        %chance of information loss
        x(:,:,k+1:end) = []; %for now, delete info randomly above k
        %assign random p? randomly assign 0 or 1? 
      
    end %end of trial by trial loop
    rewards_sim = [rewards_sim; rewards]; resp_sim = [resp_sim; resp_vec];
    cor_sim = [cor_sim; cor_vec];
end %end of block by block loop

simdata = data; %everything else was the same from real data except responses/rewards
simdata.rew = rewards_sim; simdata.resp = resp_sim; simdata.cor = cor_sim;

end

