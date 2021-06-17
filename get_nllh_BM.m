function [nllh] = get_nllh_BM(params)
% fit WMP data with BM model 
% Where x is all the reward information you have for that stimulus
llh = 0;
%p(C|x) = U_C*p(C)*p(x|C)
%p(C|x_vec) = U_C*p(C)*p(x_1|C)*p(x_2|C)*...*p(x_t|C)
%uninformative prior, re-update posterior every time, don't just learn

global k onesubj
data = onesubj; 
beta = params(1)*100; %higher beta -> more greediness
forget = params(2);
metalearning = false; 

na = sum(unique(data.resp)>0);
rewards_sim = []; resp_sim = []; cor_sim = [];
for b = 1:length(unique(data.block))
    stims = data.stim(data.block==b); %stimuli
    rewards = data.rew(data.block==b); %rewards presented to subjects
    cor_vec = data.cor(data.block==b);
    resp_vec = data.resp(data.block==b);
    p_reward = unique(data.prew(data.block==b));
    p_0 = (1-p_reward)./(1-p_reward + 1); %probability of getting 0 given you chose correctly is probability 
    % you chose correctly and got 0 (1-preward), normalized with probability you chose
    % incorrectly and got 0 (1)
    ns = length(unique(stims));

    p = ones(ns,na)./na; %initialize with equal values at start of each block
    x = ones(ns,na,1); % no evidence, no priors
    for trial = 1:sum(data.block==b) %ntrials
        stim = stims(trial); %which stimulus is it?
        % run softmax, choose action
        p_softmax = (exp(p(stim,:).*beta)./sum(exp(p(stim,:).*beta))); %get probabilities 
        
        resp = resp_vec(trial);
        if resp > 0
            p_a = p_softmax(resp);
            llh = llh + log(p_a);
        end
        %get reward from sequence
        rew = rewards(trial);
        if cor_vec(trial) %correct response, give pre-determined feedback
            r = rew;
        else %they were wrong
            r = 0;
        end
        % learn!
        %likelihood changes, starts out totally uninformative
        x = cat(3,ones(size(p)),x);
        if resp > 0 %non-response trials, don't do any inference
            if r == 1
                x(stim,resp,1) = 1;
                notaction = [1 2 3]; notaction(resp) = [];
                x(stim,notaction,1) = 0;
            elseif r == 0
                x(stim,resp,1) = p_0; %probably of not getting rewarded this block
                notaction = [1 2 3]; notaction(resp) = [];
                x(stim,notaction,1) = (1-p_0)/2;
            end
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
        
        %decay to priors w forget parameter
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
        x(:,:,k+1:end) = []; %for now, delete ANY info above k
        %assign random p? randomly assign 0 or 1?
        
    end %end of trial by trial loop
end %end of block by block loop

nllh = -llh;

end

