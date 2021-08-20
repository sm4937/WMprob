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
beta = 100; forget = 0; epsilon = 0; alphabino = 1; betabino = 1; 
% start with uninformative alphabino = betabino = 1;
if sum(contains(model,'beta'))
    beta = params(contains(model,'beta'));
end
if sum(contains(model,'forget'))
    forget = params(contains(model,'forget'));
end
if sum(contains(model,'epsilon'))
    epsilon = params(contains(model,'epsilon'));
end
if sum(contains(model,'a1_bino'))
    alpha1bino = params(contains(model,'a1_bino'));
end
if sum(contains(model,'b1_bino'))
    beta1bino = params(contains(model,'b1_bino'));
end
if sum(contains(model,'a0_bino'))
    alpha0bino = params(contains(model,'a0_bino'));
end
if sum(contains(model,'b0_bino'))
    beta0bino = params(contains(model,'b0_bino'));
end

K = sum(unique(data.resp)>0);
rewards_sim = []; resp_sim = []; cor_sim = [];
overalltrial = 0;

xs = 0:0.01:1;
p_r1 = betapdf(xs,alpha1bino,beta1bino); %start uninformative
p_r0 = betapdf(xs,alpha0bino,beta0bino); 

for b = 1:length(unique(data.block))
    stims = data.stim(data.block==b); %stimuli
    rewards = data.rew(data.block==b); %rewards presented to subjects
    key_vec = data.coract(data.block==b); %key, correct answers
    r1_true = unique(data.prew(data.block==b)); %probability of getting 1 given you chose correctly
    r0_true = 0; %probability of getting 1 given you chose incorrectly 
    ns = length(unique(stims));

    prior = ones(ns,K)./K; p = prior;
     %initialize with equal values at start of each block
    resp_vec = []; cor_vec = []; %for storing responses by block
    for trial = 1:sum(data.block==b) %ntrials
        overalltrial = overalltrial + 1;
        stim = stims(trial); %which stimulus is it?
        % run softmax, choose action
        p_softmax = epsilon/K + (1-epsilon)*(exp(p(stim,:).*beta)./sum(exp(p(stim,:).*beta))); %get probabilities 
        C_hat = randsample(1:K,1,true,p_softmax);
        resp_sim(overalltrial,:) = C_hat;
        
        %get reward from sequence
        rew = rewards(trial);
        if key_vec(trial) == C_hat %correct response, give pre-determined feedback
            R = rew;
            cor_vec(trial,:) = 1;
        else %they were wrong
            R = 0;
            cor_vec(trial,:) = 0;
        end
        rewards(trial) = R; %overwrite with rewards this simulated dude is getting
        rewards_sim(overalltrial,:) = R;
        cor_sim(overalltrial,:) = cor_vec(trial,:);
        % learn about p(r0) and p(r1) 
        % p(r1) = the probability of getting a 1 when correct
        % p(r0) = the probability of getting a 1 when incorrect
        % using beta distribution with alpha & beta parameters
        C_idx = false(overalltrial,1);
        C_idx(resp_sim(1:overalltrial,:) == C_hat) = true; 
        % grab trials where C_hat was chosen, to infer over p(r==1|correct,C_hat)
        % and p(r==0|correct,C_hat)
        % necessary for marginalization?
        
        n11 = sum(rewards_sim(1:overalltrial,:)&cor_sim(1:overalltrial,:)&C_idx);
        n10 = sum(~rewards_sim(1:overalltrial,:)&cor_sim(1:overalltrial,:)&C_idx);
        n01 = sum(rewards_sim(1:overalltrial,:)&~cor_sim(1:overalltrial,:)&C_idx);
        n00 = sum(~rewards_sim(1:overalltrial,:)&~cor_sim(1:overalltrial,:)&C_idx);
        %p_r0 = betapdf(0,alphabino+n01, betabino+n00) * betapdf(xs,alphabino+n01, betabino+n00);
        p_r0 = betapdf(xs,alpha0bino+n01, beta0bino+n00); p_r0 = p_r0./sum(p_r0);
        %p_r1 = betapdf(1,alphabino+n11, betabino+n10) * betapdf(xs,alphabino+n11, betabino+n10);
        p_r1 = betapdf(xs,alpha1bino+n11, beta1bino+n10); p_r1 = p_r1./sum(p_r1);
        % the first term of each is the probability of getting that based
        % on how many failures and successes you've had thus far
        % alpha biases prior towards success, beta towards failures
        % (getting a 0 in the 1 case, getting a 1 in the 0 case)
        
        [~,which] = max(p_r1); r1 = xs(which);
        [~,which] = max(p_r0); r0 = xs(which);
        r1 = sum(xs.*p_r1); r0 = sum(xs.*p_r0);
        if sum(isnan(p_r1))>0 | sum(isnan(p_r0))>0
            blah = true
        end
        
        % learn!
        %likelihood changes
        x = ones(size(p)); %starts out totally uninformative
        if R == 1
%             marginal = p_r1./(p_r1+p_r0); marginal = marginal(~isnan(marginal));
%             x(stim,resp) = prod(marginal);
            x(stim,C_hat) = r1./(r1+r0); %probability of being right given you got a 1
            notaction = [1 2 3]; notaction(C_hat) = [];
            x(stim,notaction) = r0./(r1+r0); %probability of being wrong given you got a 1
%             marginal = p_r0./(p_r1+p_r0); marginal = marginal(~isnan(marginal));
%             x(stim,resp) = prod(marginal);
        elseif R == 0
            x(stim,C_hat) = (1-r1)./(1-r0 + 1-r1);
%             marginal = (1-p_r1)./((1-p_r1)+(1-p_r0)); marginal = marginal(~isnan(marginal));
%             x(stim,resp) = prod(marginal);
            notaction = [1 2 3]; notaction(C_hat) = [];
            x(stim,notaction) = (1-r0)./(1-r0 + 1-r1); % probability being wrong given you got zero
%             marginal = (1-p_r1)./((1-p_r1)+(1-p_r0)); marginal = marginal(~isnan(marginal));
%             x(stim,resp) = prod(marginal);
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
    % rewards_sim = [rewards_sim; rewards]; 
    %resp_sim = [resp_sim; resp_vec];
    % cor_sim = [cor_sim; cor_vec];
end %end of block by block loop

simdata = data; %everything else was the same from real data except responses/rewards
simdata.rew = rewards_sim; simdata.resp = resp_sim; simdata.cor = cor_sim;

end

