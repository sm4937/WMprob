function [nllh] = get_nllh_BM(params)
% fit WMP data with Bayesian Model (BI) 
% Where x is all the reward information you have for that stimulus
llh = 0;
%p(C|x) = U_C*p(C)*p(x|C)
%p(C|x_vec) = U_C*p(C)*p(x_1|C)*p(x_2|C)*...*p(x_t|C)
%uninformative prior, re-update posterior every time, don't just learn

global k onesubj model
data = onesubj; 
beta = 100; forget = 0; epsilon = 0;
if sum(contains(model,'beta'))
    beta = params(contains(model,'beta'))*100;
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
    cor_vec = data.cor(data.block==b);
    resp_vec = data.resp(data.block==b);
    p_reward = unique(data.prew(data.block==b));
    p_0 = (1-p_reward)./(1-p_reward + 1); %probability of getting 0 given you chose correctly is probability 
    % you chose correctly and got 0 (1-preward), normalized with probability you chose
    % incorrectly and got 0 (1)
    ns = length(unique(stims));

    prior = ones(ns,K)./K; p = prior;
    for trial = 1:sum(data.block==b) %ntrials
        stim = stims(trial); %which stimulus is it?
        % run softmax, choose action
        p_softmax = epsilon/K + (1-epsilon)*(exp(p(stim,:).*beta)./sum(exp(p(stim,:).*beta))); %get probabilities 
        
        resp = resp_vec(trial);
        if resp > 0
            p_a = p_softmax(resp);
            llh = llh + log(p_a);
        end
        %get reward from sequence
        r = rewards(trial);
        % learn!
        %likelihood changes
        x = ones(size(p)); %starts out totally uninformative
        if resp > 0 %non-response trials, don't do any inference
            if r == 1
                x(stim,resp) = 1;
                notaction = [1 2 3]; notaction(resp) = [];
                x(stim,notaction) = 0;
            elseif r == 0
                x(stim,resp,1) = p_0; %probably of not getting rewarded this block
                notaction = [1 2 3]; notaction(resp) = [];
                x(stim,notaction) = (1-p_0)/2;
            end
        end
        
        if k > 0 
            % given data in x so far, update p of each category for each
            % stimulus
            p = ones(ns,K)./K; %initialize with flat prior before inference for each trial
            for tt = 1:size(x,3) %loop over REMEMBERED trials
                p = p.*x(:,:,tt); %update w likelihood of x if category
                p = p./sum(p,2); %normalize rows
            end
            % posterior gets completely re-updated every trial, since some info
            % gets corrupted
        else %update posterior from previous posterior
            p = p.*x; %update w new likelihood
            p = p./sum(p,2); %normalize rows
        end
        %decay to priors w forget parameter
        p = (1-forget)*p + forget*prior;
        
        % wipe any information above capacity limit k
        if k > 0; x(:,:,k+1:end) = []; end
        
    end %end of trial by trial loop
end %end of block by block loop

nllh = -llh;

end

