function [nllh] = get_nllh_BM(params)
% fit WMP data with Bayesian Model (BI) 
% Where x is all the reward information you have for that stimulus
llh = 0;
%p(C|x) = U_C*p(C)*p(x|C)
%p(C|x_vec) = U_C*p(C)*p(x_1|C)*p(x_2|C)*...*p(x_t|C)
%uninformative prior, re-update posterior every time, don't just learn

global k onesubj model
data = onesubj; 
beta = 100; forget = 0; epsilon = 0; alphabino = 1; betabino = 1;
if sum(contains(model,'beta'))
    logbeta = params(contains(model,'beta'));
    beta = exp(logbeta);
end
if sum(contains(model,'forget'))
    forget = params(contains(model,'forget'));
end
if sum(contains(model,'epsilon'))
    epsilon = params(contains(model,'epsilon'));
end
if sum(contains(model,'a1_bino'))
    alpha1bino = exp(params(contains(model,'a1_bino')));
    % fitting log(alphabino)
end
if sum(contains(model,'b1_bino'))
    beta1bino = exp(params(contains(model,'b1_bino')));
    % fitting log(beta1bino)
end
if sum(contains(model,'a0_bino'))
    alpha0bino = exp(params(contains(model,'a0_bino')));
    % fitting log(alphabino)
end
if sum(contains(model,'b0_bino'))
    beta0bino = exp(params(contains(model,'b0_bino')));
    % fitting log(betabino)
end

overalltrial = 0;
xs = 0:0.01:1;
p_r1 = betapdf(xs,alpha1bino,beta1bino); %start uninformative
p_r0 = betapdf(xs,alpha0bino,beta0bino); 

K = sum(unique(data.resp)>0);
for b = 1:length(unique(data.block))
    stims = data.stim(data.block==b); %stimuli
    rewards = data.rew(data.block==b); %rewards presented to subjects
    resp_vec = data.resp(data.block==b);
    p_reward_true = unique(data.prew(data.block==b));
    p_0 = (1-p_reward_true)./(1-p_reward_true + 1); %probability of getting 0 given you chose correctly is probability 
    % you chose correctly and got 0 (1-preward), normalized with probability you chose
    % incorrectly and got 0 (1)
    ns = length(unique(stims));

    prior = ones(ns,K)./K; p = prior;
    for trial = 1:sum(data.block==b) %ntrials
        overalltrial = overalltrial + 1; 
        stim = stims(trial); %which stimulus is it?
        % run softmax, choose action
        p_softmax = epsilon/K + (1-epsilon)*(exp(p(stim,:).*beta)./sum(exp(p(stim,:).*beta))); %get probabilities 
        
        C_hat = resp_vec(trial);
        x = ones(size(p)); %likelihood starts out totally uninformative
        if C_hat > 0
            p_a = p_softmax(C_hat);
            llh = llh + log(p_a);
            %get reward from sequence
            R = rewards(trial);

            % update learned probabilities of getting reward given
            % correctness/incorrectness
            C_idx = false(overalltrial,1);
            C_idx(data.resp(1:overalltrial,:) == C_hat) = true; 
            % grab trials where C_hat was chosen, to infer over p(r==1|correct,C_hat)
            % and p(r==0|correct,C_hat)

            n11 = sum(data.rew(1:overalltrial,:)&data.cor(1:overalltrial,:)&C_idx);
            n10 = sum(~data.rew(1:overalltrial,:)&data.cor(1:overalltrial,:)&C_idx);
            n01 = sum(data.rew(1:overalltrial,:)&~data.cor(1:overalltrial,:)&C_idx);
            n00 = sum(~data.rew(1:overalltrial,:)&~data.cor(1:overalltrial,:)&C_idx);
            %p_r0 = betapdf(0,alphabino+n01, betabino+n00) * betapdf(xs,alphabino+n01, betabino+n00);
            p_r0 = betapdf(xs,alpha0bino+n01, beta0bino+n00); p_r0 = p_r0./nansum(p_r0);
            %p_r1 = betapdf(1,alphabino+n11, betabino+n10) * betapdf(xs,alphabino+n11, betabino+n10);
            p_r1 = betapdf(xs,alpha1bino+n11, beta1bino+n10); p_r1 = p_r1./nansum(p_r1);
            % the first term of each is the probability of getting that based
            % on how many failures and successes you've had thus far
            % alpha biases prior towards success, beta towards failures
            % (getting a 0 in the 1 case, getting a 1 in the 0 case)

%             [~,which] = max(p_r1); r1 = xs(which);
%             [~,which] = max(p_r0); r0 = xs(which);
            r1 = sum(xs.*p_r1); r0 = sum(xs.*p_r0);
            if sum(isnan(p_r1))>0 | sum(isnan(p_r0))>0
                blah = true
            end
            
            % learn!
            %likelihood changes
            if R == 1
                %x(stim,resp) = 1 * r1_true;
                %x(stim,notaction) = 0 * r1_true;
                x(stim,C_hat) = r1./(r1+r0); %probability of being right given you got a 1
                notaction = [1 2 3]; notaction(C_hat) = [];
                x(stim,notaction) = r0./(r1+r0); %probability of being wrong given you got a 1
            elseif R == 0
                x(stim,C_hat) = (1-r1)./(1-r0 + 1-r1);
                notaction = [1 2 3]; notaction(C_hat) = [];
                x(stim,notaction) = (1-r0)./(1-r0 + 1-r1); % probability being wrong given you got zero
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

