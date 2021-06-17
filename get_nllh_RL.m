function [nllh] = get_nllh_RL(params)
% get negative log likelihood for param values
global onesubj
% fit alpha, beta, forget
alpha = params(1); %0 to 1, search for log alpha to make easier to search
beta = 100; %higher beta -> more greediness
epsilon = params(2);
forget = params(3); %0 to 1

data = onesubj; %global data variable for fmincon

na = sum(unique(data.resp)>0);
llh = 0; %initialize to 0
for b = 1:length(unique(data.block))
    stims = data.stim(data.block==b);
    ns = length(unique(stims));
    rewards = data.rew(data.block==b);
    key_vec = data.resp(data.block==b);
    cor_vec = data.cor(data.block==b);
    q = ones(ns,na)./na; %initialize with equal values at start of each block
    for trial = 1:sum(data.block==b) %ntrials
        stim = stims(trial); %which stimulus is it?
        % what action did they choose and what was the probability of it?
        a = key_vec(trial);
        p_softmax = epsilon/na + (1-epsilon)*(exp(q(stim,:).*beta)./sum(exp(q(stim,:).*beta))); %get probabilities 
        if a < 0 %trials where subjects did not respond
            % do nothing
        else %if they responded, get p(a)
            p_a = p_softmax(a);
            llh = llh + log(p_a);
        end
        if cor_vec(trial)
            %get reward from sequence
            rew = rewards(trial);
        else %incorrect answer, no reward
            rew = 0;
        end
        % learn!
        if a > 0 %subjects actually responded
            q(stim,a) = q(stim,a) + alpha*(rew-q(stim,a)); %RPE (r - q) times alpha  
        end
        q = q - forget*(q-(ones(ns,na)./na)); %decay to original values
    end %end of trial by trial loop
end %end of block by block loop

nllh = -llh;

end

