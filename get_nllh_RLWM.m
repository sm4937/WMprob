function [nllh] = get_nllh_RLWM(params)
% fit data with the RLWM model
% optimized for probabilistic RL task, as briefly described in McDougle &
% Collins 2020

global onesubj model
llh = 0; %initialize at zero
data = onesubj;

%default parameter values
beta = 100; forget = 0; epsilon = 0; alpha = 0; rho3 = 0.5; rho6 = 0.5;
if sum(contains(model,'beta'))
    beta = params(contains(model,'beta'))*100;
    % shared across RL & WM, free or fixed (most likely this will be fixed)
end
if sum(contains(model,'alpha'))
    alpha = params(contains(model,'alpha'));
    % RL only, learning rate for RPE-dependent q update
end
if sum(contains(model,'forget'))
    forget = params(contains(model,'forget'));
    %forget is just for WM for now, though could easily also be applied to
    %RL
end
if sum(contains(model,'epsilon'))
    epsilon = params(contains(model,'epsilon'));
    %epsilon shared across RL and WM modules
end
if sum(contains(model,'rho3'))
    rho3 = params(contains(model,'rho3'));
end
if sum(contains(model,'rho6'))
    rho6 = params(contains(model,'rho6'));
end

K = sum(unique(data.resp)>0);
for b = 1:length(unique(data.block))
    stims = data.stim(data.block==b); %stimuli
    ns = length(unique(stims)); %set size this block (3 or 6)
    eval(['weight = rho' num2str(ns) ';']);
    rewards = data.rew(data.block==b); %rewards presented to subjects
    resp_vec = data.resp(data.block==b);
    q_RL = ones(ns,K)./K; %initialize with equal values at start of each block
    q_WM = q_RL;
    for trial = 1:sum(data.block==b) %ntrials
        stim = stims(trial); %which stimulus is it?
        % run softmax
        p_RL = (exp(q_RL(stim,:).*beta)./sum(exp(q_RL(stim,:).*beta))); %get probabilities 
        p_WM = (exp(q_WM(stim,:).*beta)./sum(exp(q_WM(stim,:).*beta))); %get probabilities 
        % probability of each response under RL and WM learning
        p = weight*p_WM + (1-weight)*p_RL; %tradeoff between probabilities from WM and RL, based on set-size-dependent weights
        p = epsilon/K + (1-epsilon)*p;
        % turn into a response:
        resp = resp_vec(trial,:);
        if resp > 0
            llh = llh + log(p(resp)); 
        
            %get reward from sequence
            r = rewards(trial);

            % learn!
            q_RL(stim,resp) = q_RL(stim,resp) + alpha*(r-q_RL(stim,resp)); %RPE (r - q) times alpha 
            q_WM(stim,resp) = q_WM(stim,resp) + 1*(r-q_WM(stim,resp)); %in WM, the learning is instant
        end
        
        q_WM = q_WM - forget*(q_WM-(ones(ns,K)./K)); %decay to original values
        
    end %end of trial by trial loop
end %end of block by block loop

nllh = -llh;

end

