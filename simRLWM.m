function [simdata] = simRLWM(params,data,model)
% simulate fake data with the RLWM model
% optimized for probabilistic RL task, briefly described in McDougle &
% Collins 2020

% TO DO : Ask Anne whether this should be included, or whether to keep it
% simple with rho3 and rho6 as free params
%             global k 
%             w = rho * min(1,k/ns(b)); % inital weighting of WM

%default parameter values
beta = 100; forget = 0; epsilon = 0; alpha = 0; rho3 = 0.5; rho6 = 0.5;
if sum(contains(model,'beta'))
    beta = params(contains(model,'beta'));
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
rewards_sim = []; resp_sim = []; cor_sim = [];
for b = 1:length(unique(data.block))
    stims = data.stim(data.block==b); %stimuli
    ns = length(unique(stims)); %set size this block (3 or 6)
    eval(['weight = rho' num2str(ns) ';']);
    rewards = data.rew(data.block==b); %rewards presented to subjects
    key_vec = data.coract(data.block==b); %key, correct answers
    q_RL = ones(ns,K)./K; %initialize with equal values at start of each block
    q_WM = q_RL;
    resp_vec = []; cor_vec = []; %for storing responses by block
    for trial = 1:sum(data.block==b) %ntrials
        stim = stims(trial); %which stimulus is it?
        % run softmax
        p_RL = (exp(q_RL(stim,:).*beta)./sum(exp(q_RL(stim,:).*beta))); %get probabilities 
        p_WM = (exp(q_WM(stim,:).*beta)./sum(exp(q_WM(stim,:).*beta))); %get probabilities 
        % probability of each response under RL and WM learning
        p = weight*p_WM + (1-weight)*p_RL; %tradeoff between probabilities from WM and RL, based on set-size-dependent weights
        p = epsilon/K + (1-epsilon)*p;
        % turn into a response:
        resp = randsample(1:K,1,true,p);
        resp_vec(trial,:) = resp; %catalogue response for later
        
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
        q_RL(stim,resp) = q_RL(stim,resp) + alpha*(r-q_RL(stim,resp)); %RPE (r - q) times alpha 
        q_WM(stim,resp) = q_WM(stim,resp) + 1*(r-q_WM(stim,resp)); %in WM, the learning is instant
        
        q_WM = q_WM - forget*(q_WM-(ones(ns,K)./K)); %decay to original values
        
    end %end of trial by trial loop
    rewards_sim = [rewards_sim; rewards]; resp_sim = [resp_sim; resp_vec]; cor_sim = [cor_sim; cor_vec];
end %end of block by block loop

simdata = data; %everything else was the same from real data except responses/rewards
simdata.rew = rewards_sim; simdata.resp = resp_sim; simdata.cor = cor_sim;

end

