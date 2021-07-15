function [simdata] = simRL(params,data)
% simRL
% sim WMP data with RL model 
alpha = params(1); %0 to 1
beta = 100;% params(2); %higher beta -> more greediness
epsilon = params(2);
forget = params(3);

na = sum(unique(data.resp)>0);
rewards_sim = []; resp_sim = []; cor_sim = [];
for b = 1:length(unique(data.block))
    stims = data.stim(data.block==b); %stimuli
    ns = length(unique(stims));
    rewards = data.rew(data.block==b); %rewards presented to subjects
    key_vec = data.coract(data.block==b); %key, correct answers
    q = ones(ns,na)./na; %initialize with equal values at start of each block
    resp_vec = []; cor_vec = []; %for storing responses by block
    for trial = 1:sum(data.block==b) %ntrials
        stim = stims(trial); %which stimulus is it?
        % run softmax, choose action
        p_softmax = epsilon/na + (1-epsilon)*(exp(q(stim,:).*beta)./sum(exp(q(stim,:).*beta))); %get probabilities 
        resp = randsample(1:na,1,true,p_softmax);
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
        q(stim,resp) = q(stim,resp) + alpha*(r-q(stim,resp)); %RPE (r - q) times alpha 
        q = q - forget*(q-(ones(ns,na)./na)); %decay to original values
    end %end of trial by trial loop
    rewards_sim = [rewards_sim; rewards]; resp_sim = [resp_sim; resp_vec]; cor_sim = [cor_sim; cor_vec];
end %end of block by block loop

simdata = data; %everything else was the same from real data except responses/rewards
simdata.rew = rewards_sim; simdata.resp = resp_sim; simdata.cor = cor_sim;

end

