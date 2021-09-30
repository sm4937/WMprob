function [nllh] = get_nllh_BM_v03(params)

%simBM_v03
% recover a Bayesian Inference model,
% a full 3-d inference process over p(C|r0,r1,I) (classes of each stimulus)
% p(R=1|Correct,C,I), and p(R=1|Incorrect,C,I)
global onesubj model
data = onesubj;
llh = 0;

beta_temp = 100; forget = 0; epsilon = 0; 
alpha0bino = 1; beta0bino = 1; 
alpha1bino = 1; beta1bino = 1; 
% start with uninformative alphabino = betabino = 1;
if sum(contains(model,'beta'))
    beta_temp = exp(params(contains(model,'beta')));
    % fit log(beta_temp)
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
end
if sum(contains(model,'a0_bino'))
    alpha0bino = exp(params(contains(model,'a0_bino')));
end
if sum(contains(model,'b0_bino'))
    beta0bino = exp(params(contains(model,'b0_bino')));
end

K = sum(unique(data.resp)>0); %in other code, this variable is called na

r1 = 0:0.01:1; 
r0 = 0:0.02:1;
r0 = reshape(r0,1,1,length(r0));
prior_r1 = betapdf(r1,alpha1bino,beta1bino) * beta(alpha1bino,beta1bino); %row vector
prior_r0 = betapdf(r0,alpha0bino,beta0bino) * beta(alpha0bino,beta0bino);
% down the 3rd dimension

overalltrial = 0; %initialize trial counter
for b = 1:length(unique(data.block))
    stims = data.stim(data.block==b); %stimuli
        
    ns = length(unique(stims));
    % Set up full space of C (thousands of possibilities)
    allC = unique(nchoosek(repmat(1:K,1,ns),ns),'rows');
    numC = size(allC,1);
    
    % Generate a flat prior over C (category for each stimulus)
    prior_C = ones(numC, 1)/numC;
    % 1st dimension, column vector
    prior_3d = prior_C .* prior_r1 .* prior_r0; prior_3d = prior_3d./sum(prior_3d(:));
    %prior_3d = prior_C(C_meshed) .* prior_r1(r1_meshed) .* prior_r0(r0_meshed)
    posterior_3d = prior_3d;

    ntrials = sum(data.block==b);
    for t = 1:ntrials
        overalltrial = overalltrial + 1;
        % Increment trial count
        i_t = stims(t);  % index of stimulus, between 1 and N

        % Decision
        posterior_C = sum(sum(posterior_3d,3),2);
        marginal = NaN(1,K);
        for j = 1:K % loops over possible choices
            marginal(j) = sum(posterior_C(allC(:,i_t)==j));
        end

        p_resp = marginal.^beta_temp;
        p_resp = epsilon/K + (1-epsilon)*(p_resp/sum(p_resp));
        C_hat = data.resp(overalltrial);
        llh = llh + log(p_resp(C_hat));

        % Reward
        R = data.rew(overalltrial);
        
        % Get likelihood of that outcome, conditioned on specific
        % hypotheses over C
        temp = posterior_3d;
        for hyp = 1:size(posterior_3d,1)
            % cycle over all possible C hypotheses
            C = allC(hyp,:);
            C_true_hyp = C(i_t);

            if C_true_hyp == C_hat & R == 1
                %if you're correct, and you get rewarded
                likelihood = r1; %r1_meshed; % gotta be 2d
            elseif C_true_hyp ~= C_hat & R ==1
                likelihood = r0;
            elseif C_true_hyp == C_hat & R == 0
                likelihood = 1-r1;
            elseif C_true_hyp ~= C_hat & R == 0
                likelihood = 1-r0;
            end

            temp(hyp,:,:) = temp(hyp,:,:) .* likelihood;
            
        end
        
        % Updating the posterior
        posterior_3d = temp;
        posterior_3d = posterior_3d/sum(posterior_3d(:));
        
        % Forgetting
        posterior_3d = (1-forget)* posterior_3d + forget*prior_3d;
        
    end % of each trial
    
end % of each block

nllh = -llh;

end
