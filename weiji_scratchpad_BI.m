% P needs to be marginal over C, r1, r0
% meshgrid(C,r1,r0)
% C (rows) x r1 (cols) x r0 (3rd-d)
% Normalize over everything so entire total is 1
% Loop over C's, given that C_hat was your response
% was it correct or incorrect?

% Define 3d posterior (over C, r1, and r0)
% (use meshgrid)
% posterior(Cind, r1ind, r0ind)
% initially: 

posterior_3d = prior_C(C_meshed) .* prior_r1(r1_meshed) .* prior_r0(r0_meshed)

% Now for the iterative part:

for trial = 1:ntrials
    
    % Marginalize to get posterior over C:
    posterior_C = sum(sum(posterior_3d, 3),2);
    % can check this against equation in overleaf
    
    % Evaluate at presented stimulus:
    stim_trial = stimuli(trial)
    C_hat = softmax(posterior_C(stim_trial));
    
    % Simulate reward based on true C
    R = ..
    
    % based on this C_hat, calculate likelihood
    temp = posterior_3d;
    for Cind = 1:length(Cvec)
        C = Cvec(Cind);
        
        if C(stim_trial) == C_hat & R == 1 % correct AND reward received
            likelihood = r1_meshed; % 2d mesh
        elseif C(stim_trial) ~= C_hat & R == 1
            likelihood = r0_meshed; % 2d mesh
            etc.
        end 
            temp(C, :, :) = temp(C,:,:) .* likelihood
    end
     
    % Normalization
    posterior_3d = temp/sum(temp(:));
end