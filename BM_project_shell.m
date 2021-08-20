%% Outer level script for Bayesian Modeling Final Project, Spring 2021
% Written by SLM in BK, April 2021
clear all
%close all
addpath(genpath('C:/Users/sarah/OneDrive/Documents/MATLAB/BayesianModeling/Project/'))

subs = [1 2 4:8 10:41 98 99]; %hard-coded to get subjects included in original analyses
excl = [4     8    10    24    25    30];
subs(excl)=[];
N = length(subs); 
% load data in to get responses, stimuli, block numbers, etc.

% for reference, data.Code is response
% data.Cor is the correctness of that response
% data.Rew is whether they got a +1 or a 0 
% data.seq is the stimulus they saw
% unique(data.seq) is the set size
% unique(data.Code) is the number of possible responses

%SDM_plot_learningcurves()

% prepare data for simulation or fitting
for s = 1:N
    % load data
    load(['Data/WMP_ID',num2str(subs(s))]);
    block_data = dataT;
    blocks = block_data{end}.blocks; % no "-1" for pnas/ejn, -1 for FMRI
    data{s} = [];
    for b = 1:length(blocks)
        tmp = table;
        tmp.resp = block_data{b}.Code';
        tmp.coract = block_data{b}.actionseq';
        tmp.cor = block_data{b}.Cor';
        tmp.stim = block_data{b}.seq';
        tmp.ns = repmat(blocks(b),length(block_data{b}.Cor),1);
        tmp.prew = repmat(matrice.prs(b),length(block_data{b}.Cor),1);
        tmp.block = repmat(b,length(block_data{b}.Cor),1);
        tmp.rew = matrice.prSeqs{b}';
        data{s} = [data{s}; tmp];
    end
    for n = 2:5
        nbackacc(s,n-1) = mean(dataNback(dataNback(:,1)==n,6)==1);
    end
end

% Use all of this to do a gen/rec with an RL model
global onesubj
RLparamnames = {'\alpha','\epsilon','forget'};
RLparams = [normrnd(0.3,0.1,N,1) normrnd(0.05,0.01,N,1) normrnd(0.1,0.05,N,1)];
RLparams(RLparams(:,3)<0,3) = 0; %forgetting should be small but NEVER negative
nparamsRL = size(RLparams,2);

for s = 1:N
    simdata{s} = simRL(RLparams(s,:),data{s});
    onesubj = simdata{s}; tries = []; nllhs_tries = [];
    for ii = 1:3 %use xx starting points per subject... it's pretty consistent tho
        inits = rand(1,nparamsRL); 
        [tries(ii,:),nllhs_tries(ii)] = fmincon(@get_nllh_RL,inits,[],[],[],[],[0 0 0],[1 1 1]);
    end
    [nllhs(s),which] = min(nllhs_tries);fitparams(s,:) = tries(which,:); %save best of all tries for each subject
end
    
% %plot generate/recover from RL model
%fitparams(:,2) = fitparams(:,2).*100;
figure
for p = 1:nparamsRL
    subplot(2,2,p)
    scatter(RLparams(:,p),fitparams(:,p),'Filled')
    hold on
    plot([0 1],[0 1],'k--')
    title(['RL param ' RLparamnames{p}])
    xlabel('Real param val')
    ylabel('Fit param val')
    ax = gca; ax.FontSize = 10;
end
fig = gcf; fig.Color = 'w';

%plot real subjects' learning curves
figure(2)
subplot(2,2,1)
SLM_plot_learningcurves(data)
title('Learning curve for real subjects')

% Plot learning curves for simulated RL subjects
subplot(2,2,2)
SLM_plot_learningcurves(simdata)
title('Learning curve for RL w/ forgetting')
fig = gcf; fig.Color = 'w';

% SAME AS ABOVE, but for BM NOW!
clear simdata fitparams tries nllhs_tries nllhs
% Use all of this to do a gen/rec with an BM model
maxk = 15; global k model; ks = zeros(N,1); %ks = ceil(rand(N,1)*maxk);

%specify which BM model to run
model = {'epsilon','forget','a1_bino','b1_bino','a0_bino','b0_bino'};
BMparamnames = {'\epsilon','forget','\alpha1_b','\beta1_b','\alpha0_b','\beta0_b'}; 

BMparams = [normrnd(0.1,0.05,N,1) normrnd(0.103,0.07,N,1) ...
    normrnd(2,0.25,N,1) normrnd(2,0.25,N,1) normrnd(2,0.25,N,1) ...
    normrnd(2,0.25,N,1)]; %epsilon, forget rate, then 4 
    % binomial prior parameters, alpha_bino & beta_bino for each 
    % inference condition (0 and 1)
    % alpha bino and beta bino should always be 1 or greater, to avoid issues
    % with beta function
BMparams(BMparams(:,2)<0,2) = 0.01; % don't let forget rate be negative
BMparams(BMparams(:,1)<0,1) = 0.01; % don't let epsilon be negative, either

nparamsBM = length(model);
lb = zeros(1,nparamsBM); ub = ones(1,nparamsBM);
lb(contains(model,{'a1_bino','b1_bino','a0_bino','b0_bino'})) = 0; ub(contains(model,{'a1_bino','b1_bino','a0_bino','b0_bino'})) = 5;
niters = 5; %5 seems to be where fitting curve stabilizes at a min
tries = cell(N,1); nllhs_tries = cell(N,1);

% find parameters which produce reasonable curves (separation between ns
% conditions) and then do gen/rec with those
fitflag = false;
if fitflag %don't run this whole thing unless you have to
    for s = 1:N
        k = ks(s);
        tries{s} = NaN(niters,nparamsBM); nllhs_tries{s} = NaN(niters,1); %save for simpler BM model
        simdata{s} = simBM(BMparams(s,:),data{s},model);
        onesubj = simdata{s};
        for ii = 1:niters
            inits = rand(1,nparamsBM); 
            [tries{s}(ii,:),nllhs_tries{s}(ii,:)] = fmincon(@get_nllh_BM,inits,[],[],[],[],lb,ub);
        end
        [nllhs(s),which] = min(nllhs_tries{s});fitparams(s,:) = tries{s}(which,:); %save best of all tries for each subject
    end
else
    load('BM_genrec.mat');
end

% Plot learning curves for simulated bayesian subjects
figure(2)
subplot(2,2,3)
SLM_plot_learningcurves(simdata)
title('Learning curve for BI')
fig = gcf; fig.Color = 'w';

figure
% %plot generate/recover from BM model
fitparams(:,contains(model,'bino')) = exp(fitparams(:,contains(model,'bino')));
for p = 1:nparamsBM
    subplot(3,3,p)
    scatter(BMparams(:,p),fitparams(:,p),'Filled')
    hold on
    plot([0 1],[0 1],'k--')
    title(['BI param ' BMparamnames{p}])
    xlabel('Real param val')
    ylabel('Fit param val')
    ax = gca; ax.FontSize = 10;
end

% % NOW THE SAME FOR THE ORIGINAL RLWM MODEL % % 
model = {'epsilon','alpha','rho3','rho6','forget','beta'};
RLWMparamnames = {'\epsilon','\alpha','\rho_3','\rho_6','forget','\beta'};
RLWMparams = [normrnd(0.05,0.01,N,1) normrnd(0.1,0.25,N,1) normrnd(0.5,0.1,N,1) normrnd(0.3,0.1,N,1) normrnd(0.1,0.25,N,1) normrnd(1,0.25,N,1)];
RLWMparams(RLWMparams(:,1)<0,1) = 0; %epsilon should be small but NEVER negative
RLWMparams(RLWMparams(:,2)<0,2) = 0.01; %alpha should also stay positive
RLWMparams(RLWMparams(:,5)<0,5) = 0; %same w forget rate (here, just for WM)
nparamsRLWM = size(RLWMparams,2); lb = zeros(1,nparamsRLWM); ub = ones(1,nparamsRLWM);
lb(contains(model,'beta')) = -Inf; ub(contains(model,'beta')) = Inf;

clear nllhs fitparams
for s = 1:N
    simdataRLWM{s} = simRLWM(RLWMparams(s,:),data{s},model);
    onesubj = simdataRLWM{s}; tries = []; nllhs_tries = [];
    for ii = 1:niters+10 %use xx starting points per subject... it's pretty consistent tho
        inits = rand(1,nparamsRLWM); 
        [tries(ii,:),nllhs_tries(ii)] = fmincon(@get_nllh_RLWM,inits,[],[],[],[],lb,ub);
    end
    [nllhs(s),which] = min(nllhs_tries);fitparams_RLWM(s,:) = tries(which,:); %save best of all tries for each subject
end

figure(2)
subplot(2,2,4)
SLM_plot_learningcurves(simdataRLWM)
title('Learning curve for RLWM')
fig = gcf; fig.Color = 'w';

figure
for p = 1:nparamsRLWM
    subplot(3,2,p)
    scatter(RLWMparams(:,p),fitparams_RLWM(:,p),'Filled')
    hold on
    plot([0 1],[0 1],'k--')
    title(['RLWM param ' RLWMparamnames{p}])
    xlabel('Real param val')
    ylabel('Fit param val')
    ax = gca; ax.FontSize = 10;
end
fig = gcf; fig.Color = 'w';

%% Fit real data & see how the numbers come out
niters = 5;
clear tries nllhs
k = 0;

tries_RL = cell(N,1); nllhs_tries_RL = cell(N,1);
tries_BM = cell(N,1); nllhs_tries_BM = cell(N,1);
tries_RLWM = cell(N,1); nllhs_tries_RLWM = cell(N,1);

fitflag = true;
if fitflag %want to run? it'll take forever!
    for s = 1:N
        onesubj = data{s}; tries_RL{s} = NaN(niters,nparamsRL); nllhs_tries_RL{s} = NaN(niters,1); 
        for ii = 1:niters %use xx starting points per subject... it's pretty consistent tho
            inits = rand(1,nparamsRL); 
            [tries_RL{s}(ii,:),nllhs_tries_RL{s}(ii,:)] = fmincon(@get_nllh_RL,inits,[],[],[],[],zeros(1,nparamsRL),ones(1,nparamsRL));
        end %iterate over RL values
        model = {'epsilon','forget','a1_bino','b1_bino','a0_bino','b0_bino'};
        tries_BM{s} = NaN(niters,nparamsBM); nllhs_tries_BM{s} = NaN(niters,1);
        for ii = 1:niters-2 %this takes a long time, so shorten a little
            inits = rand(1,nparamsBM); 
            [tries_BM{s}(ii,:),nllhs_tries_BM{s}(ii,:)] = fmincon(@get_nllh_BM,inits,[],[],[],[],zeros(1,nparamsBM),ones(1,nparamsBM));
        end
        model = {'epsilon','alpha','rho3','rho6','forget'};
        tries_RLWM{s} = NaN(niters,nparamsRLWM); nllhs_tries_RLWM{s} = NaN(niters,1);
        for ii = 1:niters+10 %takes a while to search entire param space
            inits = rand(1,nparamsRLWM); 
            [tries_RLWM{s}(ii,:),nllhs_tries_RLWM{s}(ii,:)] = fmincon(@get_nllh_RLWM,inits,[],[],[],[],zeros(1,nparamsRLWM),ones(1,nparamsRLWM));
        end
        [nllhs_BM(s,:),which] = min(nllhs_tries_BM{s}); fitparams_BM(s,:) = tries_BM{s}(which,:); 
        [nllhs_RL(s,:),which] = min(nllhs_tries_RL{s}); fitparams_RL(s,:) = tries_RL{s}(which,:); %save best of all tries for each subject
        [nllhs_RLWM(s,:),which] = min(nllhs_tries_RLWM{s}); fitparams_RLWM(s,:) = tries_RLWM{s}(which,:); %save best of all tries for each subject    
    end
else
    load('realfits.mat')
    %load('realfits_v02.mat')
end
s = 35;
ntrials = length(data{s}.resp);
AICs_subjs = [2*nllhs_RL + 2*nparamsRL 2*nllhs_BM + 2*nparamsBM 2*nllhs_RLWM + 2*nparamsRLWM];
BICs_subjs = [2*nllhs_RL + 2*((log(ntrials)/nparamsRL)*2) 2*nllhs_BM + 2*((log(ntrials)/nparamsBM)*2) 2*nllhs_RLWM + 2*((log(ntrials)/nparamsRLWM)*2)];

AICs = mean(AICs_subjs);
BICs = mean(BICs_subjs);

figure
subplot(2,1,1)
bar(AICs-min(AICs))
xticklabels({'RL fit','BI fit','RLWM fit'})
ylabel('AIC - best AIC')
title('Relative mean AIC score by model')
ax = gca; ax.FontSize = 10;

subplot(2,1,2)
[vals,bestfits] = min(AICs_subjs,[],2);
bar([sum(bestfits==1) sum(bestfits==2) sum(bestfits==3)])
ylabel('N subjs w lower AIC score')
title('N subjects best fit by model')
xticklabels({'RL','BI','RLWM'})
fig = gcf; fig.Color = 'w';

model = {'epsilon','forget','a1_bino','b1_bino','a0_bino','b0_bino'};
%fitparams_BM(:,contains(model,{'bino'})) = exp(fitparams_BM(:,contains(model,{'bino'})));

% % Re-plot learning curves based on fit parameter values
for s = 1:N
    simdataRL{s} = simRL(fitparams_RL(s,:),data{s});
    model = {'epsilon','forget','a1_bino','b1_bino','a0_bino','b0_bino'};
    simdataBM{s} = simBM(fitparams_BM(s,:),data{s},model); %simulate with fit params
    model = {'epsilon','alpha','rho3','rho6','forget'};
    simdataRLWM{s} = simRLWM(fitparams_RLWM(s,:),data{s},model);
end

% Plot learning curves for model validation 
figure
subplot(2,2,1)
SLM_plot_learningcurves(data)
title('Real learning curves (human data)')
ylabel('p(correct)')
ax = gca; ax.FontSize = 10;

subplot(2,2,2)
SLM_plot_learningcurves(simdataRL)
title('Sim learning curves (RL model)')
ylabel('p(correct)')
ax = gca; ax.FontSize = 10;

subplot(2,2,3)
SLM_plot_learningcurves(simdataBM)
title('Sim learning curves (BI model)')
xlabel('Stimulus iterations')
ylabel('p(correct)')
fig = gcf; fig.Color = 'w';
ax = gca; ax.FontSize = 10;

subplot(2,2,4)
SLM_plot_learningcurves(simdataRLWM)
title('Sim learning curves (RLWM model)')
xlabel('Stimulus iterations')
ylabel('p(correct)')
fig = gcf; fig.Color = 'w';
ax = gca; ax.FontSize = 10;
