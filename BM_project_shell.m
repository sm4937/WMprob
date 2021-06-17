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

%for reference, data.Code is response
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
figure
subplot(1,3,1)
SLM_plot_learningcurves(data)
title('Learning curve for real subjects')

% Plot learning curves for simulated RL subjects
subplot(1,3,2)
SLM_plot_learningcurves(simdata)
title('Learning curve for RL w/ forgetting')
fig = gcf; fig.Color = 'w';

% SAME AS ABOVE, but for BM NOW!
fitflag = false;
clear simdata fitparams
% Use all of this to do a gen/rec with an BM model
% global k
maxk = 15; ks = ceil(rand(N,1)*maxk); 
BMparams = [normrnd(3,0.5,N,1) normrnd(0.3,0.1,N,1)]; %beta, then k
BMparams(BMparams(:,2)<0,2) = 0; % don't let forget rate be negative, ever
BMparamnames = {'\beta','forget','k'}; 
nparamsBM = size(BMparams,2);
if fitflag %don't run this whole thing unless you have to
    for s = 1:N
        k = ks(s);
        simdata{s} = simBM_v02(BMparams(s,:),data{s}); %ks(s);
        onesubj = simdata{s};
        inits = rand(1,nparamsBM); tries = []; nllhs_tries = []; summary_k_fits = [];
        %for k = 1:maxk
            for ii = 1:20 %use xx starting points per subject... it's not very consistent for BM, compared to RL
                inits = rand(1,nparamsBM); lb = zeros(1,nparamsBM); ub = ones(1,nparamsBM);
                %[tries(ii,:,k),nllhs_tries(ii,k)] = fmincon(@get_nllh_BM_v02,inits,[],[],[],[],lb,ub);
                [tries(ii,:),nllhs_tries(ii,:)] = fmincon(@get_nllh_BM_v02,inits,[],[],[],[],lb,ub);
            end
        %end
%         summary_k_fits = min(nllhs_tries,[],1); % which k gave the best score, overall? 
%         [~,fitk(s)] = min(summary_k_fits);
%         nllhs_tries = nllhs_tries(:,fitk(s)); tries = tries(:,:,fitk(s));
        [nllhs(s),which] = min(nllhs_tries);fitparams(s,:) = tries(which,:); %save best of all tries for each subject
    end
else
    %load('BM_genrec.mat');
    load('BM_genrec_v02.mat')
end
% Plot learning curves for simulated bayesian subjects
subplot(1,3,3)
SLM_plot_learningcurves(simdata)
title('Learning curve for BI')
fig = gcf; fig.Color = 'w';

figure
% %plot generate/recover from BM model
fitparams(:,1) = fitparams(:,1)*100;
for p = 1:nparamsBM
    subplot(1,2,p)
    scatter(BMparams(:,p),fitparams(:,p),'Filled')
    hold on
    plot([0 1],[0 1],'k--')
    title(['BI param ' BMparamnames{p}])
    xlabel('Real param val')
    ylabel('Fit param val')
    ax = gca; ax.FontSize = 10;
end

% subplot(2,2,p+1)
% scatter(ks,fitk)
% xlabel('Real K value')
% ylabel('Fit K value')
% fig = gcf; fig.Color = 'w';

%% Fit real data & see how the numbers come out
niters = 20;
clear tries nllhs summary_k_fits simdata fitks ks
fitflag = false;
if fitflag %want to run? it'll take forever!
    for s = 1:N
        onesubj = data{s}; tries_RL = []; nllhs_tries_RL = []; tries_BM = []; nllhs_tries_BM = [];
        for ii = 1:niters-4 %use xx starting points per subject... it's pretty consistent tho
            inits = rand(1,nparamsRL); 
            [tries_RL(ii,:),nllhs_tries_RL(ii)] = fmincon(@get_nllh_RL,inits,[],[],[],[],zeros(1,nparamsRL),ones(1,nparamsRL));
        end %iterate over RL values
%         summary_k_fits = [];
%         for k = 1:maxk
            for ii = 1:niters %use xx starting points per subject... it's not very consistent for BM, compared to RL
                inits = rand(1,nparamsBM); 
                %[tries(ii,k),nllhs_tries(ii,k)] = fmincon(@get_nllh_BM,inits,[],[],[],[],zeros(1,nparamsBM),ones(1,nparamsBM));
                [tries_BM(ii,:),nllhs_tries_BM(ii,:)] = fmincon(@get_nllh_BM_v02,inits,[],[],[],[],zeros(1,nparamsBM),ones(1,nparamsBM));
            end
%         end
%         summary_k_fits = min(nllhs_tries,[],1); % which k gave the best score, overall? 
%         [~,fitk(s)] = min(summary_k_fits);
%         nllhs_tries = nllhs_tries(:,fitk(s)); tries_BM = tries(:,fitk(s));
        [nllhs_BM(s,:),which] = min(nllhs_tries_RL); fitparams_BM(s,:) = tries_BM(which,:); 
        [nllhs_RL(s,:),which] = min(nllhs_tries_RL); fitparams_RL(s,:) = tries_RL(which,:); %save best of all tries for each subject
    end
else
    %load('realfits.mat')
    load('realfits_v02.mat')
end
s = 35;
ntrials = length(data{s}.resp);
AICs_subjs = [2*nllhs_RL + 2*nparamsRL 2*nllhs_BM + 2*nparamsBM];
BICs_subjs = [2*nllhs_RL + 2*((log(ntrials)/nparamsRL)*2) 2*nllhs_BM + 2*((log(ntrials)/nparamsBM)*2)];

AICs = mean(AICs_subjs);
BICs = mean(BICs_subjs);

figure
subplot(2,1,1)
bar(AICs-min(AICs))
xticklabels({'RL fit','BI fit'})
ylabel('AIC - best AIC')
title('Relative mean AIC score by model')
ax = gca; ax.FontSize = 10;

% subplot(3,1,2)
% bar(BICs-min(BICs))
% xticklabels({'RL fit','BI fit'})
% ylabel('relative BIC score')
% ax = gca; ax.FontSize = 10;
% fig = gcf; fig.Color = 'w';

subplot(2,1,2)
[vals,bestfits] = min(AICs_subjs,[],2);
bar([sum(bestfits==1) sum(bestfits==2)])
ylabel('N subjs w lower AIC score')
title('N subjects best fit by model')
xticklabels({'RL','BI'})
fig = gcf; fig.Color = 'w';

% % Re-plot learning curves based on fit parameter values
fitparams_BM(:,1) = fitparams_BM(:,1)*100;
for s = 1:N
    simdataRL{s} = simRL(fitparams_RL(s,:),data{s});
%    k = fitk(s);
    simdataBM{s} = simBM(fitparams_BM(s,:),data{s}); %simulate with fit params
end

figure
subplot(3,1,1)
SLM_plot_learningcurves(data)
title('Real learning curves (human data)')
ylabel('p(correct)')
ax = gca; ax.FontSize = 10;

subplot(3,1,2)
SLM_plot_learningcurves(simdataRL)
title('Sim learning curves (RL model)')
ylabel('p(correct)')
ax = gca; ax.FontSize = 10;

subplot(3,1,3)
SLM_plot_learningcurves(simdataBM)
title('Sim learning curves (BI model)')
xlabel('Stimulus iterations')
ylabel('p(correct)')
fig = gcf; fig.Color = 'w';
ax = gca; ax.FontSize = 10;

for n = 1:4
    [r,p] = corr(nbackacc(:,n),fitk'); % does fit capacity correlate with accuracy on the n-back?
    disp(['corr ' num2str(n+1) '-back acc & fit K parameter: r = ' num2str(r) ', p = ' num2str(p)])
end
nseffect = nbackacc(:,1)-nbackacc(:,3);
[r,p] = corr(nseffect,fitk'); % does fit capacity correlate with accuracy on the n-back?
disp(['corr set size effect & fit K parameter: r = ' num2str(r) ', p = ' num2str(p)])

% %% OBSOLETE! Run a little thing over utility of each action (selecting each class)
% 
% % one stim, 3 actions, one is correct and gives reward p_rew percent of the
% % time
% means = 33.*ones(1,3);
% sigma_s = 100;
% sigma = 50;
% na = 3;
% us = -300:300;
% p_rew = 0.75;
% 
% %prior distributions of utility over each action
% action1 = normpdf(us,means(1),sigma_s);
% action2 = normpdf(us,means(2),sigma_s);
% action3 = normpdf(us,means(3),sigma_s);
% 
% figure
% for trial = 1:13
%     plot(us,action1,'DisplayName','Action 1')
%     hold on
%     plot(us,action2,'DisplayName','Action 2')
%     plot(us,action3,'DisplayName','Action 3')
%     xlabel('Utility')
%     ylabel('Probability')
%     legend('Location','Best')
%     title(['Best action = 2, p(reward) = ' num2str(p_rew)])
%     fig = gcf; fig.Color = 'w'; hold off;
%     %pause
%     if trial == 1
%         a = 1; %try third one first, every time
%     else
%         [val,a] = max([sum(us.*action1) sum(us.*action2) sum(us.*action3)]);
%         % find distribution with highest mean utility, act greedily
%         % could softmax this, too
%     end
%     if a == 2
%         if rand() < p_rew %now give probabilistic reward
%             r = 100;
%         else
%             r = 0;
%         end
%     else
%         r = 0;
%     end
%     likelihood = normpdf(us,r,sigma);
%     eval(['proto = action' num2str(a) '.*likelihood;']) %update w new likelihood
%     eval(['action' num2str(a) ' = proto./sum(proto);']) %normalize
% end

%  Plot graveyard
% subplot(2,2,1)
% %for last subject, plot llh surface of model
% realvals = RLparams(end,:);
% testvals = [0:0.01:realvals(1) realvals(1) realvals(1):0.01:1; 0:0.1:realvals(2) realvals(2) realvals(2):0.1:10; ...
% ];%    0:0.02:realvals(3) realvals(3) realvals(3):0.02:1];
% for vals = 1:length(testvals)
%     p = 1;
%     nllh = get_nllh_RL([testvals(p,vals) realvals(2)]); %realvals(3)
%     scatter(testvals(p,vals),nllh,'Filled')
%     hold on
%     ylabel('neg llh')
%     xlabel('param val')
%     scatter(realvals(p),nllh+50,'*k')
% end