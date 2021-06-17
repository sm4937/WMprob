%% Code written by Sam McDougle, for UC Berkeley
% Loads data, displays learning curves, delay effects, RT patterns
% for WMP task (data collected in 2018 in lab and on AMT)
% Original data collection by William Ryan and Sarah Master
% Code co-opted by Sarah Master in 2021
n_subs = length(subs);
bad_trials = zeros(1,n_subs);good_trials = zeros(1,n_subs);

for si = 1:n_subs
    % load data
    load(['Data/WMP_ID',num2str(subs(si))]);
    block_data = dataT;
    blocks = block_data{end}.blocks; % no "-1" for pnas/ejn, -1 for FMRI
    
    % initialize
    delay = nan(length(blocks),108); % delay temp
    iter = nan(length(blocks),108); % iter temp
    corblock = nan(length(blocks),max_iter);
    delblock = nan(length(blocks),3);
    rtblock = nan(length(blocks),max_iter);
    nssubs = nan(4,max_iter);
    delsubs = nan(4,3);
    rtsubs = nan(4,max_iter);    
    % block loop
    for b = 1:length(blocks)
        ns(b) = blocks(b);
        pr(b) = probs(b);
        na = 3;
        data = block_data{b};
        correct = data.Cor;
        reward = data.Rew;
        % invalids
        correct(correct<0) = NaN; %% too slow
        toofast = data.RT < 0.150; % invalids 2 (too fast)
        correct(toofast==1) = NaN; %% too fast       
        reward(reward<0) = NaN;
        reward(toofast==1) = NaN;        
        num_trials = length(correct);
        bad_trials(si) = bad_trials(si)+sum(isnan(correct));
        good_trials(si) = good_trials(si)+sum(~isnan(correct));
        
        seq = data.seq(1:num_trials);
        sub_action = data.Code;
        cor = nan(ns(b),max_iter);
        rt_tc = nan(ns(b),max_iter);        
        % set size effects
        for k = 1:ns(b)
            idx = correct(seq==k);
            cor(k,1:length(idx)) = idx;
            tmpid = find(seq==k);%&correct==1); % correct RTs?
            idx2 = data.RT(tmpid);            
            rt_tc(k,1:length(idx2)) = idx2;
        end
        
        corblock(b,:) = nanmean(cor);
        rtblock(b,:) = nanmedian(rt_tc);
        perf(si,b) = nanmean(nanmean(cor));
        
        % delay effects
        for t = 1:num_trials
            %% print behavior
            if b > 2
%             disp(['stim ',num2str(seq(t))]);
%             disp(['action ',num2str(sub_action(t))]);
%             disp(['outcome ',num2str(reward(t))]);
%             disp(['truth? ',num2str(reward(t)==correct(t))]);
%             waitforbuttonpress;
            end
            id=find(seq==seq(t));
            iter(b,t) = length(find(id<=t));
            if iter(b,t)>1
                delay(b,t) = t-id(find(id==t)-1);
                % only look post correct trials            
                if correct(id(find(id==t)-1)) < 1 && reward(id(find(id==t)-1)) < 1 
                    delay(b,t) = NaN;
                end
            end
            probcheck(si,b) = nanmean(correct==reward(1:length(correct)));
        end
        % delay
        delblock(b,:) = [nanmean(correct(delay(b,:)==1)) nanmean(correct(delay(b,:)==2)) nanmean(correct(delay(b,:)>=3))];
    end
    
    nssubs(1,:) = nanmean(corblock(ns==3 & pr==2,:));
    nssubs(2,:) = nanmean(corblock(ns==3 & pr==1,:));
    nssubs(3,:) = nanmean(corblock(ns==6 & pr==2,:));
    nssubs(4,:) = nanmean(corblock(ns==6 & pr==1,:));
    
    delsubs(1,:) = nanmean(delblock(ns==3 & pr==2,:));
    delsubs(2,:) = nanmean(delblock(ns==3 & pr==1,:));
    delsubs(3,:) = nanmean(delblock(ns==6 & pr==2,:));
    delsubs(4,:) = nanmean(delblock(ns==6 & pr==1,:));
    
    rtsubs(1,:) = nanmean(rtblock(ns==3 & pr==2,:));
    rtsubs(2,:) = nanmean(rtblock(ns==3 & pr==1,:));
    rtsubs(3,:) = nanmean(rtblock(ns==6 & pr==2,:));
    rtsubs(4,:) = nanmean(rtblock(ns==6 & pr==1,:));    
    
    ns3hr(si,:) = nssubs(1,:);
    ns3lr(si,:) = nssubs(2,:);
    ns6hr(si,:) = nssubs(3,:);    
    ns6lr(si,:) = nssubs(4,:);
    del3hr(si,:) = delsubs(1,:);
    del3lr(si,:) = delsubs(2,:);
    del6hr(si,:) = delsubs(3,:);
    del6lr(si,:) = delsubs(4,:);    
    rt3hr(si,:) = rtsubs(1,:);
    rt3lr(si,:) = rtsubs(2,:);
    rt6hr(si,:) = rtsubs(3,:);
    rt6lr(si,:) = rtsubs(4,:);      
end


%% PLOTS
figure;
cs = {[0.3216    0.6745    0.7020],[0.5647    0.9686    1.0000],...
    [0.7020    0.4627    0.3020],[1.0000    0.7725    0.6196]};

%% ns
subplot(1,3,1);
plot(mean(ns3hr),'color',cs{1},'linewidth',2);hold on;
plot(mean(ns3lr),'color',cs{2},'linewidth',2);
plot(mean(ns6hr),'color',cs{3},'linewidth',2);
plot(mean(ns6lr),'color',cs{4},'linewidth',2);
% sem
for i = 1:max_iter
    errorbar(i,mean(ns3hr(:,i)),std(ns3hr(:,i))/sqrt(n_subs),'Color',cs{1});
    errorbar(i,mean(ns3lr(:,i)),std(ns3lr(:,i))/sqrt(n_subs),'Color',cs{2});
    errorbar(i,mean(ns6hr(:,i)),std(ns6hr(:,i))/sqrt(n_subs),'Color',cs{3});
    errorbar(i,mean(ns6lr(:,i)),std(ns6lr(:,i))/sqrt(n_subs),'Color',cs{4});
end
plot([0 15],[.33 .33],'k--');
axis([0 13 0 1]);
lgd = legend('ns3 p(high)','ns3 p(low)','ns6 p(high)','ns6 p(low)','location','southeast');
title(lgd,'condition')
legend('boxoff');
box off
set(gca,'xtick',1:15);
xlabel('Stimulus iteration');
ylabel('p(correct)');

%% delay
subplot(1,3,2);
plot(mean(del3hr),'color',cs{1},'linewidth',2);hold on;
plot(mean(del3lr),'color',cs{2},'linewidth',2);
plot(mean(del6hr),'color',cs{3},'linewidth',2);
plot(mean(del6lr),'color',cs{4},'linewidth',2);

% sem
for i = 1:3
    errorbar(i,mean(del3hr(:,i)),std(del3hr(:,i))/sqrt(n_subs),'Color',cs{1});
    errorbar(i,mean(del3lr(:,i)),std(del3lr(:,i))/sqrt(n_subs),'Color',cs{2});
    errorbar(i,mean(del6hr(:,i)),std(del6hr(:,i))/sqrt(n_subs),'Color',cs{3});
    errorbar(i,mean(del6lr(:,i)),std(del6lr(:,i))/sqrt(n_subs),'Color',cs{4});
end
box off
axis([.5 3.5 .5 1]);
set(gca,'xtick',1:3,'xticklabel',{'1','2','>2'});
xlabel('Delay (since last correct)');
% sdm_figure_letters(2,1,1,.14,.05);
ylabel('p(correct)');

%% rt tc
subplot(1,3,3);
plot(mean(rt3hr),'color',cs{1},'linewidth',2);hold on;
plot(mean(rt3lr),'color',cs{2},'linewidth',2);
plot(mean(rt6hr),'color',cs{3},'linewidth',2);
plot(mean(rt6lr),'color',cs{4},'linewidth',2);

% sem
for i = 1:max_iter
    
    errorbar(i,mean(rt3hr(:,i)),std(rt3hr(:,i))/sqrt(n_subs),'Color',cs{1});
    errorbar(i,mean(rt3lr(:,i)),std(rt3lr(:,i))/sqrt(n_subs),'Color',cs{2});
    errorbar(i,mean(rt6hr(:,i)),std(rt6hr(:,i))/sqrt(n_subs),'Color',cs{3});
    errorbar(i,mean(rt6lr(:,i)),std(rt6lr(:,i))/sqrt(n_subs),'Color',cs{4});
end
box off
axis([0 13 .4 .75]);
set(gca,'xtick',1:15);
xlabel('Stimulus iteration');
ylabel('rt (s)');
% sdm_figure_letters(3,1,1,.14,.05);


%% global
% set(gcf,'Position',[1 410 824 295]);
set(gcf,'Position',[1 517 803 188]);
% print -dtiff -r300 figures/behavior_learningTEST


%% exclusions?
% for k = 1:n_subs
%     [h(k),p(k),~,~]=ttest(perf(k,:)-0.3333);
% end
% excl = find(p>0.05);