function [] = SLM_plot_learningcurves(data)
%SLM_plot_learningcurves for WMP
%   Plot learning curves over different set sizes and different probability
%   conditions

colors = [254 80 0; 0 133 202; 254 80 0; 0 133 202]./255; %make go from 0 to 1
ps = unique(data{1}.prew); %grab p_reward numbers from first subject, since it's the same for all subj anyway
code = [3 ps(1); 6 ps(1); 3 ps(2); 6 ps(2)];

N = size(data,2);
for s = 1:N
    one = data{s};
    one.cor(one.cor<0) = 0; %erase -1's from cor vector
    for cc = 1:size(code,1) %reinitialize for each subject
        eval(['lc_' num2str(cc) ' = NaN(1,17);']) %initialize matrix for saving condition corrects
    end
    for b = 1:length(unique(one.block)) %loop over blocks
        block = one.block==b;
        ns = unique(one.ns(block));
        type = find(code(:,1)==ns&code(:,2)==unique(one.prew(block)));
        for stim = 1:ns
            cor_vec = NaN(1,17);
            cor_vec(1:sum(block&one.stim==stim)) = one.cor(block&one.stim==stim); %grab stim presentations in order
            eval(['lc_' num2str(type) ' = [lc_' num2str(type) '; cor_vec];']);
        end
    end
    for cc = 1:size(code,1)
        eval(['subj_summary_' num2str(cc) '(s,:) = nanmean(lc_' num2str(cc) ');'])
    end
end

%figure should already exist
for t = 1:size(code,1)
    eval(['measure = subj_summary_' num2str(t) ';'])
    hold on; style = '-';
    if t < 3 %low probability of reward
        style = '--';
    end
    errorbar(nanmean(measure),nanstd(measure)./sqrt(N),'Color',colors(t,:),'LineWidth',1.5,'LineStyle',style)
end
legend({['ns = 3, p(reward) = ' num2str(round(ps(1),2,'significant'))],['ns = 6, p(reward) = ' num2str(round(ps(1),2,'significant'))],['ns = 3, p(reward) = ' num2str(round(ps(2),2,'significant'))],['ns = 6, p(reward) = ' num2str(round(ps(2),2,'significant'))]})
% categorize
end

