close all

%% Initialize variables.
filename = '/Users/carriemcclanahan/Documents/ECE 3100/ECE3100-project/OnlineNewsPopularity2.csv';
delimiter = ',';
startRow = 2;


% For more information, see the TEXTSCAN documentation.
formatSpec = '%s%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%s%s%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to the format.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string', 'EmptyValue', NaN, 'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');

%% Close the text file.
fclose(fileID);

%% Post processing for unimportable data.
% No unimportable data rules were applied during the import, so no post
% processing code is included. To generate code which works for
% unimportable data, select unimportable cells in a file and regenerate the
% script.

%% Create output variable
OnlineNewsPopularityData = table(dataArray{1:end-1}, 'VariableNames', {'url','timedelta','n_tokens_title','n_tokens_content','n_unique_tokens','n_non_stop_words','n_non_stop_unique_tokens','num_hrefs','num_self_hrefs','num_imgs','num_videos','average_token_length','num_keywords','data_channel_is_lifestyle','data_channel_is_entertainment','data_channel_is_bus','data_channel_is_socmed','data_channel_is_tech','data_channel_is_world','kw_min_min','kw_max_min','kw_avg_min','kw_min_max','kw_max_max','kw_avg_max','kw_min_avg','kw_max_avg','kw_avg_avg','self_reference_min_shares','self_reference_max_shares','self_reference_avg_sharess','weekday_is_monday','weekday_is_tuesday','weekday_is_wednesday','weekday_is_thursday','weekday_is_friday','weekday_is_saturday','weekday_is_sunday','is_weekend','LDA_00','LDA_01','LDA_02','LDA_03','LDA_04','global_subjectivity','global_sentiment_polarity','global_rate_positive_words','global_rate_negative_words','rate_positive_words','rate_negative_words','avg_positive_polarity','min_positive_polarity','max_positive_polarity','avg_negative_polarity','min_negative_polarity','max_negative_polarity','title_subjectivity','title_sentiment_polarity','abs_title_subjectivity','abs_title_sentiment_polarity','shares', 'low_high_shares', 'low_mid_high_shares'});

%% Clear temporary variables
clearvars filename delimiter startRow formatSpec fileID dataArray ans;




%% CLASSIFICATION

%% 1) low_high_shares variable
% Make a new variable that is 1 - 3,000 for low shares 
% and 3,001 - 843,300 for high shares
shares = OnlineNewsPopularityData{:,61};
%did work in excel instead of below

% url = OnlineNewsPopularityData{:,1};
% OnlineNewsPopularityData.low_high_shares = url;
% for i = 1:numel(shares)
%     if (shares(i) <= 3000)
%         OnlineNewsPopularityData.low_high_shares(i) = 'low';
%     elseif (shares(i) < 843301)
%         OnlineNewsPopularityData.low_high_shares(i) = 'high';
%     end
% end
    
%tabulate(OnlineNewsPopularityData.low_high_shares)
%tabulate(OnlineNewsPopularityData.low_mid_high_shares)
%% 2)  Split data into 70% training 30% testing
%take random 70% of training samples and 30% testing
%27751 training samples and 11893 testing samples

%% TRAINING AND TEST SET DEFINITION

%% FIRST ATTEMPT FEATURES
% X = OnlineNewsPopularityData(:,[8 10 11 19 29 30 31 45 54]);
% m = size(X,1);
% X0 = (ones(m, 1));
% X = [X0 table2array(OnlineNewsPopularityData(:,8)) table2array(OnlineNewsPopularityData(:,10))...
%     table2array(OnlineNewsPopularityData(:,11)) table2array(OnlineNewsPopularityData(:,19))...
%     table2array(OnlineNewsPopularityData(:,29)) table2array(OnlineNewsPopularityData(:,30))...
%     table2array(OnlineNewsPopularityData(:,31)) table2array(OnlineNewsPopularityData(:,45))...
%     table2array(OnlineNewsPopularityData(:,54))];

%% add 3 features: 28, 39, 50
% X = OnlineNewsPopularityData(:,[8 10 11 19 29 30 31 45 54 28 39 50]);
% m = size(X,1);
% X0 = (ones(m, 1));
% X = [X0 table2array(OnlineNewsPopularityData(:,8)) table2array(OnlineNewsPopularityData(:,10))...
%     table2array(OnlineNewsPopularityData(:,11)) table2array(OnlineNewsPopularityData(:,19))...
%     table2array(OnlineNewsPopularityData(:,29)) table2array(OnlineNewsPopularityData(:,30))...
%     table2array(OnlineNewsPopularityData(:,31)) table2array(OnlineNewsPopularityData(:,45))...
%     table2array(OnlineNewsPopularityData(:,54)) ...
%     table2array(OnlineNewsPopularityData(:,28)) table2array(OnlineNewsPopularityData(:,39))...
%     table2array(OnlineNewsPopularityData(:,50))];

%% add 3 features: 60, 12, 57
% X = OnlineNewsPopularityData(:,[8 10 11 19 29 30 31 45 54 28 39 50 60 12 57]);
% m = size(X,1);
% X0 = (ones(m, 1));
% X = [X0 table2array(OnlineNewsPopularityData(:,8)) table2array(OnlineNewsPopularityData(:,10))...
%     table2array(OnlineNewsPopularityData(:,11)) table2array(OnlineNewsPopularityData(:,19))...
%     table2array(OnlineNewsPopularityData(:,29)) table2array(OnlineNewsPopularityData(:,30))...
%     table2array(OnlineNewsPopularityData(:,31)) table2array(OnlineNewsPopularityData(:,45))...
%     table2array(OnlineNewsPopularityData(:,54)) ...
%     table2array(OnlineNewsPopularityData(:,28)) table2array(OnlineNewsPopularityData(:,39))...
%     table2array(OnlineNewsPopularityData(:,50)) table2array(OnlineNewsPopularityData(:,60))...
%     table2array(OnlineNewsPopularityData(:,12)) table2array(OnlineNewsPopularityData(:,57))];

%% add 3 more features 13, 55, 15
% 19 28 15 have highest theta in end, these seem important
X = OnlineNewsPopularityData(:,[8 10 11 19 29 30 31 45 54 28 39 50 60 12 57 13 55 15]);
m = size(X,1);
X0 = (ones(m, 1));
X = [X0 table2array(OnlineNewsPopularityData(:,8)) table2array(OnlineNewsPopularityData(:,10))...
    table2array(OnlineNewsPopularityData(:,11)) table2array(OnlineNewsPopularityData(:,19))...
    table2array(OnlineNewsPopularityData(:,29)) table2array(OnlineNewsPopularityData(:,30))...
    table2array(OnlineNewsPopularityData(:,31)) table2array(OnlineNewsPopularityData(:,45))...
    table2array(OnlineNewsPopularityData(:,54)) ...
    table2array(OnlineNewsPopularityData(:,28)) table2array(OnlineNewsPopularityData(:,39))...
    table2array(OnlineNewsPopularityData(:,50)) table2array(OnlineNewsPopularityData(:,60))...
    table2array(OnlineNewsPopularityData(:,12)) table2array(OnlineNewsPopularityData(:,57))...
    table2array(OnlineNewsPopularityData(:,13)) table2array(OnlineNewsPopularityData(:,55))...
    table2array(OnlineNewsPopularityData(:,15))];

%% add 3 more features 3, 4, 9
% X = OnlineNewsPopularityData(:,[8 10 11 19 29 30 31 45 54 28 39 50 60 12 57 13 55 15 3 4 9]);
% m = size(X,1);
% X0 = (ones(m, 1));
% X = [X0 table2array(OnlineNewsPopularityData(:,8)) table2array(OnlineNewsPopularityData(:,10))...
%     table2array(OnlineNewsPopularityData(:,11)) table2array(OnlineNewsPopularityData(:,19))...
%     table2array(OnlineNewsPopularityData(:,29)) table2array(OnlineNewsPopularityData(:,30))...
%     table2array(OnlineNewsPopularityData(:,31)) table2array(OnlineNewsPopularityData(:,45))...
%     table2array(OnlineNewsPopularityData(:,54)) ...
%     table2array(OnlineNewsPopularityData(:,28)) table2array(OnlineNewsPopularityData(:,39))...
%     table2array(OnlineNewsPopularityData(:,50)) table2array(OnlineNewsPopularityData(:,60))...
%     table2array(OnlineNewsPopularityData(:,12)) table2array(OnlineNewsPopularityData(:,57))...
%     table2array(OnlineNewsPopularityData(:,13)) table2array(OnlineNewsPopularityData(:,55))...
%     table2array(OnlineNewsPopularityData(:,15)) table2array(OnlineNewsPopularityData(:,3))...
%     table2array(OnlineNewsPopularityData(:,4)) table2array(OnlineNewsPopularityData(:,9))];
%% MAKE Y ARRAY
y = strcmp(OnlineNewsPopularityData.low_high_shares ,'high');
%[X,i] = unique(X,'rows');   % remove duplicates (care!: sorted results!!!)
%y = y(i);

%% DEFINE TRAINING AND TEST SET
rng(10)
pct = 0.7;
[m,n] = size(X);
i = randperm(m)';
m = round(pct*m);
X = X(i,:);
y = y(i);

%% FEATURE SCALING AND MEAN NORMALIZATION
avg = mean(X(1:m,2:end));
var = std(X(1:m,2:end));
X(:,2:end) = (X(:,2:end) - avg)./var;

%% TRAINING CLASSIFIER
threshold = 0.5;
lambda = 0;
T = 1e-5 * rand(n,1);
%need to make gradient find lowest cost and get T to find test error
%why are training and test cost the same?
trnJ = cost(T,X(1:m,:),y(1:m),0)
tstJ = cost(T,X(m+1:end,:),y(m+1:end),0)

%% gradient descent
% gradient params
maxiter = 10000;
mindJ = 1e-10;
alpha = 2;
dJ = mindJ;
J = zeros(maxiter,1);
h = sigmoid(X(1:m,:)*T);
niter = 1;
J(niter) = -(1/m)*sum(y(1:m).*log(h)+(1-y(1:m)).*log(1 - h)) + (lambda/(2*m))*sum(T(2:end).^2);
% gradient descent
while (niter < maxiter && abs(dJ) >= mindJ)
    % gradient step
    niter = niter + 1;
    grad = (1/m)*(X(1:m,:)'*(h - y(1:m))) + [0; (lambda/m)*T(2:end)];
    T = T - alpha * grad;
    h = sigmoid(X(1:m,:)*T);
    J(niter) = -(1/m)*sum(y(1:m).*log(h)+(1-y(1:m)).*log(1 - h)) + (lambda/(2*m))*sum(T(2:end).^2);
    dJ = J(niter - 1) - J(niter);
end


plot(1:niter,J(1:niter));
xlabel('Iteration');
ylabel('J(\theta)');
legend('J(\theta)');
grid('on');


trnX = X(1:m,:);
trny = y(1:m);
tstX = X(m+1:end,:);
tsty = y(m+1:end);

trnJ = cost(T,X(1:m,:),y(1:m),0)
tstJ = cost(T,X(m+1:end,:),y(m+1:end),0)
% display params
niter
T'

% output = sigmoid(X*T);
% output(output>=0.5) = 1;
% output(output<0.5) = 0;
% missed = sum(abs(output-y))


%% TEST RESULTS
h = sigmoid(X(m+1:end,:)*T);
output = h;
output(h>=threshold) = 1;
output(h<threshold) = 0;
% missing all positives, classifying all as low shares, none as high shares!
missed = sum(abs(output-y(m+1:end)))

tp = find(output==1 & y(m+1:end)==1);
tp = numel(tp);

fp = find(output==1 & y(m+1:end)==0);
fp = numel(fp);

tn = find(output==0 & y(m+1:end)==0);
tn = numel(tn);

fn = find(output==0 & y(m+1:end)==1);
fn = numel(fn);

err = (fp+fn)/(tp+fp+tn+fn);
recall = tp/(tp+fn);
precision = tp/(tp+fp);
beta = 1;
fscore = (1+beta^2)*(precision.*recall)/((beta^2)*precision+recall);

%% PRINTING SOME DATA
disp('--------------------RESULTS-------------------------');
fprintf('Final Error J(Training): %.4f\n',trnJ);
fprintf('Final Error J(Test): %.4f\n',tstJ);
fprintf('Accuracy (1 - Err)(Test): %.2f\n',1-err);
fprintf('Recall (Test): %.2f\n',recall);
fprintf('Precision (Test): %.2f\n',precision);
fprintf('F%d Score (Test): %.2f\n',beta,fscore);
disp('----------------------------------------------------');
% %% PLOTTING METRICS WITH RESPECT TO THRESHOLD
% P = 100; % number of points
% threshold = linspace(0,1,P);
% err = zeros(size(threshold));
% recall = zeros(size(threshold));
% precision = zeros(size(threshold));
% fscore = zeros(size(threshold));
% for i = 1:P
%     output = h;
%     output(h>=threshold(i)) = 1;
%     output(h<threshold(i)) = 0;
%     tp = find(output==1 & y(m+1:end)==1);
%     tp = numel(tp);
%     fp = find(output==1 & y(m+1:end)==0);
%     fp = numel(fp);
%     tn = find(output==0 & y(m+1:end)==0);
%     tn = numel(tn);
%     fn = find(output==0 & y(m+1:end)==1);
%     fn = numel(fn);
%     err(i) = (fp+fn)/(tp+fp+tn+fn);
%     recall(i) = tp/(tp+fn);
%     precision(i) = tp/(tp+fp);
%     fscore(i) = (1+beta^2)*(precision(i).*recall(i))/((beta^2)*precision(i)+recall(i));    
% end
% figure;
% cAx = gca;
% hold(cAx,'on');
% plot(cAx,threshold,1-err);
% plot(cAx,threshold,precision);
% plot(cAx,threshold,recall);
% plot(cAx,threshold,fscore,'k-','LineWidth',2);
% xlabel('Threshold');
% ylabel('Metric'); 
% ylim([0 1]);
% xlim([0 1]);
% legend('Accurary','Precision','Recall','F Score');
% grid(cAx,'on');
% hold(cAx,'off');

%% SIGMOID FUNCTION
function f = sigmoid (x)
    f = 1./(1+exp(-x));
end

%% FUNCTION TO COMPUTE COST AND GRADIENT
function [J, grad] = cost(T,X,y,lambda)
    m = size(X,1);
    h = sigmoid(X*T);
    J = -(1/m)*sum(y.*log(h)+(1-y).*log(1 - h)) + (lambda/(2*m))*sum(T(2:end).^2);
    grad = (1/m)*(X'*(h - y)) + [0; (lambda/m)*T(2:end)];
end
%%

%features to test first are self_reference_avg_shares (31), num_videos (11), 
%self_reference_min_shares (29), data_channel_is_world (19), 
%self_reference_max_shares (30), num_hrefs (8), num_imgs (10), 
%avg_negative polarity (54), global subjectivity  (45),

%second we will add:
% kw_avg_avg(28), is_weekend(39), rate_negative_words(50)


%third we will add:
%abs_title_sentiment_polarity (60), average_token_length (12), title_subjectivity (57)

%fourth we will add
%num_keywords (13), min_negative_polarity (55), data_channel_is_entertainment
%(15)

%LDA_03 and LDA_02 are top but we don't think they make sense

%Predict y = 0 for Low shares (1 - 3000)
%Predict y = 1 for High shares (3001 - 843,300)


%% MULTICLASS CLASSIFICATION

%% LowMidHighShares variable
% Make a new variable that is 1 - 1500 for low shares 
% and 1501 - 5,000 for mid shares 
% and 5,001 - 843,000 for high shares