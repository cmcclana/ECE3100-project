close all

%% Initialize variables.
filename = '/Users/carriemcclanahan/Documents/ECE 3100/ECE3100-project/OnlineNewsPopularity2.csv';
delimiter = ',';
startRow = 2;
formatSpec = '%s%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%s%s%s%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string', 'EmptyValue', NaN, 'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');

%% Close the text file.
fclose(fileID);

%% Create output variable
OnlineNewsPopularityData = table(dataArray{1:end-1}, 'VariableNames', {'url','timedelta','n_tokens_title','n_tokens_content','n_unique_tokens','n_non_stop_words','n_non_stop_unique_tokens','num_hrefs','num_self_hrefs','num_imgs','num_videos','average_token_length','num_keywords','data_channel_is_lifestyle','data_channel_is_entertainment','data_channel_is_bus','data_channel_is_socmed','data_channel_is_tech','data_channel_is_world','kw_min_min','kw_max_min','kw_avg_min','kw_min_max','kw_max_max','kw_avg_max','kw_min_avg','kw_max_avg','kw_avg_avg','self_reference_min_shares','self_reference_max_shares','self_reference_avg_sharess','weekday_is_monday','weekday_is_tuesday','weekday_is_wednesday','weekday_is_thursday','weekday_is_friday','weekday_is_saturday','weekday_is_sunday','is_weekend','LDA_00','LDA_01','LDA_02','LDA_03','LDA_04','global_subjectivity','global_sentiment_polarity','global_rate_positive_words','global_rate_negative_words','rate_positive_words','rate_negative_words','avg_positive_polarity','min_positive_polarity','max_positive_polarity','avg_negative_polarity','min_negative_polarity','max_negative_polarity','title_subjectivity','title_sentiment_polarity','abs_title_subjectivity','abs_title_sentiment_polarity','shares', 'low_high_shares', 'low_mid_high_shares', 'low_high_shares2'});

%% Clear temporary variables
clearvars filename delimiter startRow formatSpec fileID dataArray ans;



%% CLASSIFICATION

%% 1) low_high_shares variable
% Make a new variable that is 1 - 1500 for low shares 
% and 1501 - 843,300 for high shares

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
    
% tabulate(OnlineNewsPopularityData.low_high_shares2)
%tabulate(OnlineNewsPopularityData.low_mid_high_shares)

%% TRAINING AND TEST SET DEFINITION

%% 1) FIRST ATTEMPT FEATURES
% X = OnlineNewsPopularityData(:,[8 10 11 19 29 30 31 45 54]);
% m = size(X,1);
% X0 = (ones(m, 1));
% X = [X0 table2array(OnlineNewsPopularityData(:,8)) table2array(OnlineNewsPopularityData(:,10))...
%     table2array(OnlineNewsPopularityData(:,11)) table2array(OnlineNewsPopularityData(:,19))...
%     table2array(OnlineNewsPopularityData(:,29)) table2array(OnlineNewsPopularityData(:,30))...
%     table2array(OnlineNewsPopularityData(:,31)) table2array(OnlineNewsPopularityData(:,45))...
%     table2array(OnlineNewsPopularityData(:,54))];

% degree = 3;
% %new dataset, has X0 and polynomial features up to degree
% Xexp = expand(X(:,2),X(:,3),degree);
% X = [Xexp X(:,4:end)];

%% Streamlined, start with first 3, only include those after that improve F1
% X = OnlineNewsPopularityData(:,[8 10 11 19 45 57 15]);
% m = size(X,1);
% X0 = (ones(m, 1));
% X = [X0 table2array(OnlineNewsPopularityData(:,8)) table2array(OnlineNewsPopularityData(:,10))...
%     table2array(OnlineNewsPopularityData(:,11)) table2array(OnlineNewsPopularityData(:,19))...
%     table2array(OnlineNewsPopularityData(:,45)) table2array(OnlineNewsPopularityData(:,57))...
%     table2array(OnlineNewsPopularityData(:,15))];

%% 2) add 3 features of our choosing: 28, 39, 50 - for low_high_shares2, decreases F1 score.
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

%% 3) add 3 features: 60, 12, 57 - unchanging F1 score
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

  %% 4) add 3 more features 13, 55, 15 - helps
% X = OnlineNewsPopularityData(:,[8 10 11 19 29 30 31 45 54 28 39 50 60 12 57 13 55 15]);
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
%     table2array(OnlineNewsPopularityData(:,15))];

  %% 4) take away 3 features 28, 39, 50 - increases F1 score
% X = OnlineNewsPopularityData(:,[8 10 11 19 29 30 31 45 54 60 12 57 13 55 15]);
% m = size(X,1);
% X0 = (ones(m, 1));
% X = [X0 table2array(OnlineNewsPopularityData(:,8)) table2array(OnlineNewsPopularityData(:,10))...
%     table2array(OnlineNewsPopularityData(:,11)) table2array(OnlineNewsPopularityData(:,19))...
%     table2array(OnlineNewsPopularityData(:,29)) table2array(OnlineNewsPopularityData(:,30))...
%     table2array(OnlineNewsPopularityData(:,31)) table2array(OnlineNewsPopularityData(:,45))...
%     table2array(OnlineNewsPopularityData(:,54)) table2array(OnlineNewsPopularityData(:,60))...
%     table2array(OnlineNewsPopularityData(:,12)) table2array(OnlineNewsPopularityData(:,57))...
%     table2array(OnlineNewsPopularityData(:,13)) table2array(OnlineNewsPopularityData(:,55))...
%     table2array(OnlineNewsPopularityData(:,15))];


   %% add all features
X = OnlineNewsPopularityData(:,[3 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60]);
m = size(X,1);
X0 = (ones(m, 1));
X = [X0 table2array(OnlineNewsPopularityData(:,3:60))];
%% MAKE Y ARRAY
y = strcmp(OnlineNewsPopularityData.low_high_shares2 ,'high');
%[X,i] = unique(X,'rows');   % remove duplicates (care!: sorted results!!!)
%y = y(i);

%% DEFINE TRAINING AND TEST SET
%Split data into 70% training 30% testing
%take random 70% of training samples and 30% testing
%27751 training samples and 11893 testing samples
rng(10)
pct = 0.7;
[m,n] = size(X);
i = randperm(m)';
m = round(pct*m);
X = X(i,:);
y = y(i);
%% PCA
[U,~,L] = pca(X(1:m,:));
%r = 27; % F1 = 0.62
%r = 33; % F1 = 0.63
%r = 34; % F1 = 0.63
%in between r = 36 and 37 becomes a problem with J in learning curves?
r = 42; % F1 = 0.64
%r = 47; % F1 = 0.64
X = X*U(:,1:r);
X = [X0 X];
fprintf('Variance retained: %.2f%%\n\n', (sum(L(1:r))/sum(L))*100);
n = size(X,2);
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
trnJ = cost(T,X(1:m,:),y(1:m),lambda);
tstJ = cost(T,X(m+1:end,:),y(m+1:end),lambda);

% gradient descent
%gradient params
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
%tabulate(trny)
%tabulate(tsty)

trnJ = cost(T,X(1:m,:),y(1:m),0);
tstJ = cost(T,X(m+1:end,:),y(m+1:end),0);
% display params
niter
T'

%  output = sigmoid(X*T);
%  output(output>=0.5) = 1;
%  output(output<0.5) = 0;
%  missed = sum(abs(output-y))


 %% TEST RESULTS
h = sigmoid(X(m+1:end,:)*T);
output = h;
output(h>=threshold) = 1;
output(h<threshold) = 0;
missed = sum(abs(output-y(m+1:end)))

% tabulate(y(m+1:end))
% tabulate(output)

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

 %% PLOTTING METRICS WITH RESPECT TO THRESHOLD
P = 100; % number of points
threshold = linspace(0,1,P);
err = zeros(size(threshold));
recall = zeros(size(threshold));
precision = zeros(size(threshold));
fscore = zeros(size(threshold));
for i = 1:P
    output = h;
    output(h>=threshold(i)) = 1;
    output(h<threshold(i)) = 0;
    tp = find(output==1 & y(m+1:end)==1);
    tp = numel(tp);
    fp = find(output==1 & y(m+1:end)==0);
    fp = numel(fp);
    tn = find(output==0 & y(m+1:end)==0);
    tn = numel(tn);
    fn = find(output==0 & y(m+1:end)==1);
    fn = numel(fn);
    err(i) = (fp+fn)/(tp+fp+tn+fn);
    recall(i) = tp/(tp+fn);
    precision(i) = tp/(tp+fp);
    fscore(i) = (1+beta^2)*(precision(i).*recall(i))/((beta^2)*precision(i)+recall(i));    
end
figure;
cAx = gca;
hold(cAx,'on');
plot(cAx,threshold,1-err);
plot(cAx,threshold,precision);
plot(cAx,threshold,recall);
plot(cAx,threshold,fscore,'k-','LineWidth',2);
xlabel('Threshold');
ylabel('Metric'); 
ylim([0 1]);
xlim([0 1]);
legend('Accurary','Precision','Recall','F Score');
grid(cAx,'on');
hold(cAx,'off');


%% TRAINING CLASSIFIER - learning curve with sample size
M = 28;
mtrn = size(trnX, 1);
sample_size = [2:1000:mtrn];
threshold = 0.5;
beta = 1;
options = optimoptions('fminunc','Display','off','SpecifyObjectiveGradient',true,'MaxIterations',1000);
err = zeros(M,2); % 1 -> training set, 2 -> test
recall = zeros(M,2);
precision = zeros(M,2);
fscore = zeros(M,2);
J = zeros(M,2);
for m = 1:M
    % training classifier
    n = size(X,2);
    T = 1e-5 * rand(n,1);
    [T,~] = fminunc(@(T)(cost(T,X(1:sample_size(m),:),y(1:sample_size(m)),lambda)),T,options);
    J(m,1) = cost(T,X(1:sample_size(m),:),y(1:sample_size(m)),0);
    % only training size will change. Test set keeps same size.
    J(m,2) = cost(T,X(mtrn+1:end,:),y(mtrn + 1:end),0);
    % metrics (train set)
    h = sigmoid(X(1:sample_size(m),:)*T);
    output = h;
    output(h>=threshold) = 1;
    output(h<threshold) = 0;
    tp = find(output==1 & y(1:sample_size(m))==1);
    tp = numel(tp);
    fp = find(output==1 & y(1:sample_size(m))==0);
    fp = numel(fp);
    tn = find(output==0 & y(1:sample_size(m))==0);
    tn = numel(tn);
    fn = find(output==0 & y(1:sample_size(m))==1);
    fn = numel(fn);
    err(m,1) = (fp+fn)/(tp+fp+tn+fn);
    recall(m,1) = tp/(tp+fn);
    precision(m,1) = tp/(tp+fp);
    fscore(m,1) = (1+beta^2)*(precision(m,1).*recall(m,1))/((beta^2)*precision(m,1)+recall(m,1));
    % metrics (test set)
    h = sigmoid(X(mtrn+1:end,:)*T);
    output = h;
    output(h>=threshold) = 1;
    output(h<threshold) = 0;
    tp = find(output==1 & y(mtrn+1:end)==1);
    tp = numel(tp);
    fp = find(output==1 & y(mtrn+1:end)==0);
    fp = numel(fp);
    tn = find(output==0 & y(mtrn+1:end)==0);
    tn = numel(tn);
    fn = find(output==0 & y(mtrn+1:end)==1);
    fn = numel(fn);
    err(m,2) = (fp+fn)/(tp+fp+tn+fn);
    recall(m,2) = tp/(tp+fn);
    precision(m,2) = tp/(tp+fp);
    fscore(m,2) = (1+beta^2)*(precision(m,2).*recall(m,2))/((beta^2)*precision(m,2)+recall(m,2));
end
%% PLOTTING METRICS WITH RESPECT TO SAMPLE SIZE
figure;
cAx = subplot(2,2,1);
hold(cAx,'on');
plot(cAx,sample_size,J(:,1),'ro');
plot(cAx,sample_size,J(:,2),'bo');
m = (2:1000:size(trnX,1));
plot(m,spline(sample_size,J(:,1),m),'r-','LineWidth',1.5);
plot(m,spline(sample_size,J(:,2),m),'b-','LineWidth',1.5);
xlabel('Sample Size');
ylabel('Final Cost (Error)'); 
legend('J^~(\theta)(Training)','J_t^~(\theta)(Test)');
grid(cAx,'on');
hold(cAx,'off');
cAx = subplot(2,2,2);
hold(cAx,'on');
plot(cAx,sample_size,err(:,1),'ro');
plot(cAx,sample_size,err(:,2),'bo');
plot(m,spline(sample_size,err(:,1),m),'r-','LineWidth',1.5);
plot(m,spline(sample_size,err(:,2),m),'b-','LineWidth',1.5);
ylim([0 1]);
xlabel('Sample Size');
ylabel('Misclassification Error'); 
legend('Err (Training)','Err (Test)');
grid(cAx,'on');
hold(cAx,'off');
cAx = subplot(2,2,3);
hold(cAx,'on');
plot(cAx,sample_size,1 - fscore(:,1),'ro');
plot(cAx,sample_size,1 - fscore(:,2),'bo');
plot(m,spline(sample_size,1 - fscore(:,1),m),'r-','LineWidth',1.5);
plot(m,spline(sample_size,1 - fscore(:,2),m),'b-','LineWidth',1.5);
ylim([0 1]);
xlabel('Sample Size');
ylabel('1 - (F Score)'); 
legend('F Score (Training)','F Score (Test)');
grid(cAx,'on');
hold(cAx,'off');
%% TRAINING CLASSIFIER - use fminunc to see if lambda improves model. It doesn't, there is not overfitting
% L = 12;
% lambda = [0 0.01*(2.^(0:L-2))];
% threshold = 0.5;
% beta = 1;
% options = optimoptions('fminunc','Display','off','SpecifyObjectiveGradient',true,'MaxIterations',1000);
% err = zeros(L,2); % 1 -> training set, 2 -> test
% recall = zeros(L,2);
% precision = zeros(L,2);
% fscore = zeros(L,2);
% J = zeros(L,2);
% for l = 1:L
%     % training classifier
%     n = size(X,2);
%     T = 1e-5 * rand(n,1);
%     [T,~] = fminunc(@(T)(cost(T,X(1:m,:),y(1:m),lambda(l))),T,options);
%     J(l,1) = cost(T,X(1:m,:),y(1:m),0);
%     J(l,2) = cost(T,X(m+1:end,:),y(m+1:end),0);
%     % metrics (train set)
%     h = sigmoid(X(1:m,:)*T);
%     output = h;
%     output(h>=threshold) = 1;
%     output(h<threshold) = 0;
%     tp = find(output==1 & y(1:m)==1);
%     tp = numel(tp);
%     fp = find(output==1 & y(1:m)==0);
%     fp = numel(fp);
%     tn = find(output==0 & y(1:m)==0);
%     tn = numel(tn);
%     fn = find(output==0 & y(1:m)==1);
%     fn = numel(fn);
%     err(l,1) = (fp+fn)/(tp+fp+tn+fn);
%     recall(l,1) = tp/(tp+fn);
%     precision(l,1) = tp/(tp+fp);
%     fscore(l,1) = (1+beta^2)*(precision(l,1).*recall(l,1))/((beta^2)*precision(l,1)+recall(l,1));
%     % metrics (test set)
%     h = sigmoid(X(m+1:end,:)*T);
%     output = h;
%     output(h>=threshold) = 1;
%     output(h<threshold) = 0;
%     tp = find(output==1 & y(m+1:end)==1);
%     tp = numel(tp);
%     fp = find(output==1 & y(m+1:end)==0);
%     fp = numel(fp);
%     tn = find(output==0 & y(m+1:end)==0);
%     tn = numel(tn);
%     fn = find(output==0 & y(m+1:end)==1);
%     fn = numel(fn);
%     err(l,2) = (fp+fn)/(tp+fp+tn+fn);
%     recall(l,2) = tp/(tp+fn);
%     precision(l,2) = tp/(tp+fp);
%     fscore(l,2) = (1+beta^2)*(precision(l,2).*recall(l,2))/((beta^2)*precision(l,2)+recall(l,2));
% end
% %% PLOTTING METRICS WITH RESPECT TO LAMBDA
% figure;
% cAx = subplot(2,2,1);
% hold(cAx,'on');
% plot(cAx,lambda,J(:,1),'ro');
% plot(cAx,lambda,J(:,2),'bo');
% l = (0:0.1:L);
% plot(l,spline(lambda,J(:,1),l),'r-','LineWidth',1.5);
% plot(l,spline(lambda,J(:,2),l),'b-','LineWidth',1.5);
% xlabel('Lambda');
% ylabel('Final Cost (Error)'); 
% legend('J^~(\theta)(Training)','J_t^~(\theta)(Test)');
% grid(cAx,'on');
% hold(cAx,'off');
% cAx = subplot(2,2,2);
% hold(cAx,'on');
% plot(cAx,lambda,err(:,1),'ro');
% plot(cAx,lambda,err(:,2),'bo');
% plot(l,spline(lambda,err(:,1),l),'r-','LineWidth',1.5);
% plot(l,spline(lambda,err(:,2),l),'b-','LineWidth',1.5);
% ylim([0 1]);
% xlabel('Lambda');
% ylabel('Misclassification Error'); 
% legend('Err (Training)','Err (Test)');
% grid(cAx,'on');
% hold(cAx,'off');
% cAx = subplot(2,2,3);
% hold(cAx,'on');
% plot(cAx,lambda,1 - fscore(:,1),'ro');
% plot(cAx,lambda,1 - fscore(:,2),'bo');
% plot(l,spline(lambda,1 - fscore(:,1),l),'r-','LineWidth',1.5);
% plot(l,spline(lambda,1 - fscore(:,2),l),'b-','LineWidth',1.5);
% ylim([0 1]);
% xlabel('Lambda');
% ylabel('1 - (F Score)'); 
% legend('F Score (Training)','F Score (Test)');
% grid(cAx,'on');
% hold(cAx,'off');

 %% TRYING DIFFERENT HYPHOTHESIS COMPLEXITY BASED ON HOW MANY POLYNOMIAL FEATURES WE USE - does not work yet
% threshold = 0.5;
% lambda = 0.0;
% beta = 1;
% options = optimoptions('fminunc','Display','off','SpecifyObjectiveGradient',true,'MaxIterations',1000);
% D = 30;
% err = zeros(D,2); % 1 -> training set, 2 -> test
% recall = zeros(D,2);
% precision = zeros(D,2);
% fscore = zeros(D,12);
% J = zeros(D,2);
% for d = 1:D
%     % building features
%     x = expand(X(:,1),X(:,2),d);
%     x = x(i,:);
%     % feature scaling
%     avg = mean(x(1:m,2:end));
%     var = std(x(1:m,2:end));
%     x(:,2:end) = (x(:,2:end) - avg)./var;
%     % training classifier
%     n = size(x,2);
%     T = 1e-5 * rand(n,1);
%     [T,~] = fminunc(@(T)(cost(T,x(1:m,:),Y(1:m),lambda)),T,options);
%     J(d,1) = cost(T,x(1:m,:),Y(1:m),0);
%     J(d,2) = cost(T,x(m+1:end,:),Y(m+1:end),0);
%     % metrics (train set)
%     h = sigmoid(x(1:m,:)*T);
%     output = h;
%     output(h>=threshold) = 1;
%     output(h<threshold) = 0;
%     tp = find(output==1 & Y(1:m)==1);
%     tp = numel(tp);
%     fp = find(output==1 & Y(1:m)==0);
%     fp = numel(fp);
%     tn = find(output==0 & Y(1:m)==0);
%     tn = numel(tn);
%     fn = find(output==0 & Y(1:m)==1);
%     fn = numel(fn);
%     err(d,1) = (fp+fn)/(tp+fp+tn+fn);
%     recall(d,1) = tp/(tp+fn);
%     precision(d,1) = tp/(tp+fp);
%     fscore(d,1) = (1+beta^2)*(precision(d,1).*recall(d,1))/((beta^2)*precision(d,1)+recall(d,1));
%     % metrics (test set)
%     h = sigmoid(x(m+1:end,:)*T);
%     output = h;
%     output(h>=threshold) = 1;
%     output(h<threshold) = 0;
%     tp = find(output==1 & Y(m+1:end)==1);
%     tp = numel(tp);
%     fp = find(output==1 & Y(m+1:end)==0);
%     fp = numel(fp);
%     tn = find(output==0 & Y(m+1:end)==0);
%     tn = numel(tn);
%     fn = find(output==0 & Y(m+1:end)==1);
%     fn = numel(fn);
%     err(d,2) = (fp+fn)/(tp+fp+tn+fn);
%     recall(d,2) = tp/(tp+fn);
%     precision(d,2) = tp/(tp+fp);
%     fscore(d,2) = (1+beta^2)*(precision(d,2).*recall(d,2))/((beta^2)*precision(d,2)+recall(d,2));
% end
% %% PLOTTING METRICS WITH RESPECT TO DEGREE
% figure;
% cAx = subplot(2,2,1);
% hold(cAx,'on');
% plot(cAx,(1:D),J(:,1),'ro');
% plot(cAx,(1:D),J(:,2),'bo');
% d = (1:0.1:D);
% plot(d,spline((1:D),J(:,1),d),'r-','LineWidth',1.5);
% plot(d,spline((1:D),J(:,2),d),'b-','LineWidth',1.5);
% xlabel('Degree');
% ylabel('Final Cost (Error)'); 
% legend('J^~(\theta)(Training)','J_t^~(\theta)(Test)');
% grid(cAx,'on');
% hold(cAx,'off');
% cAx = subplot(2,2,2);
% hold(cAx,'on');
% plot(cAx,(1:D),err(:,1),'ro');
% plot(cAx,(1:D),err(:,2),'bo');
% plot(d,spline((1:D),err(:,1),d),'r-','LineWidth',1.5);
% plot(d,spline((1:D),err(:,2),d),'b-','LineWidth',1.5);
% ylim([0 1]);
% xlabel('Degree');
% ylabel('Misclassification Error'); 
% legend('Err (Training)','Err (Test)');
% grid(cAx,'on');
% hold(cAx,'off');
% cAx = subplot(2,2,3);
% hold(cAx,'on');
% plot(cAx,(1:D),1 - fscore(:,1),'ro');
% plot(cAx,(1:D),1 - fscore(:,2),'bo');
% plot(d,spline((1:D),1 - fscore(:,1),d),'r-','LineWidth',1.5);
% plot(d,spline((1:D),1 - fscore(:,2),d),'b-','LineWidth',1.5);
% ylim([0 1]);
% xlabel('Degree');
% ylabel('1 - (F Score)'); 
% legend('F Score (Training)','F Score (Test)');
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

%% FUNCTION TO ADD POLYNOMIAL TERMS
function [X] = expand(x1,x2,degree)
    m = size(x1,1);
    X = ones(m,(degree+1)*(degree+2)/2); % (d+1)*(d+2)/2 -> consecutive numbers summation
    n = 2;
    for i = 1:degree
        for j = 0:i
            X(:,n) = (x1.^(i-j)).*(x2.^j);
            n = n + 1; 
        end
    end
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

%% MULTICLASS CLASSIFICATION

%% LowMidHighShares variable
% Make a new variable that is 1 - 1500 for low shares 
% and 1501 - 5,000 for mid shares 
% and 5,001 - 843,000 for high shares
