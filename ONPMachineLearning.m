close all

%% Initialize variables.
filename = '/Users/carriemcclanahan/Documents/ECE 3100/ECE3100-project/OnlineNewsPopularity2.csv';
delimiter = ',';
startRow = 2;

%% Format for each line of text:
%   column1: text (%s)
%	column2: double (%f)
%   column3: double (%f)
%	column4: double (%f)
%   column5: double (%f)
%	column6: double (%f)
%   column7: double (%f)
%	column8: double (%f)
%   column9: double (%f)
%	column10: double (%f)
%   column11: double (%f)
%	column12: double (%f)
%   column13: double (%f)
%	column14: double (%f)
%   column15: double (%f)
%	column16: double (%f)
%   column17: double (%f)
%	column18: double (%f)
%   column19: double (%f)
%	column20: double (%f)
%   column21: double (%f)
%	column22: double (%f)
%   column23: double (%f)
%	column24: double (%f)
%   column25: double (%f)
%	column26: double (%f)
%   column27: double (%f)
%	column28: double (%f)
%   column29: double (%f)
%	column30: double (%f)
%   column31: double (%f)
%	column32: double (%f)
%   column33: double (%f)
%	column34: double (%f)
%   column35: double (%f)
%	column36: double (%f)
%   column37: double (%f)
%	column38: double (%f)
%   column39: double (%f)
%	column40: double (%f)
%   column41: double (%f)
%	column42: double (%f)
%   column43: double (%f)
%	column44: double (%f)
%   column45: double (%f)
%	column46: double (%f)
%   column47: double (%f)
%	column48: double (%f)
%   column49: double (%f)
%	column50: double (%f)
%   column51: double (%f)
%	column52: double (%f)
%   column53: double (%f)
%	column54: double (%f)
%   column55: double (%f)
%	column56: double (%f)
%   column57: double (%f)
%	column58: double (%f)
%   column59: double (%f)
%	column60: double (%f)
%   column61: double (%f)
%   column62: text (%s)
%   column63: text (%s)
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
X = OnlineNewsPopularityData(:,[8 10 11 19 29 30 31 45 54]);
m = size(X,1);
X0 = (ones(m, 1));
X = [X0 table2array(OnlineNewsPopularityData(:,8)) table2array(OnlineNewsPopularityData(:,10))...
    table2array(OnlineNewsPopularityData(:,11)) table2array(OnlineNewsPopularityData(:,19))...
    table2array(OnlineNewsPopularityData(:,29)) table2array(OnlineNewsPopularityData(:,30))...
    table2array(OnlineNewsPopularityData(:,31)) table2array(OnlineNewsPopularityData(:,45))...
    table2array(OnlineNewsPopularityData(:,54))];
y = strcmp(OnlineNewsPopularityData.low_high_shares ,'high');
%[X,i] = unique(X,'rows');   % remove duplicates (care!: sorted results!!!)
%y = y(i);

rng(10)
pct = 0.7;
m = size(X,1);
%i = randperm(m)';
m = round(pct*m);
i = randperm(m)';
X = X(i,:);
y = y(i);
%tabulate(y)
maxiter = 10000;
mindJ = 1e-10;

%% TRAINING CLASSIFIER
threshold = 0.5;
lambda = 0;
T = 1e-4*rand(10,1);

%% gradient descent
% gradient params
alpha = 2;
lambda = 0;
dJ = mindJ;
J = zeros(maxiter,1);
h = sigmoid(X*T);
niter = 1;
J(niter) = -(1/m)*sum(y.*log(h)+(1-y).*log(1 - h)) + (lambda/(2*m))*sum(T(2:end).^2);
% gradient descent
while (niter < maxiter && abs(dJ) >= mindJ)
    % gradient step
    niter = niter + 1;
    grad = (1/m)*(X'*(h - y)) + [0; (lambda/m)*T(2:end)];
    T = T - alpha * grad;
    h = sigmoid(X*T);
    J(niter) = -(1/m)*sum(y.*log(h)+(1-y).*log(1 - h)) + (lambda/(2*m))*sum(T(2:end).^2);
    dJ = J(niter - 1) - J(niter);
end

% display params
niter
T'
output = sigmoid(X*T);
output(output>=0.5) = 1;
output(output<0.5) = 0;
missed = sum(abs(output-y))


%% FUNCTION TO COMPUTE COST AND GRADIENT
function [J, grad] = cost(T,X,y,lambda)
    m = size(X,1);
    h = sigmoid(X*T);
    J = -(1/m)*sum(y.*log(h)+(1-y).*log(1 - h)) + (lambda/(2*m))*sum(T(2:end).^2);
    grad = (1/m)*(X'*(h - y)) + [0; (lambda/m)*T(2:end)];
end
%% SIGMOID FUNCTION
function f = sigmoid (x)
    f = 1./(1+exp(-x));
end
%% FEATURE SCALING AND MEAN NORMALIZATION

%%

%features to test first are self_reference_avg_shares (31), num_videos (11), 
%self_reference_min_shares (29), data_channel_is_world (19), 
%self_reference_max_shares (30), num_hrefs (8), num_imgs (10), 
%avg_negative polarity (54), global subjectivity  (45)
%LDA_03 and LDA_02 are top but we don't think they make sense

%Predict y = 0 for Low shares (1 - 3000)
%Predict y = 1 for High shares (3001 - 843,300)


%% MULTICLASS CLASSIFICATION

%% LowMidHighShares variable
% Make a new variable that is 1 - 1500 for low shares 
% and 1501 - 5,000 for mid shares 
% and 5,001 - 843,000 for high shares
