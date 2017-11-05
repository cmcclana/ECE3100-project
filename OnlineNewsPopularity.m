%% Import data from text file.

%% Initialize variables.
filename = '/Users/carriemcclanahan/Documents/ECE 3100/OnlineNewsPopularity.csv';
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
% For more information, see the TEXTSCAN documentation.
formatSpec = '%s%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%[^\n\r]';

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
OnlineNewsPopularityData = table(dataArray{1:end-1}, 'VariableNames', {'url','timedelta','n_tokens_title','n_tokens_content','n_unique_tokens','n_non_stop_words','n_non_stop_unique_tokens','num_hrefs','num_self_hrefs','num_imgs','num_videos','average_token_length','num_keywords','data_channel_is_lifestyle','data_channel_is_entertainment','data_channel_is_bus','data_channel_is_socmed','data_channel_is_tech','data_channel_is_world','kw_min_min','kw_max_min','kw_avg_min','kw_min_max','kw_max_max','kw_avg_max','kw_min_avg','kw_max_avg','kw_avg_avg','self_reference_min_shares','self_reference_max_shares','self_reference_avg_sharess','weekday_is_monday','weekday_is_tuesday','weekday_is_wednesday','weekday_is_thursday','weekday_is_friday','weekday_is_saturday','weekday_is_sunday','is_weekend','LDA_00','LDA_01','LDA_02','LDA_03','LDA_04','global_subjectivity','global_sentiment_polarity','global_rate_positive_words','global_rate_negative_words','rate_positive_words','rate_negative_words','avg_positive_polarity','min_positive_polarity','max_positive_polarity','avg_negative_polarity','min_negative_polarity','max_negative_polarity','title_subjectivity','title_sentiment_polarity','abs_title_subjectivity','abs_title_sentiment_polarity','shares'});

%% Clear temporary variables
clearvars filename delimiter startRow formatSpec fileID dataArray ans;

%%
shares = OnlineNewsPopularityData{:,61};

%mean = 3,395.4
mean(shares);
%median = 1400
median(shares);
%max = 843,300
max(shares);
%min = 1
min(shares);
%std = 11,627
std(shares);
%variance = 135,190,000
var(shares);



%Monday = 32, Tuesday=33,...,Sunday = 38
% Monday = OnlineNewsPopularityData{:,32};
% Tuesday = OnlineNewsPopularityData{:,33};
% Wednesday = OnlineNewsPopularityData{:,34};
% Thursday = OnlineNewsPopularityData{:,35};
% Friday = OnlineNewsPopularityData{:,36};
% Saturday = OnlineNewsPopularityData{:,37};
% Sunday = OnlineNewsPopularityData{:,38};

% plot(Monday, shares,'o')

%plot kw_avg_avg vs shares
kw_avg_avg = OnlineNewsPopularityData{:,28};
figure(1)
plot(kw_avg_avg, shares, 'o', 'MarkerSize', 5)
xlabel('Avg. Shares for Avg. Keyword')
ylabel('Number of Shares')
ylim([0 5*10^4])
xlim([0 1*10^4])

%plot is_weekend vs shares



%to standardize so number of days in week vs weekend has no effect on data,
%divide shares for (is_weekend = 1) by 2 and (is_weekend = 0) by 5
% for c = 1:numel(shares)
%     if is_weekend(c) == 1
%         shares(c) = shares(c)/2;
%     else
%         shares(c) = shares(c)/5;
%     end
% end
is_weekend = OnlineNewsPopularityData{:,39};
figure(2)
boxplot(shares, is_weekend)
xlabel('Article published on weekend')
ylabel('Number of Shares')


%plot rate_negative_words vs shares
rate_negative_words = OnlineNewsPopularityData{:,50};
figure(3)
plot(rate_negative_words, shares, 'or', 'MarkerSize', 5)
xlabel('Rate of negative words among non-neutral tokens')
ylabel('Number of Shares')
ylim([0 4*10^4])

%plot rate_positive_words vs shares
rate_positive_words = OnlineNewsPopularityData{:,49};
figure(4)
plot(rate_positive_words, shares, 'og', 'MarkerSize', 5)
xlabel('Rate of positive words among non-neutral tokens')
ylabel('Number of Shares')
ylim([0 4*10^4])


%global_rate_negative words is 48

%look at all variables against shares. Careful, it is a lot of plots!
% for c = 3:60
%     figure(c)
%     plot(OnlineNewsPopularityData{:,c}, shares, 'om', 'MarkerSize', 5)
% end
%     

% c = [3,4,8,9,10,11,12,13,14,15,17,21,22,23,27,28,29,30,31,34,36,37,38,39,42,45,46,47,48,49,51,52,54,56]'
% for i = 1:numel(c)
%     figure(c(i))
%     plot(OnlineNewsPopularityData{:,c(i)}, shares, 'om', 'MarkerSize', 5)
% end

%%
figure(5)

boxplot(shares)
ylabel('Number of Shares')
xlabel('All articles')
title('Shares for all articles')
ylim([0 8*10^3])

%%

