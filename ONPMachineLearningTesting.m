close all

%% Initialize variables.
filename = '/Users/carriemcclanahan/Documents/ECE 3100/ECE3100-project/OnlineNewsPopularity2.csv';
%filename = 'C:\Users\David Wade\Documents\1a School Files\2017 Fall\ECE 3100\ECE3100-project-master\OnlineNewsPopularity2.csv';
delimiter = ',';
startRow = 2;
formatSpec = '%s%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%s%s%s%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string', 'EmptyValue', NaN, 'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');

%% Close the text file.
fclose(fileID);

%% Create output variable
ONPD = table(dataArray{1:end-1}, 'VariableNames', {'url','timedelta','n_tokens_title','n_tokens_content','n_unique_tokens','n_non_stop_words','n_non_stop_unique_tokens','num_hrefs','num_self_hrefs','num_imgs','num_videos','average_token_length','num_keywords','data_channel_is_lifestyle','data_channel_is_entertainment','data_channel_is_bus','data_channel_is_socmed','data_channel_is_tech','data_channel_is_world','kw_min_min','kw_max_min','kw_avg_min','kw_min_max','kw_max_max','kw_avg_max','kw_min_avg','kw_max_avg','kw_avg_avg','self_reference_min_shares','self_reference_max_shares','self_reference_avg_sharess','weekday_is_monday','weekday_is_tuesday','weekday_is_wednesday','weekday_is_thursday','weekday_is_friday','weekday_is_saturday','weekday_is_sunday','is_weekend','LDA_00','LDA_01','LDA_02','LDA_03','LDA_04','global_subjectivity','global_sentiment_polarity','global_rate_positive_words','global_rate_negative_words','rate_positive_words','rate_negative_words','avg_positive_polarity','min_positive_polarity','max_positive_polarity','avg_negative_polarity','min_negative_polarity','max_negative_polarity','title_subjectivity','title_sentiment_polarity','abs_title_subjectivity','abs_title_sentiment_polarity','shares', 'low_high_shares', 'low_mid_high_shares', 'low_high_shares2'});

%% Clear temporary variables
clearvars filename delimiter startRow formatSpec fileID dataArray ans;

%% Graphing variables vs. shares, and vice versa (no modifications)

% MODIFY 'FOR' LOOP INDEX

for n=3:60
   % Next three lines are variable vs. shares
   scatter(cell2mat(table2cell(ONPD(:,n))),ONPD.shares)
   l1 = ylabel(ONPD.Properties.VariableNames(61), 'Interpreter', 'none');
   l2 = xlabel(cell2mat(ONPD.Properties.VariableNames(n)), 'Interpreter', 'none');
   
   % Next three lines are shares vs. variable
   %scatter(ONPD.shares,cell2mat(table2cell(ONPD(:,n))))
   %l1 = xlabel(ONPD.Properties.VariableNames(61), 'Interpreter', 'none');
   %l2 = ylabel(cell2mat(ONPD.Properties.VariableNames(n)), 'Interpreter', 'none');
   
   w = waitforbuttonpress;
   if (w == 1) || (w == 0)
       w=3;
       continue
   end
end

%% Graphing variables vs. shares, and vice verse (modified to account for "weird" variable types)