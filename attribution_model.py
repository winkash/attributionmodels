import pandas as pd
from collections import Counter
import numpy as np

# training file
train_file = "test_data.csv"
# original file in excel
conv_file = 'c:\\ashwin\\Customerattributiondata_1.csv'
train_data = pd.read_csv(train_file)


# Single-touch attribution models assign 100% of carrier credit to one marketing channel, disregarding any number of
# channels visited by a user. This is a naïve approach and has a low level of complexity. These models should only be
# used when there are five or less touch points in a customer's journey, the preferred of which is the Last-Touch
# Attribution Model.
def last_touch_model(train_data, conv_col, channel_col):
    last_touch = train_data
    res_last_touch = pd.DataFrame(round(last_touch[channel_col].value_counts(normalize=True) * 100, 2),)
    res_last_touch.columns = ['Weights (%)']
    return res_last_touch


# This model ignores direct traffic, and it assigns 100% of the credit to the last channel the user engaged with before
# making a conversion. This is best used if one wants to understand effectiveness of their final marketing activities
# without direct traffic getting in the way of their analysis. However, this is still ignoring the other campaigns and
# channels with which the customer interacted.
def last_non_direct_model(train_data, conv_col, channel_col, session_id):
    second_last = train_data
    tmp = second_last
    last_non_direct = pd.DataFrame(second_last.groupby(session_id).first(), index=second_last[session_id])
    cookie_index = list(tmp[session_id])
    mid_last_non_direct = last_non_direct.loc[cookie_index]
    res_last_non_direct = pd.DataFrame(round(mid_last_non_direct[channel_col].value_counts(normalize=True) * 100, 2))
    res_last_non_direct.columns = ['Weights (%']
    return res_last_non_direct


# if one is mainly focused on widening [the] top of [their] funnel, this is a useful model. It highlights the
# channels that first introuced a customer to [the] brand.
def first_touch_model(train_data, conv_col, channel_col, session_id):
    tmp = train_data
    first_touch = pd.DataFrame(train_data.groupby(session_id).first(), index=train_data[session_id])
    cookie_index = list(tmp[session_id])
    mid_first_touch = first_touch.loc[cookie_index]
    res_first_touch = pd.DataFrame(round(mid_first_touch[channel_col].value_counts(normalize=True) * 100, 2))
    res_first_touch.columns = ['Weights (%)']
    return res_first_touch


# Multi-touch models assume that all touchpoints play some role in driving a conversion. We look at the most popular
# multi-touch attribution models which include Linear, Position decay, and U-shaped models. These models have become
# important for marketers, particularly those looking to measure the impact of digital campaigns, since they provide a
# more granular and person-level view than traditional aggregation methods. These models should be used when the number
# of channels for a campaign is 5 to 10, the most preferable of which are position-based, like the U-shaped Attribution
# Model.
# The Linear Attribution Model gives each touchpoint across the buyer journey the same amount of credit toward driving
# a sale; it values every touchpoint evenly. This model is easy to implement and is better than all the single-touch
# attribution models. The disadvantage here is that in reality, consumers aren't equally impacted by every kind of
# channel.
def linear_model(train_data, conv_col, channel_col, session_id):
    cookie_index = list(train_data[session_id])
    train_data['new'] = train_data['SESSIONID'].isin(cookie_index)
    y = train_data['new'].isin([True])
    train_conv = train_data[y]
    tmp = pd.DataFrame(train_conv.groupby(session_id).tail(1))
    x = Counter(train_conv[session_id])
    tmp['usr_click_count'] = x.values()
    tmp.set_index(session_id, inplace=True)
    train_conv['clicks'] = train_conv[session_id].map(x)
    train_conv = train_conv.assign(clicks_per=lambda x: round(100 / train_conv['clicks'], 2))
    res_linear = train_conv.groupby(channel_col, as_index=False)['clicks_per'].mean()
    _sum = res_linear['clicks_per'].sum()
    res_linear['Weight (%)'] = res_linear.apply(lambda x: round((x['clicks_per'] / _sum) * 100, 2), axis=1)
    res_linear.drop(['clicks_per'], inplace=True, axis=1)
    res_linear = res_linear.set_index(channel_col)
    res_linear.index_name = None
    return res_linear.sort_values(by='Weight (%)', ascending=False)


df = pd.read_csv(conv_file, sep='\t', error_bad_lines=False)
header = ["CUSTOMERID",	"SESSIONID", "TIMESTAMP_TOUCHPOINT", "MARKETINGCHANNEL", "REVENUE"]
lines_to_csv = [header]
for index, row in df.iterrows():
    line = row['CUSTOMERID\tSESSIONID\tTIMESTAMP_TOUCHPOINT\tMARKETINGCHANNEL\tREVENUE']
    lines = line.split("\t")
    if len(lines) < 5:
        continue
    lines_to_csv.append(lines)

np.savetxt("C:\\Users\ASHWI\\PycharmProjects\\pythonProject\\deeplearning\\test_data.csv", lines_to_csv, delimiter =",",fmt ='% s')

last_touch = last_touch_model(train_data, 'CONVERSION', 'MARKETINGCHANNEL')
last_touch = last_non_direct_model(train_data, 'CONVERSION', 'MARKETINGCHANNEL', 'SESSIONID')
first_touch = first_touch_model(train_data, 'CONVERSION', 'MARKETINGCHANNEL', 'SESSIONID')
linear = linear_model(train_data, 'CONVERSION', 'MARKETINGCHANNEL', 'SESSIONID')


