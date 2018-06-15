
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from fastai.structured import *
from fastai.column_data import *
from fastai.imports import *
from fastai.dataset import *
from fastai.torch_imports import *
from fastai.column_data import *
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder , Imputer , StandardScaler
import operator
import os
np.set_printoptions(threshold=50,edgeitems=20)
PATH='data/airbnb/'


# In[2]:


from datetime import datetime , date


# In[3]:


get_ipython().system('ls data/airbnb')


# In[4]:


table_names = ['train_users_2','sessions']
tables = [pd.read_csv(f'{PATH}{fname}.csv',low_memory=False) for fname in table_names]


# In[5]:


test = pd.read_csv(f'{PATH}test_users.csv')


# In[6]:


for t in tables : display(t.head())


# In[7]:


for t in tables : display(DataFrameSummary(t).summary())


# new

# In[8]:


train_users = pd.read_csv(f'{PATH}train_users_2.csv')
test_users = pd.read_csv(f'{PATH}test_users.csv')
sessions = pd.read_csv(f'{PATH}sessions.csv')
users = pd.concat([train_users, test_users], axis=0, ignore_index=True)
users.drop('date_first_booking', axis=1, inplace=True)   #test users doesnt have it
user_with_year_age_mask = users['age'] > 1000
users.loc[user_with_year_age_mask, 'age'] = 2015 - users.loc[user_with_year_age_mask, 'age']
users.loc[(users['age'] > 100) | (users['age'] < 18), 'age'] = -1   #set age limit
users['age'].fillna(-1,inplace=True)
bins = [-1, 20, 25, 30, 40, 50, 60, 75, 100]
users['age_group'] = np.digitize(users['age'], bins, right=True)


# In[9]:


# number of nans
users['nans'] = np.sum([
    (users['age'] == -1),
    (users['gender'] == '-unknown-'),
    (users['language'] == '-unknown-'),
    (users['first_affiliate_tracked'] == 'untracked'),
    (users['first_browser'] == '-unknown-')
], axis=0)


# In[10]:


#convert dates
users['date_account_created'] = pd.to_datetime(users['date_account_created'], errors='ignore')
users['date_first_active'] = pd.to_datetime(users['timestamp_first_active'], format='%Y%m%d%H%M%S')
date_account_created = pd.DatetimeIndex(users['date_account_created'])
date_first_active = pd.DatetimeIndex(users['date_first_active'])
#splitting into day month and year
users['day_account_created'] = date_account_created.day
users['weekday_account_created'] = date_account_created.weekday
users['week_account_created'] = date_account_created.week
users['month_account_created'] = date_account_created.month
users['year_account_created'] = date_account_created.year
users['day_first_active'] = date_first_active.day
users['weekday_first_active'] = date_first_active.weekday
users['week_first_active'] = date_first_active.week
users['month_first_active'] = date_first_active.month
users['year_first_active'] = date_first_active.year
#time lag
users['time_lag'] = (date_account_created.values - date_first_active.values).astype(int)
#drop duplicated columns
drop_list = [
    'date_account_created',
    'date_first_active',
    'timestamp_first_active'
]

users.drop(drop_list, axis=1, inplace=True)


# In[11]:


#work on sessions
sessions.rename(columns = {'user_id': 'id'}, inplace=True)   #rename
#finding frequency
action_count = sessions.groupby(['id', 'action'])['secs_elapsed'].agg(len).unstack()
action_type_count = sessions.groupby(['id', 'action_type'])['secs_elapsed'].agg(len).unstack()
action_detail_count = sessions.groupby(['id', 'action_detail'])['secs_elapsed'].agg(len).unstack()
device_type_sum = sessions.groupby(['id', 'device_type'])['secs_elapsed'].agg(sum).unstack()

sessions_data = pd.concat([action_count, action_type_count, action_detail_count, device_type_sum],axis=1)
sessions_data.columns = sessions_data.columns.map(lambda x: str(x) + '_count')

# Most used device
sessions_data['most_used_device'] = sessions.groupby('id')['device_type'].max()

users = users.join(sessions_data, on='id')

#secs elapsed
secs_elapsed = sessions.groupby('id')['secs_elapsed']

secs_elapsed = secs_elapsed.agg(
    {
        'secs_elapsed_sum': np.sum,
        'secs_elapsed_mean': np.mean,
        'secs_elapsed_min': np.min,
        'secs_elapsed_max': np.max,
        'secs_elapsed_median': np.median,
        'secs_elapsed_std': np.std,
        'secs_elapsed_var': np.var,
        'day_pauses': lambda x: (x > 86400).sum(),
        'long_pauses': lambda x: (x > 300000).sum(),
        'short_pauses': lambda x: (x < 3600).sum(),
        'session_length' : np.count_nonzero
    }
)

users = users.join(secs_elapsed, on='id')


# In[12]:


#encode categorical features
categorical_features = [
    'gender', 'signup_method', 'signup_flow', 'language',
    'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked',
    'signup_app', 'first_device_type', 'first_browser', 'most_used_device'
]
users = pd.get_dummies(users, columns=categorical_features)
users.set_index('id', inplace=True)
users.loc[train_users['id']].to_csv(f'{PATH}train_users_f.csv')
users.loc[test_users['id']].drop('country_destination', axis=1).to_csv(f'{PATH}test_users_f.csv')


# In[13]:


joined = pd.read_csv(f'{PATH}train_users_f.csv')
joined_test = pd.read_csv(f'{PATH}test_users_f.csv')


# In[14]:


joined.head().T.head(100)


# In[15]:


joined_test.head(7).T.head(20)


# In[16]:


cat_vars = ['day_account_created','weekday_account_created','week_account_created','month_account_created','year_account_created',
           'day_first_active','weekday_first_active','week_first_active','month_first_active','year_first_active']
contin_vars = ['age','age_group','time_lag']
n=len(joined);n


# In[17]:


dep = 'country_destination'
joined = joined[cat_vars+contin_vars+[dep,'id']].copy()


# In[18]:


joined_test[dep]=0
joined_test = joined_test[cat_vars+contin_vars+[dep,'id']].copy()


# In[19]:


for v in cat_vars:
    joined[v]=joined[v].astype('category').cat.as_ordered()


# In[20]:


apply_cats(joined_test,joined)


# In[21]:


for v in contin_vars:
    joined[v]=joined[v].fillna(0).astype('float32')
    joined_test[v]=joined_test[v].fillna(0).astype('float32')


# In[22]:


samp_size = n
joined_samp = joined.set_index("id")


# In[23]:


joined_samp.head(2)


# In[25]:


df , y , nas , mapper = proc_df(joined_samp,'country_destination',do_scale= True)

