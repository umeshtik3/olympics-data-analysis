# --------------
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(path)

data.columns

data.shape

data.rename(columns={'Total':'Total_Medals'},inplace=True)

data.head(10)

data[(data['Total_Winter']>0)& (data['Total_Summer']==0)]

data['Better_Event'] = np.where(data['Total_Summer'] > data['Total_Winter'] , 'Summer', 'Winter') 
data['Better_Event'] =np.where(data['Total_Summer'] ==data['Total_Winter'],'Both',data['Better_Event'])

# better_event = data['Better_Event'].value_counts()
data['Better_Event'].value_counts()
better_event = 'Summer'
  


# top_countries.drop(146,inplace=True)
top_countries = data[['Country_Name','Total_Summer', 'Total_Winter','Total_Medals']]
top_countries = top_countries[:-1]

def top_ten(df,col_na):
    """ Presenting the top ten conutries in the dataframe
    This function accepts the dataframe and column name
    
    keyword arguments 
    df - dataframe of country names and medals
    col_na  - column name is country_name and medals
    
    returns  - top 10 country list

    """
    country_list = []
    
    a = df.nlargest(10,col_na)
    country_list = list(a['Country_Name'])
    return country_list


top_10_summer = (top_ten(top_countries,'Total_Summer'))
print(top_10_summer)
top_10_winter = list(top_ten(top_countries,'Total_Winter'))

top_10 = top_ten(top_countries,'Total_Medals')


common = set(top_10).intersection(top_10_summer,top_10_winter)

top_df = data[data['Country_Name'].isin(top_10)]
summer_df = data[data['Country_Name'].isin(top_10_summer)]
winter_df = data[data['Country_Name'].isin(top_10_winter)]

summer_df.plot(kind='bar',x='Country_Name',y='Total_Summer')
plt.xlabel('Country Name')
plt.ylabel('Counts')
plt.title('Summer')
plt.show()

def bar_graph(df,col_1,col_2):
    """
    Plotting bar plot for above column
    parameters : 
    df -  dataframe
    col_1 -  column from dataframe at x axis
    col_2 - column from dataframe at y axis
    """
    df.plot(kind='bar',x=col_1,y=col_2)
    plt.xlabel(col_1)
    plt.ylabel('Counts')
    plt.title(col_1)
    plt.show()
    

bar_graph(summer_df,'Country_Name','Total_Summer')

bar_graph(winter_df,'Country_Name','Total_Winter')

bar_graph(top_df,'Country_Name','Total_Medals')

# summer_df['Gold_Ratio'] = (summer_df['Gold_Summer']/summer_df['Total_Summer'])

summer_df.columns

# gold_max = pd.DataFrame(summer_df.groupby(['Country_Name','Gold_Ratio'])['Gold_Summer'].count()).reset_index()

# g_max = summer_df['Gold_Ratio'].max()

# summer_df.loc[summer_df['Gold_Ratio']==g_max,'Country_Name'].iloc[0]

# gr = gold_max['Gold_Ratio'].max()

# gold_max.loc[gold_max['Gold_Ratio']==gr,'Country_Name'].iloc[0]

def gold_countries(df,df_type,c_name,g_type,total):
        """
        finding the countries which has highest gold ratio
        parameters:
        df : dataframe 
        col_1 : is the Gold ratio column
        col_2 : Country name in data frame
        """
        dataframes = {'Summer':summer_df,'Winter':winter_df,'Top':top_df}
        df1 = dataframes[df_type]
        df1['Gold_Ratio'] = round((df1[g_type]/df1[total])*100,3)
            
        gold_max_ratio = df1['Gold_Ratio'].max()
#         print(gold_max_ratio)

        country_gold =  df1.loc[ df1['Gold_Ratio']==gold_max_ratio,c_name].iloc[0]
        print(country_gold)
        return round(gold_max_ratio,2),country_gold

summer_gold_ratio, summer_country_gold = gold_countries(summer_df,'Summer','Country_Name','Gold_Summer','Total_Summer')

winter_gold_ratio, winter_country_gold = gold_countries(winter_df,'Winter','Country_Name','Gold_Winter','Total_Summer')


top_df.columns


top_max_ratio, top_country_gold = gold_countries(top_df,'Top','Country_Name','Gold_Total','Total_Medals')
top_max_ratio =0.40
data_1 = data.drop(146)

data_1.head()

data_1['Gold_Total'].map(lambda x : x*3)

def points(df):
    df['Gold_Points'] = df['Gold_Total'].map(lambda x : x*3)
    df['Silver_Points'] = df['Silver_Total'].map(lambda x : x*2)
    df['Bronze_Points'] = df['Bronze_Total']
    df['Total_Points'] = df['Gold_Points']+df['Silver_Points']+df['Bronze_Points']
    
    
    return df

points(data_1)

def best_country(df,col_1):
    max_p = df['Total_Points'].max()


    best_c =  df.loc[ df['Total_Points']==max_p,col_1].iloc[0]
    return max_p,best_c

most_points,best_country = best_country(data_1,'Country_Name')

most_points,best_country



best = data[ data['Country_Name']==best_country]

best =best[['Gold_Total','Silver_Total','Bronze_Total']]

best.plot(kind='bar',stacked=True)
plt.xlabel('United States')
plt.ylabel('Meadals Tally')
plt.xticks(rotation=45)
plt.show()


