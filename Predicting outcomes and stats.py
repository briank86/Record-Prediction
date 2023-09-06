# -*- coding: utf-8 -*-
"""
@author: brkea
"""

#Predicting if 2023 teams are over .500 using hitting and pitching data (up to this point in the season)
#Also predicting what 2023 teams record will be based on there stats up to this point in the season


import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn import metrics

import seaborn as sns
import matplotlib.pyplot as plt


Jump to line 220

#making the data frame - hitting data; skipping 2019 and 2021 because of missing data and 2020 for the shortened season 
# pred_df3 = pd.DataFrame()
# df3 = pd.DataFrame()
# pred_df = pd.DataFrame()
# final_df = pd.DataFrame()
# for i in reversed([x for x in range(2002, 2024) if x != 2019 and x !=2020 and x != 2021]):
#     url = 'https://www.espn.com/mlb/stats/team/_/season/{}/seasontype/2'.format(i)

#     page = requests.get(url)
#     soup = BeautifulSoup(page.text, 'html.parser')
                     
    
#     teams_stats = soup.find_all('tr', attrs={'data-idx':re.compile('')})
#     for team in teams_stats:
#         stats = [stat.get_text() for stat in team.find_all('td')]
    
#         temp_df = pd.DataFrame(stats).transpose()
        
#         if i != 2023:
#             final_df = pd.concat([final_df, temp_df], ignore_index=True)
#         if i == 2023:
#             pred_df = pd.concat([pred_df, temp_df], ignore_index=True)
    
#     #Get rid of NaN values
#     final_df.dropna(inplace=True)
#     pred_df.dropna(inplace=True)
    
#     #getting the id, name, and rank of the teams so that the pitching and hitting tables can be joined correctly
#     if i != 2023:
#         script = soup.find_all('script', attrs={'type':"text/javascript"})[2].get_text()
#         script = script[23:-1]
#         data = json.loads(script)
#         df = pd.json_normalize(data['page']['content']['teamStats'])
#         df2 = df[['team.id', 'team.shortDisplayName', 'team.rank']].copy()
#         df3 = pd.concat([df3, df2], ignore_index = True)
        
    
#     if i == 2023:
#         script = soup.find_all('script', attrs={'type':"text/javascript"})[2].get_text()
#         script = script[23:-1]
#         data = json.loads(script)
#         df = pd.json_normalize(data['page']['content']['teamStats'])
#         df2 = df[['team.id', 'team.shortDisplayName', 'team.rank']].copy()
#         pred_df3 = pd.concat([pred_df3, df2], ignore_index = True)
        


# #creating column names 
# columns = []
# header = soup.find_all('th', attrs={'class': 'Table__TH'}) 
# for column in header:
#     columns.append(column.get_text()) 
    
# #Get rid of rank and team columns as this was done in a seperate data frame above
# columns.pop(0)
# columns.pop(0)
    
# #add column names to a dictionary to easily rename columns of final_df
# dictionary = {}
# for i in range(len(columns)):
#     dictionary.update({i:columns[i]})

# #rename columns
# final_df.rename(columns=dictionary, inplace = True)
# pred_df.rename(columns=dictionary, inplace = True)

# final_df.reset_index(inplace = True)
# pred_df.reset_index(inplace = True)

# final_df = pd.concat([df3, final_df], axis = 1)
# pred_df = pd.concat([pred_df3, pred_df], axis = 1)


# #making the data frame - pitching
# pred_df3_pitching = pd.DataFrame()
# df3_pitching = pd.DataFrame()
# pred_df_pitching = pd.DataFrame() 
# final_df_pitching = pd.DataFrame()
# for i in reversed([x for x in range(2002, 2024) if x != 2019 and x !=2020 and x != 2021]):
#     url = 'https://www.espn.com/mlb/stats/team/_/view/pitching/season/{}/seasontype/2'.format(i)

#     page = requests.get(url)
#     soup = BeautifulSoup(page.text, 'html.parser')


#     teams_stats = soup.find_all('tr', attrs={'data-idx':re.compile('')})
#     for team in teams_stats:
#         stats = [stat.get_text() for stat in team.find_all('td')]
    
#         temp_df = pd.DataFrame(stats).transpose()
        
#         if i != 2023:
#             final_df_pitching = pd.concat([final_df_pitching, temp_df], ignore_index=True)
#         if i == 2023:
#             pred_df_pitching = pd.concat([pred_df_pitching, temp_df], ignore_index=True)

#     #Get rid of NaN values
#     final_df_pitching.dropna(inplace=True)
#     pred_df_pitching.dropna(inplace=True)
    
#     #getting the id, name, and rank of the teams so that the pitching and hitting tables can be joined correctly
#     if i != 2023:
#         script = soup.find_all('script', attrs={'type':"text/javascript"})[2].get_text()
#         script = script[23:-1]
#         data = json.loads(script)
#         df = pd.json_normalize(data['page']['content']['teamStats'])
#         df2 = df[['team.id', 'team.shortDisplayName', 'team.rank']].copy()
#         df3_pitching = pd.concat([df3_pitching, df2], ignore_index = True)
     
    
#     if i == 2023:
#         script = soup.find_all('script', attrs={'type':"text/javascript"})[2].get_text()
#         script = script[23:-1]
#         data = json.loads(script)
#         df = pd.json_normalize(data['page']['content']['teamStats'])
#         df2 = df[['team.id', 'team.shortDisplayName', 'team.rank']].copy()
#         pred_df3_pitching = pd.concat([pred_df3_pitching, df2], ignore_index = True)
        
    


# #creating column names 
# columns = []
# header = soup.find_all('th', attrs={'class': 'Table__TH'}) 
# for column in header:
#     columns.append(column.get_text()) 
    
# #Get rid of rank and team columns as this was done in a seperate data frame above
# columns.pop(0)
# columns.pop(0)
    
# #add column names to a dictionary to easily rename columns of final_df
# dictionary = {}
# for i in range(len(columns)):
#     dictionary.update({i:columns[i]})
    

# #change duplicate name columns to different names
# dictionary[0] = 'GP-pitch'

# dictionary[9] = 'H-pitch'

# dictionary[11] = 'HR-pitch'

# dictionary[12] = 'BB-pitch'

# dictionary[13] = 'SO-pitch'


# #rename columns
# final_df_pitching.rename(columns=dictionary, inplace = True)
# pred_df_pitching.rename(columns=dictionary, inplace = True)

# final_df_pitching.reset_index(inplace=True)
# pred_df_pitching.reset_index(inplace=True)

# final_df_pitching = pd.concat([df3_pitching, final_df_pitching], axis = 1)
# pred_df_pitching = pd.concat([pred_df3_pitching, pred_df_pitching], axis = 1)


# #Combine the data frames
# final_df_pitching.set_index('team.shortDisplayName', inplace = True)
# final_df_pitching.sort_index(axis=0, ascending=True, inplace=True)


# final_df.set_index('team.shortDisplayName', inplace = True)
# final_df.sort_index(axis=0, ascending=True, inplace=True)

# final_df = pd.concat([final_df, final_df_pitching], axis=1)
# final_df.drop(columns=['index', 'team.id', 'team.rank'], inplace = True)




# pred_df_pitching.set_index('team.shortDisplayName', inplace = True)
# pred_df_pitching.sort_index(axis=0, ascending=True, inplace=True)

# pred_df.set_index('team.shortDisplayName', inplace = True)
# pred_df.sort_index(axis=0, ascending=True, inplace=True)


# pred_df = pd.concat([pred_df, pred_df_pitching], axis=1)
# pred_df.drop(columns=['index', 'team.id', 'team.rank'], inplace = True)






#transfer the data from the R file
#Starts in 2002 up to 2022. It excludes only 2020.

final_df_pitching = pd.read_csv('C:\\Users\\brkea\\Desktop\\espn_team_pitching.csv')
final_df = pd.read_csv('C:\\Users\\brkea\\Desktop\\espn_team_hitting.csv')

pred_df_pitching = pd.read_csv('C:\\Users\\brkea\\Desktop\\espn_team_hitting_2023.csv')
pred_df = pd.read_csv('C:\\Users\\brkea\\Desktop\\espn_team_pitching_2023.csv')



final_df.dropna(axis = 1, inplace=True)
final_df_pitching.dropna(axis = 1,inplace=True)

pred_df.dropna(axis = 1,inplace=True)
pred_df_pitching.dropna(axis = 1,inplace=True)



#Combine the data frames
final_df_pitching.set_index('Team', inplace = True)
final_df_pitching.sort_index(axis=0, ascending=True, inplace=True)


final_df.set_index('Team', inplace = True)
final_df.sort_index(axis=0, ascending=True, inplace=True)

final_df = pd.concat([final_df, final_df_pitching], axis=1)




pred_df_pitching.set_index('Team', inplace = True)
pred_df_pitching.sort_index(axis=0, ascending=True, inplace=True)

pred_df.set_index('Team', inplace = True)
pred_df.sort_index(axis=0, ascending=True, inplace=True)


pred_df = pd.concat([pred_df, pred_df_pitching], axis=1)



#Export to projects folder
final_df.to_csv(r'C:\Users\brkea\OneDrive\Baseball Projects\MLB_Season_Stats.csv', index=False, sep=',', encoding='utf-8')





# Load data
data = pd.read_csv('MLB_Season_Stats.csv')

#Column saying if the team is at or over 81 wins column - 1 yes, 0 No
above_81 = []
for i in range(len(data['W'])):
    if int(data._get_value(i, 'W')) >= 81:
        above_81.append(1)
    if int(data._get_value(i, 'W')) < 81:
        above_81.append(0)

# #data.insert(0, 'Teams', teams_list)
data.insert(0, 'Over 81 Wins', above_81)

#Drop not needed columns based on correlation and other not needed factors
#fangraphs didn't have OBA so don't drop whip
data.drop(columns=[ 'GP-pitch', '3B', 'AB', 'SB', '2B', 'L', 'SV', 'CG', 'SHO', 'IP', 'ER', 'H', 'RBI', 'SLG', 'TB', 'H-pitch', 
                    'R', 'OBP'], inplace = True)
pred_df.drop(columns=['GP-pitch', '3B', 'AB', 'SB', '2B', 'W', 'L', 'SV', 'CG', 'SHO', 'IP', 'ER', 'H', 'RBI','SLG', 'TB', 'H-pitch',
                      'R', 'OBP'], inplace = True)

#Save wins column in a data frame so it can be added back into the regression
wins = data['W']



# Save preprocessed data
data.to_csv(r'C:\Users\brkea\OneDrive\Baseball Projects\preprocessed_data.csv', index=False, sep=',', encoding='utf-8')
pred_df.to_csv(r'C:\Users\brkea\OneDrive\Baseball Projects\2023_processed_data.csv', index=False, sep=',', encoding='utf-8')





# Load preprocessed data
data = pd.read_csv('preprocessed_data.csv')


#Getting rid of commas so that it could be converted to a float

for i in range(len(data['SO'])):
    string = data._get_value(i, 'SO')
    string = string.replace(',', '')
    data.replace(data._get_value(i, 'SO'), string, inplace = True)

data = data.astype({'SO':'int'})
    
for i in range(len(data['SO-pitch'])):
    string = data._get_value(i, 'SO-pitch')
    string = string.replace(',', '')
    data.replace(data._get_value(i, 'SO-pitch'), string, inplace = True)

data = data.astype({'SO-pitch':'int'})

    
    

#average values so that it can be used to predict final wins when in the middle of the season
def avgValue(ColName):
    for i in range(len(data[ColName])):
        data[ColName + 'avg'] = data[ColName]/data['GP']

avgValue('HR')
avgValue('BB')
avgValue('SO')
avgValue('QS')
avgValue('HR-pitch')
avgValue('BB-pitch')
avgValue('SO-pitch')



data.drop(columns=['GP', 'HR', 'BB', 'SO', 'QS', 'HR-pitch', 'BB-pitch', 'SO-pitch'], inplace = True)

pred = pd.read_csv('2023_processed_data.csv')   
    


#average values so that it can be used to predict final wins when in the middle of the season
def avgValue(ColName):
    for i in range(len(pred[ColName])):
        pred[ColName + 'avg'] = pred[ColName]/pred['GP']

avgValue('HR')
avgValue('BB')
avgValue('SO')
avgValue('QS')
avgValue('HR-pitch')
avgValue('BB-pitch')
avgValue('SO-pitch')


pred.drop(columns=['GP', 'HR', 'BB', 'SO', 'QS', 'HR-pitch', 'BB-pitch', 'SO-pitch'], inplace = True)
    
    
    
#Classifier    
print('Classifier')
data.drop(columns=['W'], inplace = True)
    
# Split data into features (X) and target (y)
X = data.drop('Over 81 Wins', axis = 1) 
y = data['Over 81 Wins']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model - classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)


# Make predictions and evaluate the model
y_pred = clf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:', classification_report(y_test, y_pred))



# Get feature importances
importances = clf.feature_importances_

# Create a DataFrame to display feature importances
feature_importances = pd.DataFrame({'feature': X.columns, 'importance': importances})
feature_importances = feature_importances.sort_values(by='importance', ascending=False)

# Display top features
print("Top Features Contributing to Game Outcomes:")
print(feature_importances.head(10))




# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                            param_grid=param_grid,
                            scoring='accuracy',
                            cv=5,
                            n_jobs=-1)

# Fit and find best hyperparameters
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Evaluate the best model
best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:', classification_report(y_test, y_pred))



#Make data frame of the teams and there projected finish
prediction = clf.predict(pred)
prediction = {'81 Wins':prediction}
teams = ['Angels', 'Astros', 'Athletics', 'Blue Jays', 'Braves', 'Brewers', 'Cardinals', 'Cubs', 'Diamondbacks', 'Dodgers', 'Giants', 'Guardians', 'Mariners', 'Marlins', 'Mets', 'Nationals', 'Orioles', 'Padres', 'Phillies', 'Pirates', 'Rangers', 'Rays', 'Red Sox', 'Reds', 'Rockies', 'Royals', 'Tigers', 'Twins', 'White Sox', 'Yankees']
prediction = pd.DataFrame(prediction)
prediction.insert(0, 'Teams', teams)
print(prediction)

#Make correlation heat map
plt.subplots(figsize=(16, 6))
heatmap = sns.heatmap(data.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Classifier Correlation', fontdict={'fontsize':18}, pad=12)


print('\n')
print('\n')
print('Logistic Regression')




#Logistic Regression   
    

# Split data into features (X) and target (y)
X = data.drop('Over 81 Wins', axis = 1) 
y = data['Over 81 Wins']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = logreg.predict(X_test)
print('Accuracy:', logreg.score(X_test, y_test))
print('Classification Report:', classification_report(y_test, y_pred))






print('\n')
print('\n')
print('Regression')




#Regression

data.drop(columns=['Over 81 Wins'], inplace = True)
data.insert(0, 'W', wins)


# Split data into features (X) and target (y)
X = data.drop('W', axis = 1)
y = data['W']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create and train the Random Forest Regression model
clf = RandomForestRegressor(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)


# Make predictions and evaluate the model
y_pred = clf.predict(X_test)

print(mean_squared_error(y_test,y_pred))
print(r2_score(y_test,y_pred))






# Get feature importances
importances = clf.feature_importances_

# Create a DataFrame to display feature importances
feature_importances = pd.DataFrame({'feature': X.columns, 'importance': importances})
feature_importances = feature_importances.sort_values(by='importance', ascending=False)

# Display top features
print("Top Features Contributing to Game Outcomes:")
print(feature_importances.head(10))







# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                            param_grid=param_grid,
                            scoring='r2',
                            cv=5,
                            n_jobs=-1)

# Fit and find best hyperparameters
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Evaluate the best model
best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test)
print(mean_squared_error(y_test,y_pred))
print(r2_score(y_test,y_pred))





#Make data frame of the teams and there projected finish
prediction = best_clf.predict(pred)
prediction = {'Wins':prediction}
prediction = pd.DataFrame(prediction)
prediction.insert(0, 'Teams', teams)
print(prediction)



#used correlation heat map to figure out how the variables related to ensure low mulit-collinearity

#make correlation heat map
plt.subplots(figsize=(16, 6))
heatmap = sns.heatmap(data.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Regression Correlation', fontdict={'fontsize':18}, pad=12)



#XGB Regressor

# Split data into features (X) and target (y)
X = data.drop('W', axis = 1)
y = data['W']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create and train the XGBoost model
clf = xgb.XGBRegressor() #n_estimators=100, random_state=42
clf.fit(X_train, y_train)


# Make predictions and evaluate the model
y_pred = clf.predict(X_test)

print(mean_squared_error(y_test,y_pred))
print(r2_score(y_test,y_pred))
