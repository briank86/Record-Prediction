# Record-Prediction

The python file is the script I used to try and predict the final record of MLB teams using there team stats up to that point. I used Random Forest Regressor and XGBoost to predict an actual number and I ran Random Forest Classifier to predict if the team would win 81 or more games. 

I originially scraped data from ESPN so I could practice that skill but I was eventually blocked from doing so. Therefore I left my code for that at the top of the file but commented out. I used R to get the same data from Fangraphs then transferred it over to this file. The data it was trained and tested on was from 2002 up to 2022 but it excludes the data from 2020 due to the shortened season. Also in order to fit mid season stats to predict the final outcome I had to average all of the counting stats. I made a seperate file for the 2023 data to make it easier to see what the model would predict for just this season. I also made a dataframe that would print out the results next to the team. At the end of each predictor I put the feature importance to see what had the most effect and optimized the model by trying to find the best hyperparameters. I used a data correlation heat map to limit covariation of the terms. 

RESULTS:

The inital results from the ESPN data were not super accurate. The model loved all of the pitching data compared to the hitting data. It weighed ERA by far the most at around 40%. It wasn't terribly far off on who the good teams were but it didn't predict anyone to win over 94 games. 

Currently I'm working through a bug on the R file where I can no longer get data from Fangraphs and therefore the Python file won't run. I will update this with the actually printed results when that is resolved. 
