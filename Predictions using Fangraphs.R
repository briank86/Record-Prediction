library(ggplot2)
library(dplyr)
library(knitr)
library(baseballr)


#2002 is the first year of all of the data being included

#Fangraphs data I'm saving to do prediction work on advanced stats in the future 
#Jump to line 28 for this project's data

batterTeamData <- fg_team_batter(x= 2002, y = 2019)
batterTeamData <- rbind(batterTeamData, fg_team_batter(x=2021, y=2022))
pitcherTeamData <- fg_team_pitcher(x= 2002, y = 2019)
pitcherTeamData <- rbind(pitcherTeamData, fg_team_pitcher(x=2021, y=2022))

hitting2023 <- fg_team_batter(x= 2023, y = 2023)
pitching2023 <- fg_team_pitcher(x= 2023, y = 2023)

write.csv(batterTeamData, file = 'C:\\Users\\brkea\\Desktop\\fg_team_hitting.csv')
write.csv(pitcherTeamData, file = 'C:\\Users\\brkea\\Desktop\\fg_team_pitching.csv')

write.csv(hitting2023, file = 'C:\\Users\\brkea\\Desktop\\fg_team_hitting_2023.csv')
write.csv(pitching2023, file = 'C:\\Users\\brkea\\Desktop\\fg_team_pitching_2023.csv')




#Making data set for the python file that was designed for espn data
batterTeamData <- batterTeamData[,names(batterTeamData) %in% c("Team", "G", "AB", "R", "H", "2B", "3B", "HR", "RBI", "TB", "BB", "SO", "SB", "AVG", "OBP", "SLG", "OPS")]
pitcherTeamData <- pitcherTeamData[,names(pitcherTeamData) %in% c("Team", "G", "W", "L", "ERA", "SV", "CG", "ShO", "IP", "H", "ER", "HR", "BB", "SO", "WHIP")]
#doesn't have QS or OBA like espn

hitting2023 <- hitting2023[,names(hitting2023) %in% c("Team", "G", "AB", "R", "H", "2B", "3B", "HR", "RBI", "TB", "BB", "SO", "SB", "AVG", "OBP", "SLG", "OPS")]
pitching2023 <- pitching2023[,names(pitching2023) %in% c("Team","G", "W", "L", "ERA", "SV", "CG", "ShO", "IP", "H", "ER", "HR", "BB", "SO", "WHIP")]

pitcherTeamData <- pitcherTeamData %>% rename("G" = "GP-pitch", "H"="H-pitch", "HR" = "HR-pitch", "BB"="BB-pitch", "SO"="SO-pitch")
pitching2023 <- pitching2023 %>% rename("G" = "GP-pitch", "H"="H-pitch", "HR" = "HR-pitch", "BB"="BB-pitch", "SO"="SO-pitch")

  
write.csv(batterTeamData, file = 'C:\\Users\\brkea\\Desktop\\espn_team_hitting.csv')
write.csv(pitcherTeamData, file = 'C:\\Users\\brkea\\Desktop\\espn_team_pitching.csv')

write.csv(hitting2023, file = 'C:\\Users\\brkea\\Desktop\\espn_team_hitting_2023.csv')
write.csv(pitching2023, file = 'C:\\Users\\brkea\\Desktop\\espn_team_pitching_2023.csv')


