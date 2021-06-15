# Legislator Information and Text-based Ideal Point Values

**legislator_info_and_tbip_congresses_115_and_116.csv**: The main csv file with legislator information and TBIP values; information for the legislator spans a lot of information for their district, caucus memberships for the legislator, etc.

**codebook.txt**: Explains the columns/variables present in the above file.

**supporting_data_files/**: This contains the data files used to derive information for constructing legislator_info_and_tbip_congresses_115_and_116.csv - everything except the TBIP model derived values comes from these files.

**speeches_results/**: Results from TBIP on floor speeches (topics, topic proportions in various texts including the raw text, mean topic proportions for every legislator).

**tweets_results/**: Results from TBIP on Twitter tweets (topics, topic proportions in various texts including the raw text, mean topic proportions for every legislator).

**tbip/**: TBIP code, with commands used to run included, as well as the script for running issue-specific tbip. Includes data files, code to get from raw data to clean data TBIP code can use, and analysis code that creates the result files. 
