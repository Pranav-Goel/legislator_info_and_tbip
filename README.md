# Legislator Information and Text-based Ideal Point Values: Result Files, Replication Code, and Data for Creating the Main File used for all Analyses in our Work in: Express Yourself (Ideologically): Legislators' Ideal Points Across Audiences. 

NOTE: Due to file size limits in Github, certain files required to run codes are not present in this repository. Database with all files present can be found and downloaded from: https://zenodo.org/record/7641732

**legislator_info_and_tbip_congresses_115_and_116.csv**: The main csv file with legislator information and TBIP values; information for the legislator spans a lot of information for their district, caucus memberships for the legislator, etc -- this is the file used for al our analyses in the paper. 

**codebook.txt**: Explains the columns/variables present in the above file.

**supporting_data_files/**: This contains the data files used to derive information for constructing legislator_info_and_tbip_congresses_115_and_116.csv - everything except the TBIP model derived values comes from these files.

**speeches_results/**: Results from TBIP on floor speeches (topics, topic proportions in various texts including the raw text, mean topic proportions for every legislator).

**tweets_results/**: Results from TBIP on Twitter tweets (topics, topic proportions in various texts including the raw text, mean topic proportions for every legislator).

**tbip/**: TBIP code, with commands used to run included, as well as the script for running issue-specific tbip. Includes data files, code to get from raw data to clean data TBIP code can use, and analysis code that creates the result files. 

**combine_and_create_main_file_for_conducting_research.ipynb**: Code (Jupyter notebook) that uses result files (ideal point estimates) as well as supporting data files for legislator information and combines them to create the main csv file used to conduct analyses used in our research - legislator_info_and_tbip_congresses_115_and_116.csv. 
