Raw votes data obtained from https://voteview.com/data, member IDs were ICPSR, whereas we use BioGuide ID (bid) in all our data so a mapping is created and provided (`icpsr_to_bid.pkl`). 

If using this data, cite: **Lewis, Jeffrey B., Keith Poole, Howard Rosenthal, Adam Boche, Aaron Rudkin, and Luke Sonnet (2020). Voteview: Congressional Roll-Call Votes Database. https://voteview.com/.**

Script to convert raw votes data (in `raw/`) to processed data (in `clean/`): `../../setup/preprocess_house_congs_votes.py`.

To learn vote ideal points for House Reps (Congresses 115-116) using variational inference: `../../setup/vote_ideal_points_our_data.py`.