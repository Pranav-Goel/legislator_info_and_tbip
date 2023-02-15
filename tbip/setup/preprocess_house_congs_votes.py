"""Preprocess votes for vote-based ideal points.

#### References
[1]: Lewis, Jeffrey B., Keith Poole, Howard Rosenthal, Adam Boche, 
     Aaron Rudkin, and Luke Sonnet (2020). Voteview: Congressional Roll-Call 
     Votes Database. https://voteview.com/

"""
import os
import numpy as np
import pandas as pd

from absl import app
from absl import flags
import pickle

cong = "115-116"

def main(argv):
  del argv
  project_dir = os.path.abspath(
      os.path.join(os.path.dirname(__file__), os.pardir)) 
  source_dir = os.path.join(
      project_dir,"data/congs_115-116_votes") 
  data_dir = os.path.join(source_dir, "raw")
  save_dir = os.path.join(source_dir, "clean")
  icpsr_to_bid = pickle.load(open(os.path.join(source_dir, "icpsr_to_bid.pkl"), 'rb'))

  votes_df = pd.read_csv(
      os.path.join(data_dir, "H{}_votes.csv".format(cong)))
  members_df = pd.read_csv(
      os.path.join(data_dir, "H{}_members.csv".format(cong)))
  rollcalls_df = pd.read_csv(
      os.path.join(data_dir, "H{}_rollcalls.csv".format(cong)))
  df = votes_df.merge(members_df, left_on='icpsr', right_on='icpsr')

  def get_bid(row):
    rep = row['icpsr']
    return icpsr_to_bid[rep]
  
  # Ignore votes that aren't cast as {1, 2, 3, 4, 5, 6}. 
  def get_vote(row):
    if row['cast_code'] in [1, 2, 3]:
      return 1
    elif row['cast_code'] in [4, 5, 6]:
      return 0
    else:
      return -1

  rep = np.array(df.apply(lambda row: get_bid(row), axis=1))
  rep_to_rep_id = dict(
      [(y, x) for x, y in enumerate(sorted(set(rep)))])
  rep_indices = np.array(
      [rep_to_rep_id[s] for s in rep])
  rep_map = np.array(list(rep_to_rep_id.keys()))

  # We also download the first dimension of the fitted DW-Nominate scores.
  first_rep_locations = np.array(
      [np.where(rep == rep_name)[0][0] 
       for rep_name in rep_map])
  nominate_scores = np.array(df['nominate_dim1'])[first_rep_locations]

  # Subtract 1 to zero-index.
  bill_indices = np.array(df.rollnumber) - 1

  votes = np.array(df.apply(lambda row: get_vote(row), axis=1))

  missing_votes = np.where(votes == -1)[0]
  rep_indices = np.delete(rep_indices, missing_votes)
  bill_indices = np.delete(bill_indices, missing_votes)
  votes = np.delete(votes, missing_votes)

  # Get bill information.
  bill_descriptions = np.array(rollcalls_df.vote_desc)
  bill_names = np.array(rollcalls_df.bill_number)

  # Save data.
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  # `votes.npy` is a [num_total_votes] vector that contains a binary indicator
  # for each vote.
  np.save(os.path.join(save_dir, "votes.npy"), votes)

  # `rep_indices.npy` is a [num_total_votes] vector where each entry is an
  # integer in {0, 1, ..., num_reps - 1}, indicating whose vote is the 
  # corresponding entry in `votes.npy`.
  np.save(os.path.join(save_dir, "rep_indices.npy"), rep_indices)

  # `bill_indices.npy` is a [num_total_votes] vector where each entry is an
  # integer in {0, 1, ..., num_bills - 1}, indicating which bill is being votes 
  # on in the corresponding entry in `votes.npy`.
  np.save(os.path.join(save_dir, "bill_indices.npy"), bill_indices)

  # `nominate_scores.npy` is a [num_reps] vector where each entry contains
  # the pre-fitted DW-Nominate ideal points (using the first dimension).
  np.save(os.path.join(save_dir, "nominate_scores.npy"), nominate_scores)
    
  # `rep_map.txt` is a file of [num_reps] strings, containing the names
  # of seach representative.
  np.savetxt(os.path.join(save_dir, "rep_map.txt"), rep_map, fmt="%s")

  # `bill_descriptions.txt` is a file of [num_bills] strings, containing
  # descriptions for each bill.
  np.savetxt(os.path.join(save_dir, "bill_descriptions.txt"), 
             bill_descriptions, 
             fmt="%s")
    
  # `bill_names.txt` is a file of [num_bills] strings, containing the name of
  # each bill.
  np.savetxt(os.path.join(save_dir, "bill_names.txt"), 
             bill_names, 
             fmt="%s")


if __name__ == '__main__':
  app.run(main)
