import pdb
import sys

import pandas as pd

src_csv_path = sys.argv[1]
data = pd.read_csv(src_csv_path)

pdb.set_trace()

sorted_data = data.sort_values(by='Loss', ascending=True)
lowest_loss_val = sorted_data.iloc[0]