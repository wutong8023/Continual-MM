"""


Author: Tong
Time: --2021
"""

import json
import numpy as np

with open("data/webred/webred_21.json", "r") as file_in:
    original_data = json.load(file_in)

# process data into <x, y>
_pair_data = []
for item in original_data:
    _pair_data.append([item['sentence'], item['relation_name']])
pass

len_ = []
for i, sent in enumerate(_pair_data):
    len_.append(len(sent[0].split()))
len_ = np.array(len_)

length = len(len_)

print(np.max(len_))
print(np.size(len_))
print("200: ", float(len(np.where(len_>200)[0]) / length))
print("100: ", float(len(np.where(len_>100)[0]) / length))
print("80: ", float(len(np.where(len_>80)[0]) / length))
print("70: ", float(len(np.where(len_>70)[0]) / length))
print("60: ", float(len(np.where(len_>60)[0]) / length))
print("50: ", float(len(np.where(len_>50)[0]) / length))

