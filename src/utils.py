from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def collate_fn_pooled_tokens(data: Tuple) -> List:
    input_ids = [data[i][0] for i in range(len(data))]
    attention_mask = [data[i][1] for i in range(len(data))]

    if len(data) == 2:
        collated_data = [input_ids, attention_mask]

    else:
        labels = [data[i][2] for i in range(len(data))]
        collated_data = [input_ids, attention_mask, labels]

    return collated_data
