from typing import List

import numpy as np
import matplotlib.pyplot as plt


def collate_fn_pooled_tokens(data: List) -> List:

    n_elems = [len(datum) for datum in data]

    input_ids = [data[i][0] for i in range(len(data))]
    attention_mask = [data[i][1] for i in range(len(data))]

    if all([elem == 2 for elem in n_elems]):
        collated_data = [input_ids, attention_mask]

    else:
        labels = [data[i][2] for i in range(len(data))]
        collated_data = [input_ids, attention_mask, labels]

    return collated_data
