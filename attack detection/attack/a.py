import torch
import numpy as np

conf_matrix = torch.zeros(11, 12)
tt = torch.sum(conf_matrix, dim=1).unsqueeze(1)

conf_matrix = conf_matrix / tt
print(conf_matrix.shape)