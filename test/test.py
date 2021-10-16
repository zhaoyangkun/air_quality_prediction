# import torch.nn as nn
# import torch

# x = torch.rand(10, 24, 100)
# print(x.size())
# y = x[:, -1, :]
# print(y.size())
# # lstm = nn.LSTM(100, 16, num_layers=3)
# # output, (h, c) = lstm(x)
# # print(output.size())
# # print(h.size())
# # print(c.size())

# IAQI_DICT = {
#     'IAQI': [0, 50, 100, 150, 200, 300, 400, 500],
#     'CO': [0, 2, 4, 14, 24, 36, 48, 60],
#     'SO2': [0, 50, 150, 475, 800, 1600, 2100, 2620],
#     'NO2': [0, 40, 80, 180, 280, 565, 750, 940],
#     'O3': [0, 100, 160, 215, 265, 800,0,0],
#     'PM10': [0, 50, 150, 250, 350, 420, 500, 600],
#     'PM2.5': [0, 35, 75, 115, 150, 250, 350, 500],
# }
# IAQI_DICT_KYES = list(IAQI_DICT.keys())
# print(max(IAQI_DICT[IAQI_DICT_KYES[4]]))

import numpy as np
# a = np.zeros((3))
# print(a)

a = np.array(['' for _ in range(4)])
print(a)
