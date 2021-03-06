import torch
masks = torch.tensor([[True, True, False], [False, False, False]])
tag_probs = [[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]], [[16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]]
print(tag_probs)
print(masks.repeat_interleave(5).view(2, 3, 5))

log_softmax = torch.nn.Softmax(dim=0)
a = torch.tensor([[1, 1, 1], [2, 2, 2]])
print(log_softmax(a))