import torch 
import torch.nn as nn

sentence1 = "I am"
print(list(sentence1))
b=[]
a=[]
weights = torch.Tensor([[[2, 4, 3, 1],[2, 3, 4, 1]],[[20, 10, 3, 0],[5, 1, 2, 0]], [[20, 50, 3, 0],[5, 1, 2, 100]]])
print(weights[0][0][0:1])
for i in range(weights.size(0)):
    c = weights[i,:,:].squeeze()
    b.append(nn.functional.softmax(c, dim=1))
print(weights[0].size())
print("111")
print(weights[0].view(-1).size())
for w in b:
    a.append(torch.multinomial(w, 1, replacement=True).squeeze())
a = torch.stack(a)
print(a)
print(a.size())


for s in a:
    print(s)
# for i in range(weights.size()[0]):
#     if i == 0:
#         a = torch.multinomial(weights[i], 1, replacement=True)
#     else:
#         b = torch.multinomial(weights[i], 1, replacement=True)
#         a = torch.cat((a, b), dim=0)
#     print(a)
# a = a.squeeze().view(3, 2).t()
# for i in weights[0]:
#     print(i.numpy())