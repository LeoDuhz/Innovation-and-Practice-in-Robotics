import torch
def BMM(A, B):
    C = torch.empty(A.size()[0], A.size()[1], B.size()[2])
    B = torch.transpose(B, 2, 1)
    for i in range(A.size()[0]):
        for j in range(A.size()[1]):
            for k in range(B.size()[1]):
                C[i][j][k] = torch.dot(A[i][j], B[i][k])
    return C
A = torch.tensor([[[1,2],[3,4]],[[5,6],[7,8]]])
B = torch.tensor([[[8,7],[6,5]],[[4,3],[2,1]]])
# print(B.transpose(2, 1))
print(BMM(A,B))