import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
#权重
w = torch.Tensor([1.0])
#需要计算梯度
w.requires_grad = True
#动态计算图
def forward(x):
    return x * w
def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y)**2
print('Pridict (before learning)',4,forward(4).item())
for epoch in range(100):
   for x,y in zip(x_data,y_data):
       l=loss(x,y)
    #输入backward后，计算图被释放，下一次重新作图
       l.backward()
    #item是转换为标量的意思
       print('\tgrad:',x,y,w.grad.item())
    #grad是Tensor，要进行纯数值的计算，需要加上.data，这样才不会制作计算图，单纯进行数值计算
       w.data = w.data - 0.01*w.grad.data
    #把梯度数据全部清零，因为如果不清零，第二次作计算图时还会保留第一次的梯度，并与第二次的相加
       w.grad.data.zero_()
   print("process:",epoch,l.item())
print('Predict(after training)',4,forward(4).item())

