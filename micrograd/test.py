import torch

a = torch.Tensor([-4.0]).double()
b = torch.Tensor([2.0]).double()
a.requires_grad = True
b.requires_grad = True

c = a + b
d = a * b + b ** 4
c = c + c + 1
c = c + 1 + c + a
d = d + d * 2 + (b + a)
d = d + 3 * d + (b + a)

e = c * d

f = e**2

# g = f / 2.0

# = g + 10.0 / f

f.backward()

print(a.data, b.data, c.data)
print(a.grad, b.grad)



