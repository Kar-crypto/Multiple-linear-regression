# Multiple-linear-regression
import torch
import math
x_train = x.reshape(-1,1)
y_train = torch.reshape(y,(-1,1))
print(x_train.shape, x.shape)
print(y_train.shape, y.shape)
hidden_nodes = 5
class SimpleAnn(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.hiddenL = torch.nn.Linear(1,hidden_nodes)
        self.outputL = torch.nn.Linear(3,1)
        self.hidden2 = torch.nn.Linear(5,3)
    
    def forward(self,x):
        l1 = self.hiddenL(x)
        print("\n***Net input of hidden layer****\n",l1)
        h1 = torch.relu(l1)
        l2 = self.hidden2(h1)
        obj_relu = torch.nn.ReLU()
        h2 = obj_relu(l2)
        l3 = self.outputL(h2)
        h3 = obj_relu(l3)
        return h3
        
model = SimpleAnn()
print(model)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=10e-2)
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()    
    
    #Forward pass
    output = model(x_train)
    
    #Calculate the loss
    loss = criterion(y_train, output)
    print("Loss = ",loss)
    
    if epoch%50 == 0:
        print(f"Epoch : {epoch}, loss : {loss.item()}")
            
    #Backpropagate 
    loss.backward()
        
    #Update weights
    optimizer.step()
torch.save(model,"firstModel1.pt")
model = torch.load("firstModel1.pt")
model.eval()
ypred = model(torch.Tensor(x_train[5]))
print(type(x_train[5]), ypred, torch.sin(x_train[5]))
model = torch.load("firstModel1.pt")
model.eval()
ypred = model(x_train)
print( ypred)
import matplotlib.pyplot as plt
predicted = model(x_train).data.numpy()
print(type(predicted))
print(predicted.shape, type(predicted))
plt.clf()
plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)
plt.legend(loc='best')
plt.show()
