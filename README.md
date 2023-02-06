# PyTorch
Learning PyTorch for deep learning

### PyTorch course information

I am currently working through a Youtube PyTorch course to learn how to use PyTorch for deep learning. This course follows the instruction of Mr.Daniel Bourke and teaches all the fundamentals of what tensors are, how they work, how to create them, how to manipulate them, and of course using PyTorch for different neural network types.

Course Link: https://www.youtube.com/watch?v=Z_ikDlimN6A 

I will be using this repository to track my progress throughout the course as I dive into my PyTorch journey!

### 2/6/23

Following training our model using the training data, we put our model in testing mode using `model.eval()` and used our testing data to make predictions. Additionally, to improve our loss value, we ran our model for an additional 100 epochs (200 epochs total) compared to the first run. 

```
 ### Testing
  model_0.eval() #turns off settings in model not needed for eval/testing (dropout/batch norm layers)
  with torch.inference_mode(): # turns off gradient tracking & other things behind the scenes to speed up computing
    #1. Do the forward pass
    test_pred = model_0(X_test)

    #2. Calculate test loss
    test_loss = loss_fn(test_pred, y_test)

  # Print whats happening
  if epoch % 10 == 0:
    epoch_count.append(epoch)
    loss_values.append(loss)
    test_loss_values.append(test_loss)
    print(f"Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss}")

    # Print out model state_dict()
    print(model_0.state_dict())
```
After running our model on the test data, we saw vast improvement of the test loss value from 0.418 initially to 0.005 after 200 epochs. We plotted both the training and test loss in a loss curve to show the improvement over time (Fig 1). To visualize how well our model was able to use the test data and predict the ideal values we used a scatter plot showing our training data, testing data, and test predictions (Fig 2). As you can see, our predictions are almost perfectly lined up with our test data! While this is great, improvements can be made to make the predictions more accurate such as adjusting the learning rate (our model uses 0.01) or increasing the number of epochs.

| Loss Curve        | Trained Test Predictions (200 epochs)          |
| ---------------------------- | ---------------------------- |
| ![losscurve](https://user-images.githubusercontent.com/108199140/217107701-5b48235e-5c2e-433b-83d8-55f151d3fed4.PNG) | ![testpreds](https://user-images.githubusercontent.com/108199140/217107716-c04f7046-8009-4e33-93db-d400728eeabe.PNG) |


### 2/3/23

#### Building and running a PyTorch training loop

We created our first model and ran our training data through 100 epochs, following training you can see the predicted values are much closer to the ideal values, the loss decreased, and the weight/bias values became closer to the weight and bias we set to generate our data!

| Untrained Predictions        | Trained Predictions (100 epochs)          |
| ---------------------------- | ---------------------------- |
| <img width="379" alt="Screen Shot 2023-02-03 at 1 06 04 PM" src="https://user-images.githubusercontent.com/108199140/216676155-d9be7312-1e6f-4eaa-be8f-1b4fd5ca8553.png"> | <img width="376" alt="Screen Shot 2023-02-03 at 1 06 13 PM" src="https://user-images.githubusercontent.com/108199140/216676218-c4912c35-063e-4885-8848-f1df4dfb3927.png"> |

#### Changes in Weight, Bias, and Loss 
| Value     |    Ideal    |  10 Epochs     | 100 Epochs |
| ---         |       ----     |   ---     |     --- |
| Weight   |           0.7     |     0.06     |      0.62     |
| Bias     |           0.3     |     0.43      |     0.33     |
| Loss   |              -      |     0.36      |     0.02      |

### 2/2/23

#### Setup model training

To begin training our model, we set up our loss function and optimizer. As we are running a linear regression, we chose `nn.L1Loss()` for our loss function which calculates the mean absolute error (MAE) between our predicted and actual values. For our optimizer, we chose `torch.optim.SGD()` or Stochastic Gradient Descent (SGD).

```
# Setup a loss function
loss_fn = nn.L1Loss()

# Setup an optimizer (stochastic gradient descent)
  # Learning rate value will adjust parameters to the same power
optimizer = torch.optim.SGD(model_0.parameters(),
                            lr=0.01) # lr = learning rate = possibly most important hyperparameter you can set
```

#### Building our Linear Regression Model

We built our first model which is a linear regression model where we started with random values for our two parameters (weight & bias). Since we are subclassing `nn.module` in our model, we used the `forward()` method to define the computation to be performed in our model.

```
from torch import nn

# Create linear regression model
class LinearRegressionModel(nn.Module): #almost everything in PyTorch inherets from nn.module subclass
  def __init__(self):
    super().__init__()

    # Initialize the model parameters to be used in compuations
    self.weights = nn.Parameter(torch.randn(1, #nn.Parameter stores tensors that can be used with nn.Module
                                            requires_grad=True, #used for updating model parameters via gradient descent
                                            dtype=torch.float))
    self.bias = nn.Parameter(torch.randn(1,
                                        requires_grad=True,
                                        dtype=torch.float))
    
  # forward() defines the compuation in the model and is required when calling nn.Module subclass
  # this defines the compuation that takes place on the data passed to the particular nn.Module
  def forward(self, x: torch.Tensor) -> torch.Tensor: # <- "x" is the input data 
    return self.weights * x + self.bias #this is the linear regression formula
```
