# PyTorch
Learning PyTorch for deep learning

### PyTorch course information

I am currently working through a Youtube PyTorch course to learn how to use PyTorch for deep learning. This course follows the instruction of Daniel Bourke and teaches all the fundamentals of what tensors are, how they work, how to create them, how to manipulate them, and of course using PyTorch for different neural network types.

Course Link: https://www.youtube.com/watch?v=Z_ikDlimN6A 

I will be using this repository to track my progress throughout the course as I dive into my PyTorch journey!

### 2/3/23

We created our first model and ran our training data through 100 epochs, following training you can see the predicted values are much closer to the ideal values, the loss decreased, and the weight/bias values became closer to the weight and bias we set to generate our data!

| Untrained Predictions        | Trained Predictions          |
| ---------------------------- | ---------------------------- |
| <img width="379" alt="Screen Shot 2023-02-03 at 1 06 04 PM" src="https://user-images.githubusercontent.com/108199140/216676155-d9be7312-1e6f-4eaa-be8f-1b4fd5ca8553.png"> | <img width="376" alt="Screen Shot 2023-02-03 at 1 06 13 PM" src="https://user-images.githubusercontent.com/108199140/216676218-c4912c35-063e-4885-8848-f1df4dfb3927.png"> |

#### Changes in Weight, Bias, and Loss 
| Value     |    Ideal    |  10 Epochs     | 100 Epochs |
| ---         |       ----     |   ---     |     --- |
| Weight   |           0.7     |     0.06     |      0.62     |
| Bias     |           0.3     |     0.43      |     0.33     |
| Loss   |              -      |     0.36      |     0.02      |

### 2/2/23

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
