how do i make a nn to solve the iris data set?


from sklearn import datasets

iris = datasets.load_iris()

inputs = iris.data

# Since each prediction is a vector with a score for each of the three types of flowers,
# we need to make each target a vector with a 1 for the correct flower and a 0 for the others.
targets = np.zeros((len(iris.data), 3))
X = iris.data.astype(float)
y = iris.target

#one hot encoding 
Y = np.zeros((len(y),3))
Y[np.arange(len(y)),y] = 1

X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-14)
You said:
import numpy as np

class NeuralNetwork: 
    def __init__(
        self,
        network_input_size,
        layer_output_sizes,
        activation_funcs,
        activation_ders,
        cost_fun,
        cost_der,
        lr = 1e-2,
    ):
        self.input_size = network_input_size
        self.output_sizes = layer_output_sizes
        self.activation_funcs = activation_funcs
        self.activation_ders = activation_ders
        self.cost_fun = cost_fun
        self.cost_der = cost_der
        self.lr = lr

        rng = np.random.default_rng()

        layers = []
        in_size = self.input_size
        for out_size, act in zip(self.sizes, self.activation_funcs):
            rng = np.random.default_rng()
            W = rng.normal(0, 0.01, size=(in_size,out_size))
            b = np.zeros((1,out_size))
            layers.append([W,b])
            in_size = out_size
        self.layers = layers


    def predict(self, inputs):
        # Simple feed forward pass
        a = inputs 
        for (W, b), act in zip(self.layers, self.activation_funcs):
            z = a @ W + b 
            a = act(z)
        return a 

    def cost(self, inputs, targets):
        yhat = self.predict(inputs)
        return self.cost_fun(yhat, targets)

    def _feed_forward_saver(self, inputs):
        activations = [inputs]
        zs = []
        a = inputs
        for (W, b), act in zip(self.layers, self.activation_funcs):
            z = a @ W + b
            a = act(z)
            zs.append(z)
            activations.append(a)
        return activations, zs

    def compute_gradient(self, inputs, targets):
        m = inputs.shape[0]
        As, Zs = self._feed_forward_saver(inputs)
        yhat = As[-1]

        grads = [None] * len(self.layers)
        delta = self.cost_der(yhat, targets)*self.activation_ders[-1](Zs[-1])

        dW = (As[-2].T @ delta) / m
        db = np.sum(delta, axis=0, keepdims=True) / m
        grads[-1] = (dW, db)

        # Backpropagation
        for L in range(len(self.layers)-2,-1,-1):
            W_next = self.layers[L+1][0]
            delta = (delta @ W_next.T) * self.activation_ders[L](Zs[L])
            dW = (As[L].T @ delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            grads[L] = (dW, db)

        return grads

    def update_weights(self, layer_grads):
        for (W, b), (dW, db) in zip(self.layers, layer_grads):
            W -= self.lr * dW
            b -= self.lr * db



i have this code for a neural network.
with these activation functions

def relu(z):
    return np.maximum(0,z)

def drelu(z):
    return (z > 0).astype(z.dtype)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def dsigmoid(z):
    s = sigmoid(z)
    return s * (1.0-s)

def softmax(z):
    z_stable = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def xent(yhat_softmax, y_oneshot):
    eps = 1e-12
    return -np.mean(np.sum(y_oneshot * np.log(yhat_softmax + eps), axis=1))

def dxent_softmax(yhat_softmax, y_oneshot):
    return (yhat_softmax - y_oneshot)

im trying to use the iris dataset 

from sklearn import datasets

iris = datasets.load_iris()

inputs = iris.data

# Since each prediction is a vector with a score for each of the three types of flowers,
# we need to make each target a vector with a 1 for the correct flower and a 0 for the others.
targets = np.zeros((len(iris.data), 3))
X = iris.data.astype(float)
y = iris.target

#one hot encoding 
Y = np.zeros((len(y),3))
Y[np.arange(len(y)),y] = 1

X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-14)

how can i use the NN on the iris data and then find the accuracy after the training?
You said:
**b)** Add a figure in your report of a heatmap showing the test accuracy of a neural network with [0, 1, 2, 3] hidden layers and [5, 10, 25, 50] nodes per hidden layer.

lets do this then
You said:
Hidden layers=0, nodes= 5 -> test acc=0.967
Hidden layers=0, nodes=10 -> test acc=0.967
Hidden layers=0, nodes=25 -> test acc=0.967
Hidden layers=0, nodes=50 -> test acc=0.967
Hidden layers=1, nodes= 5 -> test acc=0.967
Hidden layers=1, nodes=10 -> test acc=0.967
Hidden layers=1, nodes=25 -> test acc=0.967
Hidden layers=1, nodes=50 -> test acc=0.967
Hidden layers=2, nodes= 5 -> test acc=0.667
Hidden layers=2, nodes=10 -> test acc=0.800
Hidden layers=2, nodes=25 -> test acc=0.667
Hidden layers=2, nodes=50 -> test acc=0.867
Hidden layers=3, nodes= 5 -> test acc=0.667
Hidden layers=3, nodes=10 -> test acc=0.533
Hidden layers=3, nodes=25 -> test acc=0.833
Hidden layers=3, nodes=50 -> test acc=0.767

does this output make sense?
You said:
Can you make a short text on explaining why we get these results.
You said:
Can you make the figure in a latex format?
You said:
Can you simply make it so that the plt. get a latex format on the text and numbers.
You said:
**c)** Add a figure in your report which meets as few requirements as possible of what we consider a good figure in this course, while still including some results, a title, figure text, and axis labels. Describe in the text of the report the different ways in which the figure is lacking. (This should not be included in the final report for project 2.)


lets do this one now. Lets simply plot the train loss over the epochs
You said:
lets not actually make a bad figure. Its just to be a figure that has the minimal needs of a figure. Lets make a proper figure where we plot the train loss over 800 epochs
You said:
this is terrible: Not what youre supposed to do.

We want a version of this where we plot the loss over the epochs. But lets include every epoch instead of only doing every 100

nn = NeuralNetwork(
    network_input_size=4,
    layer_output_sizes=[16, 3],                      # [hidden, output]
    activation_funcs=[relu, softmax],                # last = softmax
    activation_ders=[drelu, identity_derivative],    # last derivative = identity
    cost_fun=xent,
    cost_der=dxent_softmax,
    lr=0.05,
    seed=123,
)

epochs = 800
for ep in range(1, epochs + 1):
    grads = nn.compute_gradient(X_train, Y_train)
    nn.update_weights(grads)
    if ep % 100 == 0 or ep == 1:
        yhat_tr = nn.predict(X_train)
        loss_tr = xent(yhat_tr, Y_train)
        preds_tr = np.argmax(yhat_tr, axis=1)
        acc_tr = np.mean(preds_tr == y_train)
        print(f"epoch {ep:4d} | train loss={loss_tr:.4f} | train acc={acc_tr:.3f}")

yhat_test = nn.predict(X_test)
test_loss = xent(yhat_test, Y_test)
y_pred = np.argmax(yhat_test, axis=1)
test_acc = np.mean(y_pred == y_test)
print(f"\nTest loss: {test_loss:.4f} | Test accuracy: {test_acc:.3f}")
You said:
can you make a figure text for this plot?
You said:
can you also make a figure text for this plot?

import matplotlib.pyplot as plt
import matplotlib as mpl

# Enable LaTeX-style text rendering
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

fig, ax = plt.subplots(figsize=(6.5, 4.6))
im = ax.imshow(results, origin='upper', aspect='auto')

ax.set_xticks(np.arange(len(hidden_nodes_list)), labels=[str(h) for h in hidden_nodes_list])
ax.set_yticks(np.arange(len(hidden_layers_list)), labels=[str(l) for l in hidden_layers_list])
ax.set_xlabel(r"\textbf{Nodes per hidden layer}")
ax.set_ylabel(r"\textbf{Number of hidden layers}")
ax.set_title(r"\textbf{Iris -- Test Accuracy Heatmap}")

for i in range(results.shape[0]):
    for j in range(results.shape[1]):
        ax.text(j, i, f"{results[i,j]*100:.1f}\\%", ha="center", va="center",
                fontsize=9, color="black" if results[i,j] > 0.9 else "white")

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r"\textbf{Accuracy}")

plt.tight_layout()
plt.savefig("iris_nn_heatmap.pdf", bbox_inches="tight")
plt.show()
You said:
\section{Results and Discussion}\label{section:results} 

Give me the skeleton to add my two plots, and where i can simply paste in the figure text and the reflection about the first plot results.


You said:
Can you make a quick readme file?

it should be in the same format as this:

Group members: Egil Furnes, Ådne Rolstad, Bror Johannes Tidemand Ruud

Project Description: 

This project investigates regression methods in statistical learning with a focus on the bias–variance trade-off. Using the Runge function as a case study, we implement and compare Ordinary Least Squares (OLS), Ridge, and Lasso regression. Different optimization algorithms, including gradient descent, stochastic gradient descent, and momentum methods, are applied to fit the models. To evaluate performance and stability, we use resampling techniques such as bootstrapping and cross-validation. The project combines theoretical analysis with numerical experiments to highlight the strengths and limitations of these regression approaches.

To install the required packages:
Run the following in terminal !pip install -r requirements.txt 

There is only the notebook codeForPlots with code. It has the code to make the plots for the exercisesweek39 report. It makes the BiasVariancePlot and HeatMap for different lambda values for ridge regression.

Except it being about the second project instead of the first