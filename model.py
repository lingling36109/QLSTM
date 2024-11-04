# Importing all the necessary libraries
import math
import torch
from torch import nn
from torch.utils.data import Dataset

# Recylcing some old code from previous projects
# Basically creates the dataset for the project but as a class using PyTorch's Dataset
class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length):
        # Definition of the basic variables
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        # Switching the Pandas Dataframe into a PyTorch Tensor so it can be inputted into the machine learning models
        self.y = torch.tensor(dataframe[self.target].values).float()
        self.X = torch.tensor(dataframe[self.features].values).float()

    def __len__(self):
        # Getting the batch size of the dataset
        return self.X.shape[0]

    def __getitem__(self, i):
        # Creating the sequences for each time step and organzing them into batches
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]
    
# Definition of the LSTM class
# Using the nn.Module from pytorch as the base class
class LSTM_Model(nn.Module):
    def __init__(self, num_features, hidden_units):
        super().__init__()
        # Declaration of the LSTM Model
        # Creating variables that need to be accessed by other portions of the class
        # Most are self-explanatory
        self.num_features = num_features
        self.hidden_units = hidden_units
        self.num_layers = 1

        # Using the pytorch LSTM class because of convenience
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        # Linear output layer
        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        # Getting batch size from the input
        batch_size = x.shape[0]

        # Creating the hidden and control states for the matrix
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        # Front propagation portion of the code
        _, (hn, _) = self.lstm(x, (h0, c0))

        # Getting the output using the linear layer and flattening due an additional dimension from the number of layers
        out = self.linear(hn[0]).flatten()
        return out

# Defining the training and test functions for the LSTM (reused from previous projects)
def train_LSTM(data_loader, model, loss_function, optimizer):
    # Gathering the number of batches, setting model to training mode, and setting total loss equal to 0
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    # Data_loader is an iterator, so we get each X and y of the training set through this for loop
    for X, y in data_loader:
        # Getting the output of the model and calculating the loss
        output = model(X)
        loss = loss_function(output, y)

        # Clearing the gradient and then making the gradient to back propagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"LSTM Train loss: {avg_loss}")
    return avg_loss


def test_LSTM(data_loader, model, loss_function):
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    print(f"LSTM Test loss: {avg_loss}")
    return avg_loss

# Definition of the RLSTM class
# Using the nn.Module from pytorch as the base class
class RLSTM_Model(nn.Module):
    def __init__(self, num_features, num_recall, hidden_units, batch_size, omega):
        super().__init__()
        # Definition of the RLSTM Model
        # Creating variables inside the class so other parts can access them and also very useful for debugging
        self.num_features = num_features
        self.hidden_units = hidden_units
        self.num_recall = num_recall  # Number of past states to recall
        self.batch_size = batch_size
        self.omega = omega  # When to activate the recall mechanism
        self.states_array = [] # Storing the states of each sequential index
        self.recalled = [] # Stores what indices are recalled which is for debugging/research purposes

        # Declaration of the weight matrices (LSTM portion of the code)
        # Instead of creating individual weight matrices for each of the 4 gates and for both the input and hidden state
        # We can combine them into a singular weight matrix and later split them, when doing calculations
        self.W = nn.Parameter(torch.Tensor(num_features, hidden_units * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_units, hidden_units * 4))
        self.P = nn.Parameter(torch.Tensor(batch_size, batch_size * 2)) # Another matrix to fix projection size
        self.B = nn.Parameter(torch.Tensor(hidden_units * 4))

        # Defining the weight matrices for the recall mechanism and we can also add another bias term if needed
        self.W_r = nn.Parameter(torch.Tensor(num_features, num_recall))
        self.P_r = nn.Parameter(torch.Tensor(1, batch_size))
        self.U_r = nn.Parameter(torch.Tensor(hidden_units, num_recall))

        # We are declaring the linear layer for the output
        self.linearOut = nn.Linear(in_features=self.hidden_units, out_features=1)

        # Setting up the weights with this command
        self.init_weights()

    def init_weights(self):
        # Function within the class that sets up the weights using a random, uniform distribution
        stdv = 1.0 / math.sqrt(self.hidden_units)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def reset(self):
        # Reset the states array for when restarting training for each epoch
        self.states_array = []

    def forward(self, x, init_states=None):
        # Declaration of hidden states
        # Hidden states are zero matrices
        h_t, c_t = (torch.zeros(self.batch_size, self.hidden_units).to(x.device),
                   torch.zeros(self.batch_size, self.hidden_units).to(x.device))

        # Detaching the hidden and control state from the gradient
        h_t = h_t.detach()
        c_t = c_t.detach()
        
        # Defining the states array as empty
        self.states_array = []
        
        seq_len = x.size(1)
        for seq in range(seq_len):
            # First step is making the input and hidden matrices
            # This includes getting steps of the sequence from every batch
            # One very interesting technical detail is that RNNs are trained by the sequence
            # Simply put this means that each batch is trained simultaneously (vectorization of batches)
            x_t = x[:,seq,:]
            if x_t.size()[0] < self.batch_size:
                padding = torch.zeros(self.batch_size-x_t.size()[0], self.num_features)
                x_t = torch.cat((x_t, padding), 0)
            x_t_prime = x_t
            h_t_prime = h_t

            # Added additional hyperparameter to the model
            # Checks whether we should start recalling based on the length of the states array
            # If it is under the limit, we just duplicate the tensor and input it
            # Otherwise, it just duplicates the input vector to make sure our matrices are consistently sized
            if len(self.states_array) < self.omega:
                x_t_prime = torch.cat((x_t_prime, x_t), 0)
                h_t_prime = torch.cat((h_t_prime, h_t), 0)
            else:
                recall_gates = torch.sigmoid(self.P_r @ ((x_t @ self.W_r) + (h_t @ self.U_r)))
                best_indices = torch.round((len(self.states_array)*recall_gates))
                self.recalled.append(best_indices)
                for best_ind in best_indices[0]:
                    x_t_prime = torch.cat((x_t_prime, self.states_array[int(best_ind.item())][0]), 0)
                    h_t_prime = torch.cat((h_t_prime, self.states_array[int(best_ind.item())][1]), 0)

            # Updating the states array
            self.states_array.append([x_t, h_t])

            # Weight multiplications, similar to the typical LSTM
            # However, the matrix sizes have been slightly modified due to the recall mechanism
            # The recall mechanism makes the inputs different sizes from your typical
            LSTM_gates = self.P @ ((x_t_prime @ self.W) + (h_t_prime @ self.U)) + self.B
            i_t, f_t, g_t, o_t = LSTM_gates.chunk(4, 1)
            i_t, f_t, g_t, o_t = torch.sigmoid(i_t), torch.sigmoid(f_t), torch.tanh(g_t), torch.sigmoid(i_t)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

        # Getting the output from the linear layer
        out = self.linearOut(h_t)
        return out
    
# Similiar comments to the training LSTM function
def train_RLSTM(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for X, y in data_loader:
        optimizer.zero_grad()
        output = model(X)
        
        # One minor difference in the code for the train RLSTM function
        # These additional lines of code make the ouput of the model and the cirrect output the same size
        # Otherwise there is an error when using the loss function 
        # This was a major technical bottleneck during the coding the process
        y = y.unsqueeze(1)
        if y.size()[0] < model.batch_size:
            padding = torch.zeros(model.batch_size - y.size()[0], 1)
            y = torch.cat((y, padding), 0)

        loss = loss_function(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        model.reset()

    avg_loss = total_loss / num_batches
    print(f"RLSTM Train loss: {avg_loss}")
    return avg_loss

# Same as the testing LSTM function
# Seperated for the purposes of debugging 
def test_RLSTM(data_loader, model, loss_function):
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    print(f"LSTM Test loss: {avg_loss}")
    return avg_loss