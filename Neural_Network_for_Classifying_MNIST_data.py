# Import numpy for arrays and matplotlib for drawing the numbers
import numpy
import matplotlib.pyplot as plt

# Import scipy.special for the sigmoid function expit()
import scipy.special, numpy

# Neural network class definition
class NeuralNetwork:
    # Init the network, this gets run whenever we make a new instance of this class
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set the number of nodes in each input, hidden and output layer
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes

        # Weight matrices, wih (input -> hidden) and who (hidden -> output)
        self.wih = numpy.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.who = numpy.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))

        # Set the learning rate 
        self.lr = learning_rate

        # Set the activation function, the logistic sigmoid
        self.activation_function = lambda x: scipy.special.expit(x)

    # Train the network using back-propagation of errors
    def train(self, inputs_list, targets_list):
        # Convert inputs into 2D arrays
        inputs_array = numpy.array(inputs_list, ndmin=2).T
        targets_array = numpy.array(targets_list, ndmin=2).T

        # Calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs_array)

        # Calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # Calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # Current error is (target - actual)
        output_errors = targets_array - final_outputs

        # Hidden layer errors are the output errors, split by the weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # Update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
        numpy.transpose(hidden_outputs))

        # Update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
        numpy.transpose(inputs_array))

    # Query the network
    def query(self, inputs_list):
        # Convert the inputs list into a 2D array
        inputs_array = numpy.array(inputs_list, ndmin=2).T

        # Calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs_array)

        # Calculate output from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into final layer
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # Calculate outputs from the final layer
        final_outputs =self.activation_function(final_inputs)
        return final_outputs


#Create Neural Network - optimal for MNIST hand Written data
#input_nodes = 784
#hidden_nodes = 397
#output_nodes = 10
#learning_rate = 0.1

#Create Neural Network - optimal for MNIST fashion data
input_nodes = 784
hidden_nodes = 397
output_nodes = 10
learning_rate = 0.02

#Create the neural network
n = NeuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

# Load the MNIST 100 training samples CSV file into a list
training_data_file = open("fashion_mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

#Optimal range 4 for MNIST hand written data
#Optimal range 6 for MNIST fashion data (more yeilds margionally better but at far greater compute time)

for x in range(6):

    # Train the neural network on each trainingsample
    for record in training_data_list:
        # Split the record by the commas
        all_values = record.split(',')
        # Scale and shift the inputs from 0..255 to 0.01..1
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # Create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        # All_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        # Train the network
        n.train(inputs, targets)
    pass

# Load the MNIST test samples CSV file into a list
test_data_file = open("fashion_mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# Scorecard list for how well the network performs, initially empty
scorecard = []

# Loop through all of the records in the test data set
for record in test_data_list:
    # Split the record by the commas
    all_values = record.split(',')
    # The correct label is the first value
    correct_label = int(all_values[0])
    print(correct_label, "Correct label")
    # Scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # Query the network
    outputs = n.query(inputs)
    # The index of the highest value output corresponds to the label
    label = numpy.argmax(outputs)
    print(label, "Network label")
    # Append either a 1 or a 0 to the scorecard list
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
    pass
    pass

# Calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
print("Performance = ", (scorecard_array.sum() / scorecard_array.size)*100, '%')

#Show the incorrectly classified images
show_incorrect = input("Show Incorrectly Classified Images? [Y/N]: ")
if show_incorrect == "Y":

    for score in range(len(scorecard_array)):
        if (scorecard_array[score] == 0):
            # Take the first line (data_list index 0, the first sample), and split it up based on the commas
            # all_values now contains a list of [label, pixel 1, pixel 2, pixel 3, ... ,pixel 784]
            all_values = test_data_list[score].split(',')

            # Take the long list of pixels (but not the label), and reshape them to a 2D array of pixels
            image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))

            # Plot this 2D array as an image, use the grey colour map and don’t interpolate
            plt.imshow(image_array, cmap='Greys', interpolation='None')
            plt.show()
            