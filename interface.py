from tkinter import *
import generate_NN as gen
import NN as NN

#initialize GUI
interface = Tk()
interface.geometry("1000x500")
interface.title("Neural Network Interface")
interface.config(bg="#E5E4E2")

#GUI Functions
def train():
    init_NN = str(init_NN_entry.get())
    training_set = str(training_set_entry.get())
    output_NN = str(output_NN_entry.get())
    num_epochs = int(num_epochs_entry.get())
    learn_rate = float(learn_rate_entry.get())
    activation = activation_switch.get()

    if activation == 'Sigmoid':
        activationVal = 0
    elif activation == 'ReLU':
        activationVal = 1

    NeuralNet = NN.NeuralNet(init_NN, training_set, output_NN, num_epochs, learn_rate, activationVal)
    NeuralNet.train()

def test():
    NN_file = str(NN_entry.get())
    testing_set = str(testing_set_entry.get())
    result_file = str(result_file_entry.get())

    activation = activation_switch.get()

    if activation == 'Sigmoid':
        activationVal = 0
    elif activation == 'ReLU':
        activationVal = 1

    NeuralNet = NN.NeuralNet (NN_file, testing_set, result_file, 0, 0, activationVal)
    NeuralNet.test()

def generate():
    num_layers = int(num_layers_entry.get())
    num_nodes = str(num_nodes_entry.get())
    new_NN = str(new_NN_entry.get())

    gen.generate(num_layers, num_nodes, new_NN)

#GUI Gadgets
title_label = Label(interface, text="Neural Network GUI", font=('Times 30'), bg="#E5E4E2")
title_label.place(relx=0.5, rely=0.05, anchor = CENTER)

#Train
train_label = Label(interface, text="Train",font=('Times 25 underline'), bg="#E5E4E2")
train_label.place(relx=0.22, rely=0.15, anchor = E)

init_NN_label = Label(interface, text="Initial Neural Network (.txt):", bg="#E5E4E2")
init_NN_label.place(relx=0.2, rely=0.25, anchor=E)
init_NN_entry = Entry(interface, width=15)
init_NN_entry.place(relx=0.2, rely=0.25, anchor=W)

training_set_label = Label(interface, text="Training Set (.txt):", bg="#E5E4E2")
training_set_label.place(relx=0.2, rely=0.35, anchor = E)
training_set_entry = Entry(interface, width=15)
training_set_entry.place(relx=0.2, rely=0.35, anchor=W)

output_NN_label = Label(interface, text="Output Neural Network(.txt):", bg="#E5E4E2")
output_NN_label.place(relx=0.2, rely=0.45, anchor = E)
output_NN_entry = Entry(interface, width=15)
output_NN_entry.place(relx=0.2, rely=0.45, anchor=W)

num_epochs_label = Label(interface, text="Number of Epochs:", bg="#E5E4E2")
num_epochs_label.place(relx=0.2, rely=0.55, anchor = E)
num_epochs_entry = Entry(interface, width=15)
num_epochs_entry.place(relx=0.2, rely=0.55, anchor=W)

learn_rate_label = Label(interface, text="Learning Rate:", bg="#E5E4E2")
learn_rate_label.place(relx=0.2, rely=0.65, anchor = E)
learn_rate_entry = Entry(interface, width=15)
learn_rate_entry.place(relx=0.2, rely=0.65, anchor=W)

activation_label = Label(interface, text = "Activation Function:", bg="#E5E4E2")
activation_label.place(relx=0.2, rely=0.73, anchor=E)
activation_choices = ['Sigmoid', 'ReLU']
activation_switch=StringVar()
activation_entry = OptionMenu(
    interface,
    activation_switch,
    *activation_choices
)
activation_entry.place(relx=0.2, rely=0.73, anchor=W)

train_button=Button(
    interface,
    text="Train",
    height=4,
    width=13,
    bg="#E5E4E2",
    command = train
)
train_button.place(relx=0.33, rely=0.85, anchor=E)

#Test
test_label = Label(interface, text="Test",font=('Times 25 underline'), bg="#E5E4E2")
test_label.place(relx=0.54, rely=0.15, anchor = E)

NN_label = Label(interface, text="Neural Network (.txt):", bg="#E5E4E2")
NN_label.place(relx=0.52, rely=0.25, anchor=E)
NN_entry = Entry(interface, width=15)
NN_entry.place(relx=0.52, rely=0.25, anchor=W)

testing_set_label = Label(interface, text="Testing Set (.txt):", bg="#E5E4E2")
testing_set_label.place(relx=0.52, rely=0.35, anchor=E)
testing_set_entry = Entry(interface, width=15)
testing_set_entry.place(relx=0.52, rely=0.35, anchor=W)

result_file_label = Label(interface, text="Result File (.txt):", bg="#E5E4E2")
result_file_label.place(relx=0.52, rely=0.45, anchor=E)
result_file_entry = Entry(interface, width=15)
result_file_entry.place(relx=0.52, rely=0.45, anchor=W)

test_button=Button(
    interface,
    text="Test",
    height=4,
    width=13,
    bg="#E5E4E2",
    command = test
)
test_button.place(relx=0.655, rely=0.6, anchor=E)

#Generate
generate_label = Label(interface, text="Generate",font=('Times 25 underline'), bg="#E5E4E2")
generate_label.place(relx=0.9, rely=0.15, anchor = E)

num_layers_label = Label(interface, text="Number of Layers:", bg="#E5E4E2")
num_layers_label.place(relx=0.83, rely=0.25, anchor=E)
num_layers_entry = Entry(interface, width=15)
num_layers_entry.place(relx=0.83, rely=0.25, anchor=W)

num_nodes_label = Label(interface, text="Number of Nodes:", bg="#E5E4E2")
num_nodes_label.place(relx=0.83, rely=0.35, anchor=E)
num_nodes_entry = Entry(interface, width=15)
num_nodes_entry.place(relx=0.83, rely=0.35, anchor=W)

new_NN_label = Label(interface, text="Neural Network (.txt):", bg="#E5E4E2")
new_NN_label.place(relx=0.83, rely=0.45, anchor=E)
new_NN_entry = Entry(interface, width=15)
new_NN_entry.place(relx=0.83, rely=0.45, anchor=W)

generate_button=Button(
    interface,
    text="Generate",
    height=4,
    width=13,
    bg="#E5E4E2",
    command = generate
)
generate_button.place(relx=0.96, rely=0.6, anchor=E)

note_label = Label(interface, text="*For node entry (Generate), please enter the number of nodes for each layer as values separated by a single space. Example:n1 n2 n3 ...", bg="#E5E4E2")
note_label.place(relx=0.05, rely=0.95, anchor=W)

#GUI loop
interface.mainloop()