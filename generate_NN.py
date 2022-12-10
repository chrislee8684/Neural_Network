import random

def generate(num_layers, num_nodes, new_NN):

    #load number of nodes into vertex
    num_nodes_vertex = [int(x) for x in num_nodes.split()]

    #initial check for user input
    if len(num_nodes_vertex) != num_layers:
        print("Error:The number of input values for number of nodes per layer does not match the number of layers!")
        quit()

    #generate weights
    weights = [[] for i in range(num_layers-1)] #stores matrices: num weights = num layers - 1

    for i in range(num_layers-1):
        weights[i] = [[0 for x in range(num_nodes_vertex[i]+1)] for y in range(num_nodes_vertex[i+1])] #row = next layer x col = current layer+1 (for bias)

        for a in range(len(weights[i])):
            for b in range(len(weights[i][0])):
                (weights[i])[a][b] = round(random.uniform(0,1), 3)

    #write initialized NN to file
    NN = open(new_NN,"w")

    for i in range(num_layers):
        NN.write(str(num_nodes_vertex[i]) + " ") if i != num_layers-1 else NN.write(str(num_nodes_vertex[i]) + "\n")

    for weight in weights:
        for i in range(len(weight)):
            for j in range(len(weight[0])):
                NN.write("{:.3f}".format(weight[i][j])+" ") if j != len(weight[0])-1 else NN.write("{:.3f}".format(weight[i][j]) + "\n")

