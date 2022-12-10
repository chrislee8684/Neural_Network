import math #used for Euler's const - e

class NeuralNet:

    def __init__(self, initial_NN_file, dataset_file, output_file, num_epochs, learningRate, activationFunc):
        self.initial_NN_file = open(initial_NN_file,"r")
        self.dataset_file = open(dataset_file, "r")
        self.output_file = open(output_file, "w")
        self.num_epochs = num_epochs
        self.learningRate = learningRate
        self.activationFunc = activationFunc

    def loadNN(self):
        #load layer info.
        lines = self.initial_NN_file.read().splitlines()
        layers_info = [int(x) for x in lines[0].split()] #first line = number of elements for each layer
        num_layers = len(layers_info)

        #declare and load weight matrices
        weights = [[] for i in range(num_layers - 1)]  # stores matrices: num weights = num layers - 1

        for i in range(num_layers - 1):
            weights[i] = [[0 for x in range(layers_info[i] + 1)] for y in range(layers_info[i + 1])]  # row = next layer x col = current layer+1 (for bias)

        line_index = 1 #keeps track of index on lines

        for weight in weights:
            for i in range(len(weight)):
                line = lines[line_index].split()
                line_index+=1
                for j in range(len(weight[0])):
                    weight[i][j] = float(line[j])

        return layers_info, weights

    def loadDataset(self):
        lines = self.dataset_file.read().splitlines()
        num_examples, Ni, No = [int(x) for x in lines[0].split()] #only need these even if multiple layer NN

        dataset = [[0 for i in range(Ni+No)] for j in range (num_examples)]

        #load
        for row in range (num_examples):
            line = lines[1+row].split()
            for col in range(Ni+No):
                if col < Ni:
                    dataset[row][col] = float(line[col])
                else:
                    dataset[row][col] = int(line[col])

        return dataset

    def activation (self, sum):
        output=0
        if self.activationFunc == 0: #sigmoid
            output = float(1/(1+(math.e)**(-sum))) #sigmoid function
        elif self.activationFunc == 1: #ReLu
            output = max(0,sum)
        return output

    def gradience (self, input):
        output=0
        if self.activationFunc == 0: #sigmoid deriv
            output = float(self.activation(input) * (1 - self.activation(input)))  # deriv of sigmoid function
        elif self.activationFunc == 1: #ReLu deriv
            if input<0:
                output=0
            else:
                output=1
        return output

    def train (self):
        #load values from initial network and training set
        layers_info, weights = self.loadNN()
        dataset = self.loadDataset()

        #declare and load layers
        layers = [[] for i in range(2+2*(len(layers_info)-1))] #one input and output layer each, 2 layers (unactivated & activated) for each layer other than input layer: (input, Un, An, output)

        for i in range(len(layers_info)):
            if i==0: #input layer
                layers[0] = [0 for j in range(layers_info[0]+1)] #+1 for bias term
                layers[0][0] = -1 #bias term
            elif i==len(layers_info)-1: #output layer (no bias terms)
                layers[(i * 2) - 1] = [0 for j in range(layers_info[i])]  # unactivated output layer
                layers[(i * 2)] = [0 for j in range(layers_info[i])]  # activated output layer
                layers[len(layers) - 1] = [0 for j in range(layers_info[i])]  # correct output layer
            else:
                layers[(i*2)-1] = [0 for j in range(layers_info[i]+1)] #unactivated layer
                layers[(i*2)] = [0 for j in range(layers_info[i]+1)] #activated layer
                layers[(i*2)-1][0] = -1 #bias terms
                layers[(i*2)][0] = -1

        #change layers
        dU = [[] for i in range(len(layers_info)-1)]

        for i in range(len(dU)):
            if i==len(dU)-1:
                dU[i] = [0 for j in range(layers_info[i+1])] #output layer dU does not account for bias
            else:
                dU[i] = [0 for j in range(layers_info[i+1]+1)] #+1 accounts for bias term

        for epoch in range(self.num_epochs):
            for example in dataset:
                for node in range(layers_info[0]+layers_info[len(layers_info)-1]): #load input layer & output layers
                    if node<layers_info[0]:
                        layers[0][node+1] = example[node] #+1 avoids overwriting bias terms (+1 shift)
                    else:
                        layers[len(layers)-1][node-layers_info[0]] = example[node]

                # *Forward Propagation*
                for w in range(len(weights)):
                    for i in range(len(weights[w])): #for each row
                        weight_sum=0
                        for j in range(len(weights[w][0])): #for each column
                            weight_sum += (weights[w])[i][j] * layers[w*2][j]
                        if w==len(weights)-1: #last weight matrix (output x last hidden layer)
                            layers[(w*2)+1][i] = weight_sum  # enter into unactivated layer node; no bias term
                            layers[(w*2)+2][i] = self.activation((weight_sum))  # activate node
                        else:
                            layers[(w*2)+1][i+1] = weight_sum #enter into unactivated layer node; +1 to not overwrite bias(+1 shift)
                            layers[(w*2)+2][i+1] = self.activation((weight_sum)) #activate node

                #*Backward Propagation*
                for i in range(len(layers[len(layers)-2])): #estimate cost - iterate through last activation layer
                    dU[len(dU)-1][i] = self.gradience(layers[len(layers)-3][i]) * (layers[len(layers)-1][i] -  layers[len(layers)-2][i])

                for w in reversed(range(1,len(weights))):
                    for j in range(len(weights[w][0])): #for each column
                        weight_sum=0
                        for i in range(len(weights[w])):
                            weight_sum += (weights[w])[i][j] * (dU[w])[i]
                        dU[w-1][j] = self.gradience(layers[(w-1)*2+1][j]) * weight_sum

                #update weights
                for w in range(len(weights)):
                    for i in range(len(weights[w])):
                        for j in range(len(weights[w][0])):
                            if w==len(weights)-1: #final weight: bias not considered
                                (weights[w])[i][j] += self.learningRate * dU[w][i]*layers[w*2][j]
                            else:
                                (weights[w])[i][j] += self.learningRate * dU[w][i+1]*layers[w*2][j] #+1 accounts for bias

        #output to file
        for n in layers_info:
            self.output_file.write(str(n)+" ") if n!= layers_info[len(layers_info)-1] else self.output_file.write(str(n)+"\n")

        for w in range(len(weights)):
            for i in range(len(weights[w])):
                for j in range (len(weights[w][0])):
                    self.output_file.write("{:.3f}".format(weights[w][i][j]) + " ") if j!= len(weights[w][0])-1 else self.output_file.write("{:.3f}".format(weights[w][i][j]))
                self.output_file.write("\n")

    def test(self):
        # load values from initial network and training set
        layers_info, weights = self.loadNN()
        dataset = self.loadDataset()

        # declare and load layers
        layers = [[] for i in range(2 + 2 * (len(layers_info) - 1))]  # one input and output layer each, 2 layers (unactivated & activated) for each layer other than input layer: (input, Un, An, output)
        output_activ = len(layers) - 2

        for i in range(len(layers_info)):
            if i == 0:  # input layer
                layers[0] = [0 for j in range(layers_info[0] + 1)]  # +1 for bias term
                layers[0][0] = -1  # bias term
            elif i == len(layers_info) - 1:  # output layer (no bias terms)
                layers[(i * 2) - 1] = [0 for j in range(layers_info[i])]  # unactivated output layer
                layers[(i * 2)] = [0 for j in range(layers_info[i])]  # activated output layer
                layers[len(layers) - 1] = [0 for j in range(layers_info[i])]  # correct output layer
            else:
                layers[(i * 2) - 1] = [0 for j in range(layers_info[i] + 1)]  # unactivated layer
                layers[(i * 2)] = [0 for j in range(layers_info[i] + 1)]  # activated layer
                layers[(i * 2) - 1][0] = -1  # bias terms
                layers[(i * 2)][0] = -1

        # metric matrix
        No = layers_info[len(layers_info)-1]
        met_matrix = [[0 for i in range(No)] for j in range(8)]

        for example in dataset:

            for node in range(layers_info[0] + layers_info[len(layers_info) - 1]):  # load input layer & output layers
                if node < layers_info[0]:
                    layers[0][node + 1] = example[node]  # +1 avoids overwriting bias terms (+1 shift)
                else:
                    layers[len(layers) - 1][node - layers_info[0]] = example[node]

            # *Forward Propagation*
            for w in range(len(weights)):
                for i in range(len(weights[w])):  # for each row
                    weight_sum = 0
                    for j in range(len(weights[w][0])):  # for each column
                        weight_sum += (weights[w])[i][j] * layers[w * 2][j]
                    if w == len(weights) - 1:  # last weight matrix (output x last hidden layer)
                        layers[(w * 2) + 1][i] = weight_sum  # enter into unactivated layer node; no bias term
                        layers[(w * 2) + 2][i] = self.activation((weight_sum))  # activate node
                    else:
                        layers[(w * 2) + 1][
                            i + 1] = weight_sum  # enter into unactivated layer node; +1 to not overwrite bias(+1 shift)
                        layers[(w * 2) + 2][i + 1] = self.activation((weight_sum))  # activate node

            for i in range(No): # round output values
                if layers[output_activ][i] >= 0.5:
                    layers[output_activ][i] = 1
                else:
                    layers[output_activ][i] = 0

            for i in range(No):
                if layers[output_activ][i] == 0 and layers[len(layers)-1][i] == 0:  # D
                    met_matrix[3][i] += 1
                elif layers[output_activ][i] == 0 and layers[len(layers)-1][i] == 1:  # C
                    met_matrix[2][i] += 1
                elif layers[output_activ][i] == 1 and layers[len(layers)-1][i] == 0:  # B
                    met_matrix[1][i] += 1
                elif layers[output_activ][i] == 1 and layers[len(layers)-1][i] == 1:  # A
                    met_matrix[0][i] += 1

        # compute metrics for each class
        for i in range(No):
            A = met_matrix[0][i]
            B = met_matrix[1][i]
            C = met_matrix[2][i]
            D = met_matrix[3][i]
            if A + B + C + D != 0: met_matrix[4][i] = float((A + D) / (A + B + C + D))  # overall
            if A + B != 0: met_matrix[5][i] = float(A / (A + B))  # prec
            if A + C != 0: met_matrix[6][i] = float(A / (A + C))  # recall
            if met_matrix[5][i] + met_matrix[6][i] != 0: met_matrix[7][i] = float(
                (2 * met_matrix[5][i] * met_matrix[6][i]) / (met_matrix[5][i] + met_matrix[6][i]))

        # output to file (per class)
        for i in range(No):
            self.output_file.write(
                str(met_matrix[0][i]) + " " + str(met_matrix[1][i]) + " " + str(met_matrix[2][i]) + " " + str(
                    met_matrix[3][i]) + " " + "{:.3f}".format(met_matrix[4][i]) + " " + "{:.3f}".format(
                    met_matrix[5][i]) + " " + "{:.3f}".format(met_matrix[6][i]) + " " + "{:.3f}".format(
                    met_matrix[7][i]) + "\n")

        # micro-avg metrics
        sumA, sumB, sumC, sumD = 0, 0, 0, 0  # global metrics
        micro_accur, micro_prec, micro_recall, micro_F1 = 0, 0, 0, 0

        for i in range(No):
            sumA += met_matrix[0][i]
            sumB += met_matrix[1][i]
            sumC += met_matrix[2][i]
            sumD += met_matrix[3][i]

        if sumA + sumB + sumC + sumD != 0: micro_accur = float(
            (sumA + sumD) / (sumA + sumB + sumC + sumD))  # overall
        if sumA + sumB != 0: micro_prec = float(sumA / (sumA + sumB))  # prec
        if sumA + sumC != 0: micro_recall = float(sumA / (sumA + sumC))  # recall
        if micro_prec + micro_recall != 0: micro_F1 = float(
            (2 * micro_prec * micro_recall) / (micro_prec + micro_recall))

        self.output_file.write(
            str("{:.3f}".format(micro_accur)) + " " + str("{:.3f}".format(micro_prec)) + " " + str(
                "{:.3f}".format(micro_recall)) + " " + "{:.3f}".format(micro_F1) + "\n")

        # macro-avg metrics
        macro_accur, macro_prec, macro_recall, macro_F1 = 0, 0, 0, 0
        sum_accur, sum_prec, sum_recall = 0, 0, 0

        for i in range(No):
            sum_accur += met_matrix[4][i]
            sum_prec += met_matrix[5][i]
            sum_recall += met_matrix[6][i]

        macro_accur = float(sum_accur / No)
        macro_prec = float(sum_prec / No)
        macro_recall = float(sum_recall / No)
        if macro_prec + macro_recall != 0: macro_F1 = float(
            (2 * macro_prec * macro_recall) / (macro_prec + macro_recall))

        self.output_file.write(
            str("{:.3f}".format(macro_accur)) + " " + str("{:.3f}".format(macro_prec)) + " " + str(
                "{:.3f}".format(macro_recall)) + " " + "{:.3f}".format(macro_F1) + "\n")
