#processes images in each rock, paper, scissor folder:
    #each image gets resized, grey-scaled, and 1Dim.(based on pixels)
    #each 1D values are outputted as a line with a space between the values
    #at end of each line the possible output values (order: rock paper scissor) are outputted with 1 showing which is the correct output and 0 showing which are the incorrect outputs
    #the lines are shuffled to generate randomness and well-rounded distribution

from PIL import Image, ImageOps
import matplotlib.image as image
import os, sys, cv2

#defined paths
rock_path = '/Users/chrislee/Documents/Junior Year/AI/Neural Network/RPS_Dataset/Images/Rock/'
paper_path = '/Users/chrislee/Documents/Junior Year/AI/Neural Network/RPS_Dataset/Images/Paper/'
scissor_path = '/Users/chrislee/Documents/Junior Year/AI/Neural Network/RPS_Dataset/Images/Scissor/'
paths = [rock_path, paper_path, scissor_path]
train = open("train_RPS.txt","w")
test = open("test_RPS.txt","w")

#functions
def resize(path):
    for item in os.listdir(path):
        if os.path.isfile(path+item):
            image = Image.open(path+item).convert('L')
            f, e = os.path.splitext(path+item)
            image = image.resize((200,200), Image.Resampling.LANCZOS)
            image.save(f + '.png', 'PNG', quality=90)

def pixel(path):
    iter = 0
    for item in os.listdir(path):
        if iter<3:
            output=test
        else:
            output = train

        if os.path.isfile(path+item):
            img = cv2.imread(path+item,cv2.IMREAD_GRAYSCALE)
            for i in range(200):
                for j in range(200):
                    output.write("{:.3f}".format(float(img[i][j]/255))+" ")
        if path==rock_path:
            output.write("1 0 0"+"\n")
        elif path==paper_path:
            output.write("0 1 0" + "\n")
        elif path==scissor_path:
            output.write("0 0 1" + "\n")
        iter+=1

train.write("34 40000 3"+"\n")
test.write("9 40000 3"+"\n")

for path in paths:
    resize(path)
    pixel(path)