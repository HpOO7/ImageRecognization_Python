import numpy;

class newNeurons:
    #Initialization of all the necessary element called while creating the object of neurons class
    def __init__(self,inode,onode,hnode,lrate):
        #Number of node in input layer, hidden layer ,output layer
        self.inode = inode
        self.hnode = hnode
        self.onode = onode

        #Learning rate of neurons
        self.lrate = lrate

        #Martrix contains weights of links between input layer and hidden layer(wij) and wjk are weights between hidden layer and output layer
        self.wij = numpy.random.rand(self.hnode,self.inode)-0.5
        self.wjk = numpy.random.rand(self.onode,self.hnode)-0.5
        # self.wij = numpy.random.normal(0,0,pow(self.hnode,-0.5)).(self.hnode,self.inode)
        # self.wjk = numpy.random.normal(0,0,pow(self.onode,-0.5)).(self.onode,self.hnode)

    #Activation function (Sigmoid function) 1/(1+e^-x)
    def activation(self,matrix):
            for i in range(len(matrix)):
                for j in range(len(matrix[i])):
                    matrix[i][j] = 1/(1+pow(2.71828,-matrix[i][j]))
                    pass
                pass
            return matrix
           

    def train(self,input_matrix,output_matrix):
        #convering input and output list into two dimensional array
        inputs = numpy.array(input_matrix,ndmin=2).T
        outputs = numpy.array(output_matrix,ndmin=2).T

        #layer 2 input ,layer 2 output ( Signal after activation function )
        hidden_input = numpy.dot(self.wij,inputs)
        hidden_output = self.activation(hidden_input)

        #layer 3 input ,layer 3 output ( Signal after activation function )
        final_input = numpy.dot(self.wjk,hidden_output)
        final_output = self.activation(final_input)

        #final error
        output_error = outputs - final_output
        #hidden error
        hidden_error = numpy.dot(numpy.transpose(self.wjk),output_error)

        #updation of links between hidden layer and output layer
        self.wjk+=self.lrate*numpy.dot(output_error*final_output*(1-final_output),numpy.transpose(hidden_output))
        #updation of links between input layer and  hidden
        self.wij+=self.lrate*numpy.dot(hidden_error*hidden_output*(1-hidden_output),numpy.transpose(inputs))

    #Checking output of neural network
    def check(self,input_matrix):
        #convering input and output list into two dimensional array
        inputs = numpy.array(input_matrix,ndmin=2).T

        #layer 2 input ,layer 2 output ( Signal after activation function )
        hidden_input = numpy.dot(self.wij,inputs)
        hidden_output = self.activation(hidden_input)

        #layer 3 input ,layer 3 output ( Signal after activation function )
        final_input = numpy.dot(self.wjk,hidden_output)
        final_output = self.activation(final_input)

        # print(final_output)
        return numpy.argmax(final_output)
        pass

inode = 784
hnode = 100
onode = 10
learning_rate = 0.3

#NEURONS INSTANCE
con = newNeurons(inode,onode,hnode,learning_rate)

#csv file training data
file_data = open("mnist_train.csv","r")
data_list = file_data.readlines()
file_data.close()
print("****************training start********************")
for record in data_list:
    values = record.split(",")
    inputs = []
    for i in range(1,inode+1):
        k = (int(values[i])/255.0)*0.98 + 0.1
        inputs.append(k)
    pass
    target = numpy.zeros(onode) + 0.01
    target[int(values[0])] = 0.98
    ##check
    # print("inputs")
    # print(len(inputs))
    # print("targets")
    # print(target)
    # print("len",len(target))
    con.train(inputs,target)
    pass

print("****************training complete********************")

test_data = open("mnist_test.csv","r")
test_list = test_data.readlines()
test_data.close()
print("****************Test Results***********")
#counters
count = 0
count1 = 0
for record in test_list:
    values = record.split(",")
    inputs = []
    for i in range(1,inode+1):
        k = (int(values[i])/255.0)*0.98 + 0.1
        inputs.append(k)
    pass
    ans = con.check(inputs)
    print("Real Ans : ",int(values[0])," Netwok Ans : ",ans)
    
    #Counting the numeber if wrong output and input
    if ans == int(values[0]):
        count=count+1
    else:
        count1=count1+1
    pass
accuracy = (count/(count+count1))*100.0
print(accuracy)  
