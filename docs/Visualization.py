'''
Function:
	Visualization the test results while training
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import re
import matplotlib.pyplot as plt


f = open('train.log', 'r')
contents = f.read()
results = re.findall(r'Accuracy of epoch (.*?) is (.*?)\.\.\.', contents)
epochs = []
accs = []
for result in results:
	epochs.append(int(result[0]))
	accs.append(float(result[1]))
plt.title('Test accuracy vary according to epoch of train')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.plot(epochs, accs, 'b')
plt.savefig('vis.jpg')
plt.show()