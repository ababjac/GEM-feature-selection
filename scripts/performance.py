import pandas as pd
import matplotlib.pyplot as plt
import sys

filename1 = sys.argv[1]

f = open(filename1, 'r')
lines = f.readlines()

shuffled = []
phylum_name = []
AUC = []
Accuracy = []
Precision = []
Recall = []

for i in range(0, len(lines), 7):
    shuffled.append('No')
    phylum_name.append(lines[i].partition(':')[0])
    AUC.append(float(lines[i+1].partition(':')[2].replace(' ', '').replace('\n', '')))
    Accuracy.append(float(lines[i+2].partition(':')[2].replace(' ', '').replace('\n', '')))
    Precision.append(float(lines[i+3].partition(':')[2].replace(' ', '').replace('\n', '')))
    Recall.append(float(lines[i+4].partition(':')[2].replace(' ', '').replace('\n', '')))


filename2 = sys.argv[2]

f = open(filename2, 'r')
lines = f.readlines()

for i in range(0, len(lines), 7):
    shuffled.append('Yes')
    phylum_name.append(lines[i].partition(':')[0])
    AUC.append(float(lines[i+1].partition(':')[2].replace(' ', '').replace('\n', '')))
    Accuracy.append(float(lines[i+2].partition(':')[2].replace(' ', '').replace('\n', '')))
    Precision.append(float(lines[i+3].partition(':')[2].replace(' ', '').replace('\n', '')))
    Recall.append(float(lines[i+4].partition(':')[2].replace(' ', '').replace('\n', '')))

df = pd.DataFrame()
df['Shuffled'] = shuffled
df['Phylum Name'] = phylum_name
df['AUC'] = AUC
df['Accuracy'] = Accuracy
df['Precision'] = Precision
df['Recall'] = Recall

#print(df)

df.to_csv(sys.argv[3])
