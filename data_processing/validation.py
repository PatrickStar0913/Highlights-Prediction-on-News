import csv
import random
import sys
import csv

csv.field_size_limit(sys.maxsize)
with open('data/28/news_gra_sen_title_loc_sample.csv', 'r') as f:
    reader = csv.reader(f)
    reader.next()
    labels = [(int(x[0]),x[1],x[2],x[3],x[4],x[5],x[6]) for x in reader]
    # labels = ["%s" % (x) for x in labels]
f.close()
print len(labels)
print labels[6]
# no_train = random.sample(range(0, len(labels)), int(len(labels)*0.8))
# print "len",len(no_train)
# no_test = []
# for i in range(0,len(labels)):
#     if i not in no_train:
#         no_test.append(i)
# print "success!"
# print "len",len(no_train)
# print len(no_test)
# # # #
# m = 0
# text_file = open("no_train.txt", "w")
# for i in no_train:
#     m += 1
#     text_file.write(str(i)+ "\n")
# text_file.close()
# print m
# j = 0
# text_file = open("no_test.txt", "w")
# for i in no_test:
#     j += 1
#     text_file.write(str(i)+"\n")
# text_file.close()
# print j

no_trains = []
i = 0
with open('no_train.txt', 'rb') as infile:
    for line in infile:
        i += 1
        no_trains.append(int(line))
infile.close()
print "test:",i
#
#
no_tests = []
with open('no_test.txt', 'rb') as infile:
    for line in infile:
        no_tests.append(int(line))
infile.close()


writefile = file('data/28/news_gra_sen_title_loc_sample_train.csv', 'w')
# myfile = codecs.open('csv_train.csv', 'r', encoding='ascii', errors='ignore')
writer = csv.writer(writefile)
print "test",len(no_trains)
j = 0
for i in no_trains:
    j += 1
    writer.writerow((labels[i][0], labels[i][1],labels[i][2],labels[i][3],labels[i][4],labels[i][5],labels[i][6]))
writefile.close()
print j
