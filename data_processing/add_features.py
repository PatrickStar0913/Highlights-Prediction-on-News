import re, math
from collections import Counter
import csv
import itertools


# read data
with open('../data/output/news_gra_sen_title_sample.csv', 'r') as f:
    reader = csv.reader(f)
    reader.next()
    labels = [x[0] for x in reader]
    labels = ["%s" % (x) for x in labels]
f.close()

with open('../data/output/news_gra_sen_title_sample.csv', 'r') as f:
    reader = csv.reader(f)
    reader.next()
    sens = [x[3] for x in reader]
    sens = ["%s" % (x) for x in sens]
f.close()

with open('../data/output/news_gra_sen_title_sample.csv', 'r') as f:
    reader = csv.reader(f)
    reader.next()
    titles = [x[4] for x in reader]
    titles = ["%s" % (x) for x in titles]
f.close()

with open('../data/output/news_gra_sen_title_sample.csv', 'r') as f:
    reader = csv.reader(f)
    reader.next()
    locs = [x[6] for x in reader]
    locs = ["%s" % (x) for x in locs]
f.close()

print locs

WORD = re.compile(r'\w+')

def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)


cosines = []
for i in range(0, len(sens)):
    vector1 = text_to_vector(sens[i])
    vector2 = text_to_vector(titles[i])
    cosine = get_cosine(vector1, vector2)
    cosines.append(cosine)

writefile = file('../data/output/news_gra_sen_title_sample_sim.csv', 'w')
# myfile = codecs.open('csv_train.csv', 'r', encoding='ascii', errors='ignore')
writer = csv.writer(writefile)

for i in range(0, len(sens)):
    if float(locs[i]) <= 0.2:
        # print 1
        writer.writerow((labels[i], sens[i], titles[i], cosines[i],1,0,0,0,0))
    elif float(locs[i]) <=0.4:
        # print 2
        writer.writerow((labels[i], sens[i], titles[i], cosines[i],0,1,0,0,0))
    elif float(locs[i]) <=0.6:
        # print 3
        writer.writerow((labels[i], sens[i], titles[i], cosines[i],0,0,1,0,0))
    elif float(locs[i]) <=0.8:
        # print 4
        writer.writerow((labels[i], sens[i], titles[i], cosines[i],0,0,0,1,0))
    else:
        # print 5
        writer.writerow((labels[i], sens[i], titles[i], cosines[i],0,0,0,0,1))
writefile.close()
