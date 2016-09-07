import matplotlib.pyplot as plt
import csv
import operator
from collections import OrderedDict

with open('page_url_and_highlights.csv', 'r') as f:
    reader = csv.reader(f)
    reader.next()
    domains = [x[7] for x in reader]
    # sentences = ["%s" % (x) for x in sentences]
f.close()
print len(domains)
print len(list(set(domains)))

domain_num = {}

for i in domains:
    if domain_num.has_key(i):
        domain_num[i] += 1
    else:
        domain_num[i] = 1
# print domain_num

# d_ascending = sorted(domain_num.items(), key=operator.itemgetter(1))
# print d_ascending
# value_list = []
# for i in domain_num:
#     # print i, domain_num[i]
#     value_list.append(domain_num[i])
# print max(value_list)
# print sorted(list(set(value_list)))
# for i in list(set(value_list)):

domain_dis = {}
for i in domain_num:
    if domain_dis.has_key(domain_num[i]):
        domain_dis[domain_num[i]] += 1
    else:
        domain_dis[domain_num[i]] = 1
# print domain_dis

d_ascending = sorted(domain_dis.items(), key=operator.itemgetter(1))
for i in d_ascending:
    print i[0],",",i[1]
