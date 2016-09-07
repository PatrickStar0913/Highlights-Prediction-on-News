import csv
with open('../data/page_url_and_highlights.csv', 'r') as f:
    reader = csv.reader(f)
    reader.next()
    selections = [x[4] for x in reader]
    # sentences = ["%s" % (x) for x in sentences]
f.close()
#
with open('../data/page_url_and_highlights.csv', 'r') as f:
    reader = csv.reader(f)
    reader.next()
    urls = [x[5] for x in reader]
    # sentences = ["%s" % (x) for x in sentences]
f.close()

with open('../data/page_url_and_highlights.csv', 'r') as f:
    reader = csv.reader(f)
    reader.next()
    domains = [x[7] for x in reader]
    # sentences = ["%s" % (x) for x in sentences]
f.close()
print len(domains)
j = 0
valid_domain = ['thedailybeast.com','ft.com','guardian.co.uk','thenewamerican.com','politicususa.com','newyorker.com',\
                'magazines.scholastic.com','infowars.com','cnn.com','theguardian.com','forbes.com','washingtonpost.com',\
                'huffingtonpost.com','theatlantic.com','dailymail.co.uk','reuters.com','nytimes.com','bloomberg.com',\
                'thehindu.com','bbc.co.uk','time.com','abcnews.go.com','investopedia.com','tumblr.com','dailykos.com','wired.com','stlmag.com','edition.cnn.com',\
                 'rt.com','foxnews.com','sciencedaily.com', 'usatoday.com','reddit.com', 'businessinsider.com', 'latimes.com',\
                 'telegraph.co.uk', 'psychologytoday.com','bbc.com']
valid_domain2 = 'en.wikipedia.org'

valid_domain_final = valid_domain

urls_list = []
writefile = file('../data/output/news_data.csv', 'w')
writer = csv.writer(writefile)
for i in range(0,len(domains)):
    # print domains
    if domains[i] in valid_domain_final:
    # if domains[i] == valid_domain2:
        urls_list.append(urls[i])
        j += 1
        writer.writerow((urls[i],selections[i],domains[i]))
    # if j == 5:
    #     break
print j
print len(urls_list)
print len(list(set(urls_list)))