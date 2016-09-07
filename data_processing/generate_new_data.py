# -*- coding: utf-8 -*-
import csv
import requests
from goose import Goose
import difflib
import re



def is_ascii(s):
    return all(ord(c) < 128 for c in s)

# 读取url
with open('data/17/news_17_wiki.csv','rb') as csvfile:
     reader = csv.reader(csvfile)
     urls = [row[0] for row in reader]
csvfile.close()

# 读取selection
with open('data/17/news_17_wiki.csv','rb') as csvfile:
     reader = csv.reader(csvfile)
     selections = [row[1] for row in reader]
csvfile.close()

print len(urls)
print len(selections)

# 去除非英语
new_urls = []
new_selections = []
for i in range(0,len(urls)):
    selection = selections[i]
    if is_ascii(selection):
        new_urls.append(urls[i])
        new_selections.append(selection)
urls = new_urls
selections = new_selections

# 创建url-selections pair
url_selection = {}
for i in range(0,len(urls)):
    if url_selection.has_key(urls[i]):
        url_selection[urls[i]].append(selections[i])
    else:
        url_selection[urls[i]] = []
        url_selection[urls[i]].append(selections[i])
# print url_selectio

print len(urls)
print len(selections)
urls = list(set(urls))
print urls[0]
print urls[1]
print len(urls)
print len(selections)


#
sum = 0
# writefile = file('data/28/news_gra_sen_title_location.csv', 'w')
# # myfile = codecs.open('csv_train.csv', 'r', encoding='ascii', errors='ignore')
# writer = csv.writer(writefile)
writefile = file('data/28/news_gra_sen_title_wiki.csv', 'w')
writer = csv.writer(writefile)

# writefile2 = file('data/28/news_doc_sen_title_location.csv', 'w')
# # myfile = codecs.open('csv_train.csv', 'r', encoding='ascii', errors='ignore')
# writer3 = csv.writer(writefile3)
writefile2 = file('data/28/news_doc_sen_title_wiki.csv', 'w')
writer2 = csv.writer(writefile2)
j = 0

strinfo = re.compile('\n')
# b = strinfo.sub('',a)
# m = 0
for url in urls:
    j += 1
    try:
        # 获取网页内容
        response = requests.get(url, timeout=20)
        g = Goose()
        article = g.extract(url=url)
        content = article.cleaned_text
        title = article.title

        super_super_flag = 0
        blocks = content.split('\n\n')
        block_i = 0

        # data3 = []
        data2 = []

        # 把 content从一个list转换成一个长文本
        flatten_content = strinfo.sub('',content)
        flatten_content = "".join(flatten_content.encode('ascii','ignore').lower())

        # 每一个段落
        for block in blocks:

            # data = []
            data1 = []

            super_flag = 0
            block_i += 1
            # print block_i,"  ",len(blocks),"  ",block_i/floatlen(blocks))
            block = block.strip()
            # 分离句子
            sens = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', block)
            sens = [" ".join(sen.encode('ascii','ignore').lower().split()) for sen in sens]
            selections = [" ".join(sen.encode('ascii','ignore').lower().split()) for sen in url_selection[url]]
            title = title.encode('ascii','ignore').lower()
            block = block.encode('ascii','ignore').lower()
            for sen in sens:
                flag = 0
                if len(sen.split(" ")) > 3:
                    for selection in selections:
                        if difflib.SequenceMatcher(None, sen, selection).ratio() >= 0.6 or difflib.SequenceMatcher(None,selection, sen).ratio() >= 0.6:
                            flag = 1
                            super_flag = 1
                    # 纪录
                    data1.append((flag, url, block, sen, title,block_i, block_i/float(len(blocks))))
                    data2.append((flag, url,flatten_content, sen, title,block_i, block_i/float(len(blocks))))
            # 如果该段落中有selection,则存入
            if super_flag == 1:
                writer.writerows(data1)
                # writer2.writerows(data2)
                super_super_flag = 1
            # for sen in sens:
            #     if len(sen.split()) > 3:
            #         data.append((flag, url, sen, title, block_i,block_i/float(len(blocks))))

        if super_super_flag == 1:
            print j, "   ", "valid", "   ", url
            # print data
            writer2.writerows(data2)
            sum += 1
        else:
            print j,"   ","not valid","   ", url
    except Exception as e:
        print j,"   ","Exception",   "   ", url

writefile.close()
writefile2.close()