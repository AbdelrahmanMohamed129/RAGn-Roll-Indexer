import grequests
from lxml import html
import time

# read the top k pages from file "top_k.txt"
with open("top_k_ids.txt", "r") as f:
    pages = f.readlines()

# generate urls for each page, TODO: handle unique urls
urls = dict()
for page in pages:
    actual_page = int(page)//100
    row = int(page)%100 + 1
    url = "https://huggingface.co/datasets/Cohere/wikipedia-22-12-simple-embeddings/viewer/default/train?p=" + str(actual_page)
    if url not in urls:
        urls[url] = list()
    urls[url].append(row)

start = time.time()


# send requests to urls
rs = (grequests.get(u) for u in urls)
responses = grequests.map(rs)

top_k_data = list()

for response in responses:
    content = response.text
    parsed = html.fromstring(content)
    # get the rows in urls[response.url]
    rows = parsed.xpath("/html/body/div[1]/main/div/div/div/div[2]/div/table/tbody/tr")
    for row in urls[response.url]:
        text = rows[row-1].xpath("td[3]/div/div/text()")[0]
        top_k_data.append(text)

end = time.time()
print("Time taken to crawl the top K: ",end - start) #time taken to run the code

# write the data to file
with open("top_k_data.txt", "w") as f:
    for data in top_k_data:
        f.write(data + "\n")