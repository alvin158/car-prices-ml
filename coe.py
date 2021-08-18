from bs4 import BeautifulSoup
import pickle
from datetime import datetime
import datetime as dt
from urllib.request import Request, urlopen
import pandas as pd

req = Request("https://www.sgcarmart.com/news/writeup.php?AID=70&gclid=CjwKCAjw9aiIBhA1EiwAJ_GTSqqQBEROLZE2TlmuhGN5XYzNmYzD15Y7WuWpINvQQgNpuCBFV36yFBoCt5UQAvD_BwE")
webpage = urlopen(req).read()

title = []
category = []
quota = []
premium = []

soup = BeautifulSoup(webpage)

# find title
for t in soup.find_all("h2"):
    t_clean = t.text.strip()[0:].replace(',', '')
    title.append(t_clean)

# find category
for c in soup.find_all("div", class_="fixed_h coe_sub_header cat_desc"):
    c_clean = c.text.strip()[0:5].replace(',', '')
    category.append(c_clean)

# find quota
for q in soup.find_all("div", limit=4, class_="coe_sub_header h_27 font_18"):
    q_clean = q.text.strip()[0:].replace(',', '')
    quota.append(q_clean)

# find premium
for p in soup.find_all("div", class_="coe_sub_header font_bold h_27"):
    p_clean = p.text.strip()[1:].replace(',', '')
    premium.append(p_clean)

data = pd.DataFrame()
data["Category"] = category
data["Quota"] = quota
data["Premium"] = premium


def coe_prices():
    coe_prices_json = data.to_json(orient="records")
    return coe_prices_json


data2 = pd.DataFrame()
data2["Title"] = title


def coe_title():
    coe_title_json = data2.to_json(orient="index")
    return coe_title_json


def coe_test():
    firstRow = data.iloc[0:1]
    coe_price_json = firstRow.to_json(orient="index")
    return coe_price_json
