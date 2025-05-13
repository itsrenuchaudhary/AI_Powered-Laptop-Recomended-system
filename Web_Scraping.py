import pandas as pd
import requests
from bs4 import BeautifulSoup

product_name = []
Prices = []
Description = []
Ratings = []

for i in range(2,60):
    url = "https://www.flipkart.com/search?q=laptops&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=off&as=off&page="+str(i)
    
    r = requests.get(url)
    
    soup = BeautifulSoup(r. text, "lxml")
    box = soup.find("div",class_ = "DOjaWF gdgoEp")
    
    names = box.find_all("div", class_ = "KzDlHZ")
    # print(names)
    
    for i in names:
        name = i.text
        product_name.append(name)
    
    #print(product_name)
    
    
    prices = box.find_all("div", class_ = "hl05eU")
    for i in prices:
        name = i.text
        Prices.append(name)
    
    #print(Prices)
    
    desc = box.find_all("ul",class_ = "G4BRas")
    
    for i in desc:
        name = i.text
        Description.append(name)
    
    #print(Description)
    
    ratings = box.find_all("div", class_ = "XQDdHH")
    
    for i in ratings:
        name = i.text
        Ratings.append(name)

#print(len(Ratings))
min_len = min(len(product_name), len(Prices), len(Description), len(Ratings))
product_name = product_name[:min_len]
Prices = Prices[:min_len]
Description = Description[:min_len]
Ratings = Ratings[:min_len]


df = pd.DataFrame({"Product Name":product_name,"Price":Prices,"Description":Description,"Ratings":Ratings})
#print(df)

df.to_csv("D:\/flipkartt_data.csv")