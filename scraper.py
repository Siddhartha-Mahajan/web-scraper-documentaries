
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
import pandas as pd
documentary_titles = []
documentary_links = []
image_links = []
comment_divs_list = []
for i in range(0,25):
    url = f"https://www.documentaryarea.com/category/Science/page/{i}/"
    response = requests.get(url)
    html_content = response.content
    soup = BeautifulSoup(html_content, "html.parser")

    # Find all the <h2> tags on the page
    h2_tags = soup.find_all("h2")

    # Extract links within each <h2> tag and add them to the list
    for h2 in h2_tags:
        link = h2.find("a")  # Find the first <a> tag within the <h2>
        if link:
            documentary_links.append("https://www.documentaryarea.com"+link["href"])
            k="https://www.documentaryarea.com"+link["href"]
            
            parsed_url = urlparse(k)
            query_params = parse_qs(parsed_url.query)

            # Extract the title parameter
            try:
                title = query_params["title"][0]
                title = title.replace("+", " ")  # Replace '+' with spaces
                documentary_titles.append(title)
            except:
                title = "Not Available"
                title = title.replace("+", " ")  # Replace '+' with spaces
                documentary_titles.append(title)
            

# Find all <img> tags with title and alt attributes
            img_tags = soup.find_all("img", alt=True, title=True)

            for img in img_tags:
                if img["alt"] == title:
                    # Get the value of the "src" attribute (image link)
                    try:
                        image_link = img["data-src"]
                        image_links.append("https://www.documentaryarea.com"+image_link)
                    except:
                        image_link = img["src"]
                        image_links.append("https://www.documentaryarea.com"+image_link)
           

for i in documentary_links:
    response = requests.get(i)
    html_content = response.content
    soup = BeautifulSoup(html_content, "html.parser")


    comment_divs = soup.find_all("div", class_="comments")

    for comment_div in comment_divs:
        description_meta = comment_div.find("meta", itemprop="description")
        upload_date_meta = comment_div.find("meta", itemprop="uploadDate")

        if description_meta and upload_date_meta:
            content = description_meta.find_next_sibling(string=True).strip()
            comment_divs_list.append(content)



f=[]        
for i in range(len(documentary_titles)):
    f.append("Science")               
               
df = pd.DataFrame(list(zip(documentary_links, documentary_titles,f,image_links,comment_divs_list)),
               columns =['links', 'titles','category','image_links','synopsis'])
print(df)
df.to_csv('file6.csv')





