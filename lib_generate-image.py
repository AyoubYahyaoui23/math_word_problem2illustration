import os
import time
import sqlite3
import requests
from bs4 import BeautifulSoup
from requests import get
import openai
from openai.error import RateLimitError

api_keys = [
    # "wajdi": "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",       # used up
    # "salah": "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",       # used up
    # "ayoub": "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"        # expired
    # "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",       # 5$ used up
    # "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",       # 5$ used up
    # "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",       # 18$ expires tomorrow 01/04/2021 # used up
    # "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",       # 5$ used up
    # "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",       # 5$ used up
    # "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",       # 5$ used up
    # "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",       # 5$ used up
    # "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",       # 5$ used up
    # "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",       # 18$ used up
    # "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",       # 5$ used up
    # "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",       # 5$
    # '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',       # 5$
    "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",  # 5$
]


def get_img_from_dalle(lemma, token):
    global api_keys

    success = False
    while True:
        openai.api_key = "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"
        # openai.api_key = random.choice(list(api_keys))

        try:
            response = openai.Image.create(
                prompt=lemma + " clipart",
                n=1,
                size="1024x1024")
            break

        except RateLimitError:
            print(f"Rate limit exceeded with key {openai.api_key}. Waiting and retrying...")
            time.sleep(60)
            continue

        except openai.error.AuthenticationError as e:
            if "402" in str(e):
                print(f"DALL-E API key {openai.api_key} used up")
                api_keys.remove(openai.api_key)
                continue
        except openai.error.InvalidRequestError as e:
            if str(e) == "Billing hard limit has been reached":
                print("Billing hard limit has been reached for key " + openai.api_key)
                api_keys.remove(openai.api_key)
                continue

    r = get(response["data"][0]["url"])
    success = True
    connection = sqlite3.connect(r'D:/Documents/NLP/math_problems_tokens.sqlite')
    cur = connection.cursor()
    params = (
        sqlite3.Binary(r.content), lemma, token['chunk'], token['start_position'], token['end_position'])
    sql = f"""
    INSERT INTO tokens(img1, lemma, token, start_index, end_index) VALUES (?, ?, ?, ?, ?)
    """
    cur.execute(sql, params)
    connection.commit()
    connection.close()

    with open('images/' + lemma + '.png', 'wb') as f:
        f.write(r.content)
        print('"images/' + lemma + '.png" ' + 'DALL-E Image has been saved successfully')
    return success


def get_img_from_clipart(lemma, token):
    success = False
    url = f'https://www.clipartmax.com/so/{lemma}/'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    images = soup.select('.img-part')
    a = requests.get(images[0]['href'])
    soup2 = BeautifulSoup(a.content, 'html.parser')
    download_link = soup2.findAll('a', href=lambda href: href and href.startswith('https://www.clipartmax.com/download/'))[0]['href']
    download_page = requests.get(download_link)
    soup3 = BeautifulSoup(download_page.content, 'html.parser')
    image_link2 = soup3.findAll('a', href=lambda href: href and href.startswith('https://www.clipartmax.com/png/full/'))[0]['href']
    r = requests.get(image_link2)


    if not os.path.exists(f'images'):
        os.mkdir(f'images')

    success = True
    connection = sqlite3.connect(r'D:/Documents/NLP/math_problems_tokens.sqlite')
    cur = connection.cursor()
    params = (
        sqlite3.Binary(r.content), lemma, token['chunk'], token['start_position'], token['end_position'])
    sql = f"""
    INSERT INTO tokens(img2, lemma, token, start_index, end_index) VALUES (?, ?, ?, ?, ?)
    """
    cur.execute(sql, params)
    connection.commit()
    connection.close()

    with open(f'images/{lemma}.png', 'wb') as f:
        f.write(r.content)
        print('"images/' + lemma + '.png" ' + 'Clipart Image has been saved successfully')
    return success

if __name__ == '__main__':
    # url = f'https://www.clipartmax.com/so/boy/'
    # response = requests.get(url)
    # soup = BeautifulSoup(response.content, 'html.parser')
    # images = soup.select('.img-part')
    # a = requests.get(images[0]['href'])
    # soup2 = BeautifulSoup(a.content, 'html.parser')
    # download_link = soup2.findAll('a', href=lambda href: href and href.startswith('https://www.clipartmax.com/download/'))[0]['href']
    # download_page = requests.get(download_link)
    # soup3 = BeautifulSoup(download_page.content, 'html.parser')
    # image_link2 = soup3.findAll('a', href=lambda href: href and href.startswith('https://www.clipartmax.com/png/full/'))[0]['href']
    # image = requests.get(image_link2)
    # with open(f'boy.png', 'wb') as f:
    #     f.write(image.content)
    print("hello")
    pass
