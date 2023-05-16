import os
import re
import gender_guesser.detector as gender
from transformers import pipeline
import matplotlib.colors as mcolors
from image import get_img_from_clipart, get_img_from_dalle
import sqlite3 as sq
from flair.data import Sentence
from transformers import AutoModelForTokenClassification, AutoTokenizer

def extract_numbers(text):
    """
    Extract all numbers from a text using regular expressions.
    Args: text (str): The input text to extract numbers from.
    Returns: list: A list of numbers extracted from the text.
    """
    # Define a regular expression pattern to match number words from zero to twenty

    # Use a regular expression to extract numbers from the text
    numbers = re.findall(r'\d+\.\d+|\d+', text)
    return numbers


from random import choice as random_choice


def get_tokens(input, tagger_chunk, tagger_ner):
    # load tagger

    detector = gender.Detector()

    sentence_chunk = Sentence(input)
    sentence_ner = Sentence(input)
    l = []
    # predict NER tags
    tagger_chunk.predict(sentence_chunk)
    for entity in sentence_chunk.get_spans('np'):
        l.append({'chunk': entity.text, 'start_position': entity.start_position, 'end_position': entity.end_position,
                  'pos': entity.get_label().value, })
    tagger_ner.predict(sentence_ner)
    for index, i in enumerate(l):
        split_l = []
        for j in sentence_ner.get_spans('ner'):
            if j.end_position in range(i['start_position'], i['end_position'] + 1):
                split_l.append({'chunk': j.text, 'start_position': j.start_position, 'end_position': j.end_position,
                                'pos': j.get_label().value, })
        if len(split_l) >= 2:
            l.pop(index)
            l += split_l
    # for index, i in enumerate(l):
    #     split_l = []
    #     for j in sentence_ner.get_spans('ner'):
    #         if j.end_position in range(i['start_position'],
    #                                    i['end_position'] + 1) and ((j.get_label().value == 'PERSON' and i['chunk'] != j.text)  or (j.get_label().value == 'CARDINAL' and i['chunk'] != j.text)):
    #             i['chunk'] = i['chunk'].replace(j.text, '')

    # iterate over entities and print
    for entity in sentence_ner.get_spans('ner'):
        for i in l:
            if entity.text in i['chunk'] and entity.start_position >= i['start_position'] and entity.end_position <= i[
                'end_position']:
                if entity.get_label().value == 'ORDINAL':
                    i['ner'] = entity.get_label().value
                    i["value"] = entity.text
                elif entity.get_label().value == 'CARDINAL':
                    i["value"] = extract_numbers(i["chunk"])[0]
                    print(i["value"])
                elif entity.get_label().value == 'PERSON':
                    i['ner'] = entity.get_label().value
                    i["gender"] = detector.get_gender(f'{i["chunk"]}')
                    if 'female' in i['gender']:
                        i["drawable_tag"] = 'female'
                    elif 'male' in i["gender"]:
                        i["drawable_tag"] = 'male'
                    else:
                        i["drawable_tag"] = random_choice(['male', 'female'])

                else:
                    i['ner'] = entity.get_label().value
                    try:
                        i["value"] = extract_numbers(i["chunk"])[0]
                    except:
                        pass
    l.sort(key=lambda x: x['start_position'])
    return l


def add_drawable_label(sentence, tokens, nlp, model_path):
    loaded_model = AutoModelForTokenClassification.from_pretrained(model_path)
    loaded_tokenizer = AutoTokenizer.from_pretrained(model_path)
    loaded_pipe = pipeline("ner", model=loaded_model, tokenizer=loaded_tokenizer, aggregation_strategy="simple")
    out2 = loaded_pipe(
        sentence)

    for i in tokens:
        for j in out2:
            if j['word'] in i['chunk'] and j['start'] >= i['start_position'] and j['end'] <= i['end_position'] \
                    or i['chunk'] in j['word'] and i['start_position'] >= j['start'] and i['end_position'] <= j['end']:
                i['drawable_tag'] = j['entity_group']

    for i in tokens:
        docc = nlp(i['chunk'])
        words = [token for token in docc if not token.is_stop and not token.is_punct and not token.is_space]

        if 'drawable_tag' in i.keys():
            try:
                if not words:
                    i['lemma'] = i['chunk']
                else:
                    i['lemma'] = str(max(words, key=len)).lower().strip()
            except:
                print('error getting lemma of', i['chunk'])
                pass

    return tokens


def get_color_with_lemma(token):
    """
    if the chunk has a color name in it, add it to the lemma
    :param token: dict with chunk and lemma
    :return: the given token with added color in lemma if it existed in chunk
    """
    colors = list(mcolors.CSS4_COLORS.keys())
    pattern = re.compile(r'\b(' + r'|'.join(colors) + r')\b\s*', flags=re.IGNORECASE)
    matches = re.findall(pattern, token["chunk"])
    if matches:
        if matches[0] not in token['lemma']:
            token['lemma'] = matches[0] + ' ' + token['lemma']
    return token

def get_image_with_api(token, list_to_html, get_img_from_api):
    """
    get image with api
    :param token:
    :param list_to_html: list of tuples with image path and chunk to append to
    :param get_img_from_api: function to get the image with
    :return: list of tuples with added image path and chunk
    """
    print('Api call for : ', token['lemma'])
    image = get_img_from_api(token['lemma'], token)

    if image:
        list_to_html.append((f'images/{token["lemma"]}.png', token['chunk']))
    else:
        print('NO IMAGE FOUND! for : ', token['lemma'])
    return list_to_html


def fix_folder_path(folder_path):
    """
    Fix the folder path to be absolute and normalized
    :param folder_path:
    :return:
    """
    # Normalize the folder path to remove redundant separators and references to parent directories
    folder_path = os.path.normpath(folder_path)

    # Check if the path is relative or absolute
    if not os.path.isabs(folder_path):
        # Convert relative path to absolute path
        folder_path = os.path.abspath(folder_path)

    return folder_path


def get_images(tokens, image_folder_path):
    """
    get images for drawable tokens
    :param tokens: dict of tokens with certain keys
    :return: list of tuples with image path and chunk
    """
    image_token_list = []
    conn = sq.connect(r'D:/Documents/NLP/math_problems_tokens.sqlite')
    cur = conn.cursor()
    image_folder_path = fix_folder_path(image_folder_path)

    for i in tokens:
        i = get_color_with_lemma(i)

        DrawableTagExists = 'drawable_tag' in i.keys()

        if DrawableTagExists:
            print(i['lemma'])

            DrawableTagIsGender = i['drawable_tag'] in ['male', 'female']
            DrawableTagIsClipart = i['drawable_tag'] == 'clipart'
            DrawableTagIsDalle = i['drawable_tag'] == 'dalle'

            if DrawableTagIsGender:
                image_token_list.append((image_folder_path + f'/{i["drawable_tag"]}.jpg', i['chunk']))
                continue

            elif DrawableTagIsDalle:
                img = 'img1'
                get_img_from_api = get_img_from_dalle

            elif DrawableTagIsClipart:
                img = 'img2'
                get_img_from_api = get_img_from_clipart

            cur.execute(f"""select {img} from tokens where lemma='{i["lemma"]}' or token='{i["chunk"]}'""")
            images = cur.fetchall()
            # print(imgs)

            if images:
                if images[0][0]:
                    image = images[0][0]
                    print('saving image of ', i['lemma'], ' in ', image_folder_path)
                    with open(image_folder_path + f'/{i["lemma"]}.png', "wb") as f:
                        f.write(image)
                        image_token_list.append((image_folder_path + f'/{i["lemma"]}.png', i['chunk']))
                else:
                    image_token_list = get_image_with_api(i, image_token_list, get_img_from_api)

            else:
                image_token_list = get_image_with_api(i, image_token_list, get_img_from_api)

    return image_token_list


def word_exits_after_number_pattern(sentence):
    """
    This function takes a sentence with number and returns a regex pattern that matches the sentence and a word after the number
    :param sentence: sentence to process
    :return: compiled regex pattern
    """
    pattern = r"(\d+)"
    result = re.sub(pattern, r"\1" + r'(?:\\b.*?\\b)' + r'?', sentence)
    return re.compile(result)


def produce_sentence_with_images(sentence, image_list, image_folder_path):
    """

    :param sentence: input sentence
    :param image_list: list of tuples of image paths and words
    :return:  list of tuples containing image paths and tokens forming the sentence
    """
    input_string = sentence
    input_list = image_list
    image_folder_path = fix_folder_path(image_folder_path)
    output_list = []

    start_pos = 0
    for drawable_word in input_list:
        if re.findall(r'\d+', drawable_word[1]):
            pattern = word_exits_after_number_pattern(drawable_word[1])
            pattern = re.compile(pattern)
        else:
            pattern = re.compile(re.escape(drawable_word[1]))

        for match in re.finditer(pattern, input_string[start_pos:]):
            if start_pos < match.start() + start_pos:
                output_list.append(input_string[start_pos:match.start() + start_pos])

            output_list.append((drawable_word[0], match.group(0)))
            start_pos = match.end() + start_pos
            break

    if start_pos < len(input_string):
        output_list.append(input_string[start_pos:])

    final_list = []
    for i in output_list:
        if not isinstance(i, tuple):
            if i.strip():
                final_list.append((image_folder_path + '/pngwing.png', i.strip()))
        else:
            final_list.append(i)
        if 'pngwing.png' in final_list[-1][0]:
            final_list[-1] = ('', final_list[-1][1])
    return final_list


def create_html(imaged_sentence, output_html_file_path):
    """
    Create HTML file from input list containing image paths and text
    :param imaged_sentence: list of tuples containing image paths and tokens forming the sentence
    :param output_html_file_path: path to output HTML file
    :return:
    """

    # Sample input list containing image paths and text
    input_list = imaged_sentence

    # Function to create an HTML element based on input
    def create_html_element(item):
        item = list(filter(None, item))
        if len(item) == 2:
            img_path, text = item
            img_element = f'<img src="{img_path}" alt="{text}">'
            text_element = f'<p>{text}</p>'
            return f'<div>{img_element}{text_element}</div>'
        else:
            text = item[0]
            arrow_element = '<span>&#8594;</span>'
            return f'<div>{arrow_element}<p>{text}</p></div>'

    # Generate HTML content
    html_content = [create_html_element(item) for item in input_list]
    html_content = '\n'.join(html_content)

    # Generate full HTML
    html = f"""<!DOCTYPE html>
    <html>
    <head>
    <style>
        img {{
            width: 200px;
            height: auto;
        }}
        div {{
            display: inline-block;
            vertical-align: top;
            margin: 10px;
        }}
        span {{
            font-size: 24px;
            margin-right: 10px;
        }}
    </style>
    </head>
    <body>
    {html_content}
    </body>
    </html>"""

    # Save HTML to a file
    with open(output_html_file_path, 'w') as f:
        f.write(html)

    print('HTML file generated: ', output_html_file_path)
    return html

def replace_numbers_with_numerals(text):
    # Define a regular expression pattern to match number words from zero to twenty
    pattern = r'\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\b'

    # Define a dictionary to map number words to numerals
    number_mapping = {
        'zero': '0',
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'ten': '10',
        'eleven': '11',
        'twelve': '12',
        'thirteen': '13',
        'fourteen': '14',
        'fifteen': '15',
        'sixteen': '16',
        'seventeen': '17',
        'eighteen': '18',
        'nineteen': '19',
        'twenty': '20'
    }

    # Replace number words with numerals using regular expression substitution
    replaced_text = re.sub(pattern, lambda match: number_mapping.get(match.group(0).lower(), match.group(0)), text)

    return replaced_text


def get_html_from_sentence(sentence, tagger_chunk, tagger_ner, nlp, model_math, output_html_file_path="output.html", image_folder_path='images'):
    sentence = replace_numbers_with_numerals(sentence)

    tokens = get_tokens(sentence, tagger_chunk, tagger_ner)
    tokens = add_drawable_label(sentence, tokens, nlp, model_math)

    image_list = get_images(tokens, image_folder_path)
    imaged_sentence = produce_sentence_with_images(sentence, image_list, image_folder_path)
    html = create_html(imaged_sentence, output_html_file_path)
