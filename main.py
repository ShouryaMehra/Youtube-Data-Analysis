    from dotenv import load_dotenv
    import os
    from youtube_search import YoutubeSearch
    from youtubesearchpython import VideosSearch
    import urllib.parse
    from googleapiclient.discovery import build
    import pandas as pd
    import time
    import psutil
    import spacy
    import string
    from sklearn import neighbors
    import numpy as np
    import re
    from flask import Flask,jsonify,request,send_file
    import json

    # Define functions       
    app = Flask(__name__)
    app.config['JSON_SORT_KEYS'] = False
    # set env for secret key
    load_dotenv()

    def check_for_secret_id(request_data):    
        try:
            if 'secret_id' not in request_data.keys():
                return False, "Secret Key Not Found."
            
            else:
                if request_data['secret_id'] == secret_id:
                    return True, "Secret Key Matched"
                else:
                    return False, "Secret Key Does Not Match. Incorrect Key."
        except Exception as e:
            message = "Error while checking secret id: " + str(e)
            return False,message

    secret_id = os.getenv('AI_SERVICE_SECRET_KEY')

    def get_results(search_query, label_names_):
        api_key = 'AIzaSyD9vOiFeXCt26n8HxRV9QlO6ZJRyCk9OcM'  # your api key
        label_names = []
        for i in label_names_.split(','):
            label_names.append(i)

        search = VideosSearch(search_query, limit=50)

        print('extracting the data...')
        data = {'titles': [], 'duration': [], 'video_ids': [], 'tags': [], 'tags_string': [],
                'comment_count': [], 'dislike_count': [], 'view_count': [],
                'like_count': []}  # data structure for our reference and to store collected informatinon
        flag = False  # check variable
        results = search.result()['result']
        for ite in range(10):  # control loop
            if (flag):
                search.next()
            results = search.result()['result']
            # pprint(results)
            for i in results:  # parsing search results and appending to relevant values
                data['titles'].append(i['title'])
                data['duration'].append(i['duration'])
                url = i['link']
                url_data = urllib.parse.urlparse(url)
                query = urllib.parse.parse_qs(url_data.query)
                video_id = query["v"][0]
                data['video_ids'].append(video_id)
            if (flag == False):
                flag = True
        # print(url)

        # pprint(data)
        # print('\n\n length of data : ',len(data['duration']),'\n\n')
        youtube = build('youtube', 'v3', developerKey=api_key)

        for video_id in data['video_ids']:
            part_string = 'contentDetails,statistics,snippet'  # getting data of the following video parameters

            response = youtube.videos().list(
                part=part_string,
                id=video_id
            ).execute()
            # print('\n\n response : \n\n')
            # pprint(response)
            stat_dict = response['items'][0]['statistics']
            try:  # for comments if available
                data['comment_count'].append(int(stat_dict['commentCount']))
            except KeyError:  # for comments if not available
                # print('comment_count not available')
                data['comment_count'].append(None)
            try:  # for dislikes if available
                data['dislike_count'].append(int(stat_dict['dislikeCount']))
            except KeyError:  # for dislikes if not available
                # print('dislike count not available')
                data['dislike_count'].append(None)
            try:  # for view count if available
                data['view_count'].append(int(stat_dict['viewCount']))
            except KeyError:  # for view count if not available
                # print('view count not available')
                data['view_count'].append(None)
            try:  # for like count if available
                data['like_count'].append(int(stat_dict['likeCount']))
            except KeyError:  # for like count if not available
                # print('like count not available')
                data['like_count'].append(None)

            try:  # try block for videos with tags
                video_tags = response['items'][0]['snippet']['tags']
                string = ''
                for i in video_tags:
                    string = string + ' ' + i
                # print(i)
                # print('string : ',string,'\n')
                data['tags_string'].append(string)
                data['tags'].append(video_tags)
            except KeyError:  # try block for videos with tags
                print('tags is not available')
                data['tags'].append(None)
                data['tags_string'].append(None)

        # pprint(data)

        df = pd.DataFrame.from_dict(data)
        #df.to_csv(f'Youtube_{search_query}_extracted_data.csv', index=False)  # saving data to csv
        # print(
        #     f'Data extracted and save to your pc in excel file named Youtube_{search_query}_extracted_data in csv format !')

        docs = []
        labels = []
        # df = pd.read_csv(f"Youtube_{search_query}_extracted_data")  # Read excel with youtube data
        titles = df["tags_string"]  # capture tags column
        docs = titles
        labels = label_names

        # import string
        punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

        def clean_text(text):  # function to clean the text
            REPLACE_BY_SPACE_RE = re.compile("[/(){}/[/]/|@,;]")
            BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
            # STOPWORDS = set(stopwords.words('english'))
            REPLACE_IP_ADDRESS = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')
            text = text.replace('\n', ' ').lower()  # lowercase text
            text = REPLACE_IP_ADDRESS.sub('', text)
            text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
            text = BAD_SYMBOLS_RE.sub('', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
            # text = ' '.join([w for w in text.split() if not w in STOPWORDS])  # delete stopwords from text
            text = text.lower()
            text = text.translate(str.maketrans('', '', punctuations))
            #     text = text.replace('\n', ' ')
            text = ' '.join(text.split())  # remove multiple whitespaces
            return text

        list_titles = []
        print('Cleaning Data.....')
        for doc in docs:  # cleaning the text in tags and storing onto a different data structure
            doc = clean_text(str(doc))
            list_titles.append(doc)
        # list_titles

        nlp = spacy.load('en_core_web_lg')

        print('embeding the texts.....')

        def embed(tokens, nlp):
            """Return the centroid of the embeddings for the given tokens.

            Out-of-vocabulary tokens are cast aside. Stop words are also
            discarded. An array of 0s is returned if none of the tokens
            are valid.

            """

            lexemes = (nlp.vocab[token] for token in tokens)  # creates tokens of words

            vectors = np.asarray([
                lexeme.vector
                for lexeme in lexemes
                if lexeme.has_vector
                   and not lexeme.is_stop
                   and len(lexeme.text) > 1
            ])

            if len(vectors) > 0:
                centroid = vectors.mean(axis=0)
            else:
                width = nlp.meta['vectors']['width']  # typically 300
                centroid = np.zeros(width)

            return centroid

        centroid_list = []
        for doc in list_titles:  # finding position of each tag in vector space
            tokens = doc.split(' ')
            centroid = embed(tokens, nlp)  # embed the vectors in the vector space
            centroid_list.append(centroid)
        #     print(centroid.shape)
        # (300,)

        label_vectors = np.asarray([
            embed(label.split(' '), nlp)  # embed user definedtags/label in vector space
            for label in label_names
        ])

        print('classifying tags....')
        labels_list = []
        # finding distance of tags/labels from the collected video tags
        neigh = neighbors.NearestNeighbors(n_neighbors=1)
        neigh.fit(label_vectors)
        for centroid in centroid_list:
            closest_label = neigh.kneighbors([centroid], return_distance=False)[0, 0]  # find closest label for given tag
            labels_list.append(label_names[closest_label])
        df["tag_classified"] = (labels_list)

        saved = []
        for doc in docs:
            doc = str(doc)
            tokens = doc.split(' ')
            centroid = embed(tokens, nlp)  # find centroid of tags from video in vector space

            label_vectors = np.asarray([
                embed(label.split(' '), nlp)
                for label in label_names
            ])
            neigh = neighbors.NearestNeighbors(n_neighbors=1)  # find the closest tags/labels to the input youtube tag
            neigh.fit(label_vectors)

            closest_label = neigh.kneighbors([centroid], return_distance=False)[0, 0]  # find tag for classification
            saved.append(label_names[closest_label])

        df["tag1"] = (saved)
        print('classified !')

        df['duration_in_seconds'] = df['duration']
        # converting time of the video into seconds
        for idx, i in enumerate(df['duration']):
            #     print(type(i))
            i = str(i)
            if (i == 'nan' or i == None or i == 'None'):
                continue
            time_ = []
            for j in i.split(':'):
                time_.append(int(j))
            #     print(time)
            time_in_seconds = 0
            if (len(time_) == 1):
                time_in_seconds = time_[0]
            if (len(time_) == 2):
                time_in_seconds = time_[0] * 60 + time_[1]
            if (len(time_) == 3):
                time_in_seconds = time_[0] * 60 * 60 + time_[1] * 60 + time_[2]
            df['duration_in_seconds'][idx] = time_in_seconds

        our_dict = df.tag_classified.value_counts()
        duration_sum = df.groupby("tag_classified")["duration_in_seconds"].sum()
        result = {}
        for key_1, value_1, in our_dict.items():
            result[key_1] = int(duration_sum[key_1] // value_1)
        result = dict(sorted(result.items(), key=lambda x: x[0]))
        # converting unix format to xx Hr yy m zz s
        for key_1, value_1 in result.items():
            unix = int(result[key_1])
            hours = str(unix // 3600)
            minutes = str((unix % 3600) // 60)
            seconds = str(((unix % 3600) % 60))
            result[key_1] = hours + 'Hr ' + minutes + 'm ' + seconds + 's'

        no_comments = df.groupby("tag_classified")["comment_count"].sum()
        no_views = df.groupby("tag_classified")["view_count"].sum()
        no_dislike = df.groupby("tag_classified")["dislike_count"].sum()
        no_like = df.groupby("tag_classified")["like_count"].sum()

        # result_2= {'Framework':[],'demo':[],'snake':[],'tutorial':[]}
        result_2 = {}
        for i in labels:
            result_2[i] = []
        my_list = [our_dict, result, no_comments, no_like, no_dislike, no_views]
        for key, value in result_2.items():
            for i in my_list:
                result_2[key].append(i[key])

        result_2['index'] = ['Number of videos', 'Avg Duration', 'Comment Count', 'Likes Count', 'DisLikes Count',
                             'Views Count']
        output_2 = pd.DataFrame.from_dict(result_2, orient='index').T
        output_2.set_index(keys=['index'], inplace=True)

        return output_2, df


    @app.route('/youtube_classifier',methods=['POST'])  #main function
    def main():
        params = request.get_json()
        input_query=params["data"]
        search_query = input_query[0]['query']
        label_names_ = input_query[0]['tags']
        key = params['secret_id']

        request_data = {'secret_id' : key}
        secret_id_status,secret_id_message = check_for_secret_id(request_data)
        print ("Secret ID Check: ", secret_id_status,secret_id_message)
        if not secret_id_status:
            return jsonify({'message':"Secret Key Does Not Match. Incorrect Key.",
                            'success':False}) 
        else:
            final_dfs=get_results(search_query,label_names_)#final dataframes will be stored here
            final_dict={}
            final_dict["Analysis"]=json.loads(final_dfs[0].to_json(orient="index"))
            final_dict["video_details"]=json.loads(final_dfs[1].to_json(orient="index"))
        return jsonify(final_dict)