from django.apps import AppConfig
import pandas as pd

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

class MachinelearningConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'machinelearning'
    model = None
    items_scaler = None

    def initialize():
        df = pd.read_csv("https://drive.google.com/uc?id=1fxQdiCgnc3CwUhU91omaU5nf9Rj-XKru", encoding='ISO-8859-1',  index_col='Unnamed: 0')

        #Clean up the columns that will be unused
        data = df.drop(['Item URL'],axis=1)

        #Merge Original Price Columns with NaN Values with Discounted Price
        data['Original Price'] = data['Original Price'].fillna(data['Discounted Price'])

        #Clean Up NaN Values on Columns
        data['Discount'] = data['Discount'].fillna('0%off')
        data['Review'] = data['Review'].fillna(0.0)
        data['Brand Name'] = data['Brand Name'].fillna('No Brand')

        #Convert Items Sold column to take integer
        data['Items Sold'] = data['Items Sold'].str.replace('sold','')
        data['Items Sold'] = data['Items Sold'].str.replace(' ','')
        data['Items Sold'] = data['Items Sold'].str.replace(',','')
        data = data[data["Items Sold"].str.contains("\+")==False]
        data['Items Sold'] = data['Items Sold'].fillna('0')
        data['Items Sold'] = data['Items Sold'].astype('int')

        #Normalize the 'Attributes' Column
        data['Attributes'] = data['Attributes'].str.lower()

        import re
        import string

        data_nlp = data.copy(True)
        description_raw = data_nlp['Description'].tolist()

        # Remove URLs
        def removeURLs(text):
            return re.sub(r'http\S+', '', text)

        description_clean = [removeURLs(t) for t in description_raw]

        # Remove emoticons
        url_emote = "https://drive.google.com/uc?id=1HDpafp97gCl9xZTQWMgP2kKK_NuhENlE"
        df_emote = pd.read_pickle(url_emote)

        def removeEmoticons(text):
            for emot in df_emote:
                text = re.sub(u'('+emot+')', "", text)
                text = text.replace("<3", "" ) # not included in emoticons database
            return text

        description_clean = [removeEmoticons(t) for t in description_clean]

        # Convert to lowercase
        description_clean = [t.lower() for t in description_clean]

        # Remove punctuation
        description_clean = [t.translate(str.maketrans("", "", string.punctuation)) for t in description_clean]

        # Remove non-ascii characters
        def removeSpecial(text):
            return re.sub(r'[^\x00-\x7f]', r'', text) 

        description_clean = [removeSpecial(t) for t in description_clean]

        # Add description back to data frame
        data_nlp['Description'] = description_clean

        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize

        descriptions = data_nlp['Description'].tolist()

        # Tokenizes description
        descriptions_tokenized = []
        for desc in descriptions:
            # Tokenization
            words = word_tokenize(desc)

            # Remove stopwords
            filtered_words = [word for word in words if word.lower() not in stopwords.words('english')]

            # Convert back to sentence
            filtered_sentence = " ".join(filtered_words)
            descriptions_tokenized.append(filtered_sentence)

        data_nlp['Description'] = descriptions_tokenized

        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        descriptions = data_nlp['Description'].tolist()

        # Lemmatize descriptions
        descriptions_lemmatized = []
        for desc in descriptions:
            words = desc.split()
            lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
            lemmatized_sentence = " ".join(lemmatized_words)
            descriptions_lemmatized.append(lemmatized_sentence)

        data_nlp['Description'] = descriptions_lemmatized

        special_attributes = ['male', 'man', 'men'] # Special checking --> Was done manually
        attributes = ['cotton', 'silk screen', 'loose', 'collar', 'crew neck', 'drifit', 'small', 'medium', 'large', 'round neck', 'short sleeve', 'unisex', 'casual', 'adult',
                    'polyester', 'graphic', 'vneck', 'turtleneck', 'slim', 'spandex', 'blouse', 'jersey', 'sport', 'fitness', 'sweater', 'vinyl', 'micro fiber', 'nylon',
                    'combed', 'knitted', 'floral', 'plain', 'boy', 'girl', 'kid', 'female', 'woman', 'women', 'regular fit', 'uniform', 'crop top', 'denim', 'washable', 'chubby',
                    'relaxed fit', 'semi fit', 'lady', 'size chart', 'tank shirt', 'freesize', 'cartoon', 'sublimation', 'dress', 'decorated', 'retro', 'oversized', 'street wear',
                    'sweat sgurt', 'long sleeve', 'boutique', 'dry fit', 'athletic', 'sando', 'nylon', 'emboss', 'elastic', 'direct film', 'mesh', 'youth', 'muscle', 'stretchable',
                    'seamless rib', 'baby', 'plus size', 'xl', 'xxl', '2xl', 'xs', 'polyester', 'sbust', 'slength', 'lbust', 'llength', 'mbust', 'mlength', 'xlbust', 'xllength', 'xxlbust', 'xxllength']
        color_attributes = ['colors', 'color', 'white', 'red', 'blue', 'gray', 'black', 'orange', 'green', 'beige', 'pink', 'caramel', 'yellow', 'purple', 'maroon']
        brand_attributes = ['moso', 'unifit', 'puma', 'fruit loom', 'hanes', 'original', 'champion', 'black horse', 'yalex', 'disney', 'star war', 'petrol', 'aiden', 'stylistic mr lee',
                            'tommy hilfiger', 'crown', 'bny', 'hanford', 'sletic', 'nike', 'team manila', 'rough rider']
        sport_attributes = ['sports', 'basketball', 'cycling', 'swimming', 'running', 'playing', 'gym', 'football', 'soccer', 'jogging', 'workout', 'yoga']
        potential_attributes = ['urban', 'summer', 'asian', 'american', 'european', 'modern', 'authentic', 'environmental', 'ecofriendly', 'durable', 'durability', 'comfortable', 'breathable',
                                'natural', 'organic', 'premium', 'fashion', 'inklock', 'korean', 'vintage', 'licensed', 'anime', 'fashionable', 'trendy', 'free shipping', 'recycled',
                                'odor protection', 'gift', 'holiday', 'travel', 'party', 'daily', 'work', 'school', 'beauty', 'independent', '2 layer', 'double layer', 'reinforced',
                                'sweat absorbent', 'tagless', 'non itch', 'high quality', 'compression support', 'quick dry', 'wicking', 'hang dry', 'good hygiene', 'semi body fit',
                                'sun protection', 'anti uv', 'valentine', 'birthday', 'occasion', 'quarterturned', 'collar on neck', 'ultra soft', 'soft']

        ## IMPORTANT -- I

        all_attributes = attributes + color_attributes + brand_attributes + sport_attributes + potential_attributes
        all_attributes = list(dict.fromkeys(all_attributes))  # Removes duplicates from list

        data_attr_extract = data_nlp.copy(True)

        # Gets attributes found in description 
        def getAttributes(text, attribute_list):
            attributes = []
            text = text.replace(" ", "")

            for a in attribute_list:
                a = a.replace(" ", "")
                if text.find(a) != -1:
                    attributes.append(a)
            return attributes

        # Gets special attributes
        def getSpecialAttribute(text, attr, prefix):
            while True:
                ind = text.find(attr)
                # Attribute not found
                if ind == -1:
                    return -1
                # Attribute first/second character
                if ind <= 1:
                    return attr
                # Checks if attribute starts with prefix
                if text[ind - len(prefix): ind] == prefix:
                    return -1
                return attr

        # Appends attribute if found
        def getSpecialHandler(text, attr, prefix, attr_list):
            res = getSpecialAttribute(text, attr, prefix)
            if res == -1:
                return
            attr_list.append(res)


        descriptions = data_attr_extract['Description'].tolist()
        new_attributes = [getAttributes(t, all_attributes) for t in descriptions]

        # Gets the special attributes
        for i in range(len(descriptions)):
            desc = descriptions[i]
            attr_list = new_attributes[i]
            getSpecialHandler(desc, "male", "fe", attr_list)
            getSpecialHandler(desc, "man", "wo", attr_list)
            getSpecialHandler(desc, "men", "wo", attr_list)

        # Returns true if text has a number
        def hasNum(text):
            return bool(re.search(r'\d', text))

        # Append new attributes to existing attributes
        attributes_feature = data_attr_extract['Attributes'].tolist()
        for i in range(len(attributes_feature)):
            attr_text = attributes_feature[i]  
            new_attr = new_attributes[i]

            if type(attr_text) == type("hello"):
                # Converts attributes from string into a list
                attr_list = attr_text[1:-1].split(",")
                to_del = []
                for j in range(len(attr_list)):
                    attr_list[j] = attr_list[j].strip(" '")

                    # Cleans attributes
                    attr_list[j] = attr_list[j].translate(str.maketrans("", "", string.punctuation)) 
                    attr_list[j] = attr_list[j].replace(" ", "")
                    if hasNum(attr_list[j]):
                        to_del.append(j)

                # Deletes unneeded attributes
                for j in range(len(to_del)-1, -1, -1):
                    del_ind = to_del[j]
                    del attr_list[del_ind]

                # Adds new attributes
                for a in new_attr:
                    if not a in attr_list:
                        attr_list.append(a)
                attributes_feature[i] = attr_list

            else:
                # Attributes is empty
                attributes_feature[i] = new_attr

        # Adding attributes back into data frame
        data_attr_extract['Attributes'] = attributes_feature

        # Display new attributes
        changes = pd.DataFrame({'Attributes': data_attr_extract['Attributes']})
        changes.style.set_properties(**{"text-align": "left", "min-width": "400px", "white-space": "normal"})

        #Replace Unnecessary Characters from the List
        onehot_enc = data_attr_extract.dropna(subset=['Attributes'])
        onehot = list(onehot_enc['Attributes'])

        #Perform One Hot Encoding
        mlb = MultiLabelBinarizer()
        x = mlb.fit_transform(onehot)
        oh_data = pd.DataFrame(x,columns=mlb.classes_, index=onehot_enc.index)

        #Merge One Hot Encoding to the "Data" Frame
        data_oh = data_attr_extract.join(oh_data)

        data_final = data_oh.drop(['Name', 'Discount', 'Original Price', 'Rating Score', 'Review', 'Description', 'Attributes'], axis=1)
        data_final['Brand Name'] = data_final['Brand Name'].str.lower() 
        data_final['Location'] = data_final['Location'].str.lower()
        data_final = pd.get_dummies(data_final, columns=['Brand Name', 'Location'], prefix=['', ''], prefix_sep='')

        cols = []
        unique_cols = set()
        count = 1
        # https://datascience.stackexchange.com/questions/41448/how-to-rename-columns-that-have-the-same-name
        for col in data_final.columns:
            if col in unique_cols:
                cols.append(f'{col}_{count}')
                count+=1
                continue
            unique_cols.add(col)
            cols.append(col)
        data_final.columns = cols

        data_final['durable'] = data_final['durable'] | data_final['durability']
        data_final['graphic'] = data_final['graphic'] | data_final['graphicprint']
        data_final['jersey'] = data_final['jersey'] | data_final['jerseystretch']
        data_final['male'] = data_final['male'] | data_final['man'] | data_final['men']
        data_final['natural'] = data_final['natural'] | data_final['naturalfibers']
        data_final['polo'] = data_final['polo'] | data_final['polocollar']
        data_final['female'] = data_final['female'] | data_final['woman'] | data_final['women'] | data_final['womens'] | data_final['lady']
        data_final['blue corner'] = data_final['blue corner'] | data_final['blue corner comfort wear tee-shirts']
        data_final['gildan'] = data_final['gildan'] | data_final['gildan t-shirt']
        data_final['durable'] = data_final['durable'] | data_final['durability']
        data_final['2layer'] = data_final['2layer'] | data_final['doublelayer']

        data_final['bny'] = data_final['bny'] | data_final['bny_1']
        data_final['crown'] = data_final['crown'] | data_final['crown tshirt'] | data_final['crown_3']
        data_final['champion'] = data_final['champion'] | data_final['champion_2']
        data_final['disney'] = data_final['disney'] | data_final['disney_4']
        data_final['korean'] = data_final['korean'] | data_final['korean_7']
        data_final['nike'] = data_final['nike'] | data_final['nike_8']
        data_final['petrol'] = data_final['petrol'] | data_final['petrol_10']
        data_final['organic'] = data_final['organic'] | data_final['organic_9']
        data_final['puma'] = data_final['puma'] | data_final['puma_11']
        data_final['unifit'] = data_final['unifit'] | data_final['unifit_13']
        data_final['yalex'] = data_final['yalex'] | data_final['yalex_14']
        data_final['hanford'] = data_final['hanford'] | data_final['hanford_6']
        data_final['hanes'] = data_final['hanes'] | data_final['hanes_5'] | data_final['hanes tagless tee']
        data_final['sletic'] = data_final['sletic'] | data_final['sletic_12']
        data_final['unifit'] = data_final['unifit'] | data_final['unifit_13']

        data_final['stylisticmrlee'] = data_final['stylisticmrlee'] | data_final['stylistic mr. lee'] | data_final['lee']
        data_final['inspi'] = data_final['inspi'] | data_final['inspi basics']
        data_final['drifit'] = data_final['drifit'] | data_final['dryfit']
        data_final['kentucky'] = data_final['kentucky'] | data_final['kentucky white sando']
        data_final['local brand'] = data_final['local brand'] | data_final['local brand/ 100% made in ph'] | data_final['local brands'] | data_final['local made ph']
        data_final['moose gear'] = data_final['moose gear'] | data_final['moose girl']
        data_final['no brand'] = data_final['no brand'] | data_final['no'] | data_final['no band'] | data_final['no brand - pg'] | data_final['no brand m'] | data_final['no brand name'] | data_final['no brand no brand'] | data_final['no brand no brand no brand'] | data_final['no brandds'] | data_final['no brandname'] | data_final['no brands'] | data_final['no brandss'] | data_final['nobrand'] | data_final['there is no brand'] | data_final['â¤ï¸no brand'] | data_final['â¤ï¸no brandâ¤ï¸'] | data_final['â¤ï¸nobrand']
        data_final['2xl'] = data_final['2xl'] | data_final['xxl']
        data_final['cottonspandexblend'] = data_final['cotton'] & data_final['spandex'] | data_final['cottonspandexblend']
        data_final['tankshirt'] = data_final['tankshirt'] | data_final['tanktops'] | data_final['sando']

        data_final['semifit'] = data_final['semifit'] | data_final['semibodyfit']

        data_final = data_final.drop(['durability', 
                                'graphicprint', 
                                'jerseystretch', 
                                'man', 
                                'men', 
                                'naturalfibers', 
                                'polocollar', 
                                'woman', 
                                'women', 
                                'womens', 
                                'blue corner comfort wear tee-shirts', 
                                'crown tshirt', 
                                'bny_1',
                                'champion_2',
                                'crown_3',
                                'disney_4',
                                'hanes_5',
                                'hanford_6',
                                'korean_7',
                                'nike_8',
                                'organic_9',
                                'petrol_10',
                                'puma_11',
                                'sletic_12',
                                'unifit_13',
                                'yalex_14',
                                'hanes tagless tee', 
                                'inspi basics', 
                                'kentucky white sando', 
                                'local brand/ 100% made in ph', 
                                'local brands', 
                                'local made ph', 
                                'moose girl', 
                                'no', 
                                'no band', 
                                'no brand - pg', 
                                'no brand m', 
                                'no brand name', 
                                'no brand no brand', 
                                'no brand no brand no brand', 
                                'no brandds', 
                                'no brandname', 
                                'no brands', 
                                'no brandss', 
                                'nobrand', 
                                'there is no brand', 
                                'â¤ï¸no brand', 
                                'â¤ï¸no brandâ¤ï¸', 
                                'â¤ï¸nobrand',
                                'lady',
                                'dryfit',
                                'doublelayer',
                                'stylistic mr. lee',
                                'xxl',
                                'lee',
                                'tanktops',
                                'sando',
                                'semibodyfit'
                                ], axis=1)

        for col in data_final.columns:
            if data_final[col].sum() < 3:
                data_final.drop(col, axis=1, inplace=True)


        data_final.drop(['tshirts', 'premium', 'comfortable'], axis=1, inplace=True)

        import scipy.stats as stats
        # removing outliers
        # credits to https://stackoverflow.com/questions/46245035/pandas-dataframe-remove-outliers
        def Remove_Outlier_Indices(df):
            Q1 = df.quantile(0.35)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            trueList = ~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR)))
            return trueList

        data_final = data_final[Remove_Outlier_Indices(data_final['Discounted Price'])]

        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, MinMaxScaler

        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import mean_absolute_error
        from sklearn.metrics import r2_score

        import numpy as np

        X = data_final.drop('Discounted Price', axis=1)
        y = data_final['Discounted Price']

        items_scaler = StandardScaler()

        # generating test and train data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=23)

        X_train['Items Sold'] = items_scaler.fit_transform(X_train['Items Sold'].values.reshape(-1, 1))
        X_test['Items Sold'] = items_scaler.transform(X_test['Items Sold'].values.reshape(-1, 1))

        # import the model
        from sklearn.ensemble import RandomForestRegressor

        rf_reg = RandomForestRegressor(n_estimators=500, max_features='sqrt', max_depth=30, random_state=17)

        # fit then predict
        rf_reg.fit(X_train, y_train.ravel())

        MachinelearningConfig.model = rf_reg
        MachinelearningConfig.items_scaler = items_scaler