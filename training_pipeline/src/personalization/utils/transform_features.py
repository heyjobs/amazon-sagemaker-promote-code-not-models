import re
import json
import datetime
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from rapidfuzz import fuzz
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler

employment_mapping = {
    "apprenticeship": "apprenticeship",
    "azubi": "apprenticeship",
    "trainee": "trainee",
    "permanent employment": "permanent_employment",
    "festanstellung": "permanent_employment",
    "self employment": "self_employment",
    "selbständig": "self_employment",
    "internship": "internship",
    "praktikum": "internship",
    "working student": "working_student",
    "werkstudent": "working_student",
    "dual study": "dual_study",
    "duales studium": "dual_study",
    "marginal employment": "marginal_employment",
    "geringfügige beschäftigung": "marginal_employment",
    "minijob": "mini_job",
    "secondary activity": "secondary_activity",
    "nebentätigkeit": "secondary_activity",
    "temporary help": "temporary_help",
    "aushilfe": "temporary_help",
    "other": "other",
    "andere": "other"
}

# Mapping from German and English phrases to desired English forms
working_hours_mapping = {
    "teilzeit": "part_time",
    "vollzeit": "full_time",
    "part-time": "part_time",
    "full-time": "full_time",
    "part": "part_time",
    "full": "full_time",
    "time": "part_time"
}

num_cols = ["jobseeker_preference_salary_min",
     "job_salary_amount",
     "job_salary_range_min",
     "job_salary_range_max"
     ]

def dict_to_list(x):
    if pd.isna(x):  # Check if the value is NaN
        return []
    result = [f'{elem.strip()}' for elem in str(x).strip('{}').split(',') if elem.strip() != ""]
    return result

# Adjusted function to convert a string list of employment types to a list of enum values
def convert_to_enum(string_list):
    list_of_strings = json.loads(string_list.replace("'", "\""))
    return [employment_mapping[item.lower()] for item in list_of_strings]

# Function to convert list of German/English phrases to list of desired English forms
def convert_to_english(string_list):
    list_of_strings = json.loads(string_list.replace("'", "\""))
    result = [working_hours_mapping[item.lower()] for item in list_of_strings]
    return result if result else []

def handle_nan_and_split(series):
    return series.fillna('').str.replace('full_time_part_time', 
                                         'full_time,part_time').str.split(',').apply(lambda x: [] if x == [''] else x)

def process_dataframe(df):
    
    df.loc[:,'job_allows_easy_apply'] = (df.loc[:,'job_allows_easy_apply'] == 't')
    df.loc[:,'job_open_for_career_changers'] = (df.loc[:,'job_open_for_career_changers'] == True)

    df.loc[:,'job_kldb_code'] = df.loc[:,'job_kldb_code'].fillna(0).astype(int).astype(str).str.zfill(4)
    df.loc[:,'jobseeker_preference_job_title_kldb_code'] = df.loc[:,'jobseeker_preference_job_title_kldb_code'].fillna(0).astype(int).astype(str).str.zfill(4)
    df.loc[:,'job_product_type'] = df.loc[:,'job_product_type'].apply(lambda x: x.lower() if isinstance(x, str) else x)
    df.loc[:,'job_origin'] = df.loc[:,'job_origin'].apply(lambda x: x.lower() if isinstance(x, str) else x)

    df.loc[:,'jobseeker_preference_employment_types'] = df.loc[:,'jobseeker_preference_employment_types'].apply(dict_to_list)
    df.loc[:,'job_employment_types'] = df.loc[:,'job_employment_types'].apply(dict_to_list)

    df.loc[:,'last_feed_employment_type_filter'] = df.loc[:,'last_feed_employment_type_filter'].apply(convert_to_enum)
    df.loc[:,'last_feed_working_hours_filter'] = df.loc[:,'last_feed_working_hours_filter'].apply(convert_to_english)

    df.loc[:,'jobseeker_preference_working_hours'] = handle_nan_and_split(df.loc[:,'jobseeker_preference_working_hours'])
    df.loc[:,'job_working_hours'] = handle_nan_and_split(df.loc[:,'job_working_hours'])

    df.loc[:,'common_employment_types'] = df.apply(lambda row: list(set(row['job_employment_types'] + row['last_feed_employment_type_filter'])), axis=1)
    df.loc[:,'common_working_hours'] = df.apply(lambda row: list(set(row['job_working_hours'] + row['last_feed_working_hours_filter'])), axis=1)

    max_date = datetime.date.today()
    scale = 60
    df.loc[:,'gauss_recency'] = df.loc[:,'derived_tstamp'].apply(lambda date: np.round(np.exp(- ((max_date - date).days ** 2 / (2 * scale ** 2))),4))
    
    df.loc[:, num_cols] = df.loc[:, num_cols].fillna(0.0)
    
    return df


def process_traindata_only(df):
    
    three_months_ago = df.loc[:,'derived_tstamp'].max() - datetime.timedelta(days=90)
    top_50_company_uids = df[df['derived_tstamp'] >= three_months_ago]['job_company_uid'].value_counts().nlargest(50).index
    
    df.loc[~df['job_company_uid'].isin(top_50_company_uids), 'job_company_uid'] = 'other'
    return df


class SimilarityScoreCalculator(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.legal_entity = [
                "a.g",
                "a. g",
                "ag",
                "co",
                "de",
                "e.k",
                "e. k",
                "ek",
                "e.Kfr",
                "e.u",
                "e. u",
                "eu",
                "e.v",
                "e. v",
                "ev",
                "e.g",
                "e. g",
                "eg",
                "ewiv",
                "gag",
                "gbr",
                "germany",
                "gesbr",
                "ges. mb h",
                "gmbh",
                "ggmbh",
                "group",
                "gesbr",
                "gesmbh",
                "ges. mb h",
                "g-reit",
                "holding",
                "i.g",
                "inc",
                "invag",
                "keg",
                "kdör",
                "kg",
                "k.i.e.ö.r",
                "kgaa",
                "ohg",
                "mbh",
                "partg",
                "plc",
                "sce",
                "se",
                "ug",
                "vvag",
            ]

        self.stop_words = [
                "ab",
                "und",
                "e",
                "mit",
                "auch",
                "am",
                "ohne",
                "die",
                "zum",
                "b",
                "für",
                "im",
                "in",
                "gn",
                "der",
                "all genders",
                "als",
                "area",
                "basis",
                "befristet",
                "bei",
                "bewerben",
                "bonus",
                "branchenzuschlägen",
                "brutto",
                "bundesweit",
                "chance neu durchzustarten",
                "company confidential",
                "da",
                "daz",
                "dazuv",
                "dazuver",
                "dazuverdie",
                "dazuverdien",
                "dazuverdiene",
                "dazuverdienen",
                "dein",
                "direktvermittlung",
                "dringend",
                "dtm2FULL-TIME",
                "eur",
                "eur/h",
                "euro",
                "fahrgeld",
                "festanstellung",
                "für die region",
                "gehalt",
                "gesucht",
                "großer",
                "großraum",
                "hamburg",
                "im raum",
                "in 3 minuten erfolgreich",
                "in teilzeit",
                "inhouse",
                "jahresdurchschnitt",
                "jetzt",
                "job",
                "m w d",
                "mgl",
                "mind",
                "monat",
                "mwd",
                "neu",
                "neuer",
                "oder",
                "pro",
                "pro stunde",
                "pro",
                "prämie",
                "ref",
                "refnr",
                "remote",
                "schichten",
                "schichtzuschlägen",
                "sehr gutes",
                "signing",
                "sofort",
                "st",
                "standort",
                "standorte",
                "startprämie",
                "std",
                "std br beginnend",
                "stunde",
                "stunde",
                "stundenlohn",
                "tage",
                "team",
                "top",
                "unbefristet",
                "unbefristete",
                "up to",
                "urlaub",
                "verdienst",
                "von",
                "wechseln",
                "with growth opportunities",
            ]

        self.size = 100
        self.window = 25
        self.min_count = 1
        self.workers = 4
        self.epochs = 10

        # remove alphanumeric words like J123, 123, 123-123, 1,000 from text
        self.pattern0 = re.compile(r"(\b[a-zA-Z]?\d+(,|\.|-)?(?!:|th|[rn]d|st)(\w+)?)", re.IGNORECASE)
        # remove m/f | m/f/d | m/f/div patterns from text
        self.pattern1 = re.compile(r"\b(\w{1})([\/|\||\*])(\w{1})(([\/|\||\*]))?(\b\w{0,7})?(([\/|\||\*]))?(\b\w{1})?\b")
        # remove circular brackets from text
        self.pattern2 = re.compile(r"\&|\"|\,|\(|\)")
        # remove multiple continuous white spaces from text
        self.pattern3 = re.compile(r" +")
        # remove special characters other than alphabets from pre / suffix of the text
        self.pattern4 = re.compile(r"^\W+|\W+$|\s+\.\s+|\-\s+\-")
        # remove words at the enf of the text
        self.pattern5 = re.compile(
        r"((?<!/)(?<!/ )(?<!/-)(?<!/- )\bin\s*$|(-)\s*$|\b(ab\b|am|auf|bad|bei|bis|da|das|der|für die|für|/h|in\s+in\b|im|in der|mit|oder|und)\s*$)",
        re.IGNORECASE,
        )
        # remove specual characters from text
        self.pattern6 = re.compile(r"[€|$|£|,]")
        # remove company legal entities from text
        self.pattern7 = re.compile(r"\b%s\b" % r"\b|\b".join(map(re.escape, self.legal_entity)), re.IGNORECASE)
        # remove stopwords from text
        self.pattern8 = re.compile("|".join([r"\b" + word + r"\b" for word in self.stop_words]), re.IGNORECASE)
        self.pattern9 = re.compile(r"\s\/\s") 
        self.pattern10 = re.compile(r"\s-\s") 

        self.title_regex_preprocess = [self.pattern0, self.pattern1, self.pattern2,  
                                  self.pattern3, self.pattern4, self.pattern9, self.pattern10]
        self.title_regex_postprocess = [self.pattern5, self.pattern4, self.pattern5, self.pattern6, self.pattern3]
        self.title_remove_stopwords = [self.pattern8]
        self.company_legal_entity_regex = [self.pattern7, self.pattern2, self.pattern3, self.pattern4]

    def remove_patterns(self, text: str, regex_patterns: list) -> str:
        """
        This function identifies patterns and removes it from the text
        :param text:
        :param regex_patterns:
        :return: clean text
        """
        for pattern in regex_patterns:
            if re.findall(pattern, text):
                text = re.sub(pattern, " ", text)
        return text.strip()
        
    def clean_text(self, text: str) -> str:
        """
        preprocesing function to remove special chars, numbers & some unwanted text like (m/f/d) combinations from text
        The function retains "/" & "-" in the text and removes multiple white spaces. It also removes the stopwords
        and german company legal entities from text
        :param text:
        :return:
        """

        clean_txt = self.remove_patterns(text, self.title_regex_preprocess)
        clean_txt = self.remove_patterns(clean_txt, self.title_remove_stopwords)
        clean_txt = self.remove_patterns(clean_txt, self.company_legal_entity_regex)
        clean_txt = self.remove_patterns(clean_txt, self.title_regex_postprocess)
        return clean_txt.lower().split(" ")
    
    def pre_calculate_vector(self, words):
        words = words.split()
        vectors = [self.word_vectors_dict.get(word) for word in words if self.word_vectors_dict.get(word) is not None]
        return np.nanmean(vectors, axis=0) if vectors else np.zeros(self.size)
    
    def fit(self, X, y=None):
        
        X['target_w2v'] = y  
        X = X[(X['target_w2v'] == 1)]  
        X.drop(['target_w2v'], axis=1, inplace=True)
        X = X.fillna("")

        X['job_title'] = X['job_title'].astype(str).apply(self.clean_text)
        X['jobseeker_preference_job_title_b2c'] = X['jobseeker_preference_job_title_b2c'].apply(lambda row: [val.lower().split('/')[0] 
                                                                                                             for val in str(row).split() 
                                                                                                             if len(val) > 1])
        X['last_feed_search_query'] = X['last_feed_search_query'].apply(lambda row: [val.lower().split('/')[0] 
                                                                                     for val in str(row).split() 
                                                                                     if len(val) > 1])
        X['job_title'] = X['job_title'].apply(lambda row: [val.lower() for val in row if len(val) > 1])
        X['combined'] = X.apply(lambda row: list(set(i for sublist in [row['jobseeker_preference_job_title_b2c'], 
                                                                       row['last_feed_search_query'], 
                                                                       row['job_title']] for i in sublist if i)), axis=1)

        df = X.groupby('jobseeker_uid').agg({'combined': 'sum'}).reset_index()
        self.model = Word2Vec(sentences=df['combined'].tolist(), 
                              vector_size=self.size, 
                              window=self.window,
                              min_count=self.min_count, 
                              workers=self.workers,
                              epochs=self.epochs)
        
        unique_b2c_queries = X.apply(lambda row: ' '.join(row['jobseeker_preference_job_title_b2c'] + row['last_feed_search_query']), axis=1).unique()
        # Pre-calculate the vectors for unique values of jobseekers and all the words
        self.word_vectors_dict = {word: self.model.wv[word] for word in self.model.wv.index_to_key}
        self.precalculated_b2c_query_vectors = {value: self.pre_calculate_vector(value) for value in unique_b2c_queries}
        return self
    
    def process_row(self, row):
        return " ".join([val.lower().split('/')[0] for val in str(row).split() if len(val) > 1])
    
    def calculate_similarity(self, row):
        if np.isnan(np.sum(row['jobseeker_b2c_queries_vec'])) or np.isnan(np.sum(row['job_title_vec'])):
            return 0.0
        norm_product = np.linalg.norm(row['jobseeker_b2c_queries_vec']) * np.linalg.norm(row['job_title_vec'])
        return np.round(np.dot(row['jobseeker_b2c_queries_vec'], row['job_title_vec']) / norm_product, 4) if norm_product > 0 else 0.0

    def transform(self, X):
        X = X.fillna("")
        X['job_title'] = X['job_title'].str.lower()
        X['jobseeker_preference_job_title_b2c'] = X['jobseeker_preference_job_title_b2c'].apply(self.process_row)
        X['last_feed_search_query'] = X['last_feed_search_query'].apply(self.process_row)
        X['jobseeker_b2c_queries'] = X['jobseeker_preference_job_title_b2c'] + " " + X['last_feed_search_query']
        X['jobseeker_b2c_queries'] = X['jobseeker_b2c_queries'].str.strip()

        X['jobseeker_b2c_queries_vec'] = [self.precalculated_b2c_query_vectors.get(
                                            query,
                                            np.nanmean([
                                                self.word_vectors_dict.get(word)
                                                for word in query.split(" ")
                                                if self.word_vectors_dict.get(word) is not None
                                            ], axis=0) if query.split() else np.zeros(self.size))
                                        for query in X['jobseeker_b2c_queries']
                                    ]
        X['job_title_vec'] = [np.nanmean([
                                    self.word_vectors_dict.get(word)
                                    for word in query.split(" ")
                                    if self.word_vectors_dict.get(word) is not None
                                ], axis=0) if query.split() else np.zeros(self.size)
                            for query in X['job_title']
                           ]

        X['w2v_similarity_score'] =  X[['jobseeker_b2c_queries_vec','job_title_vec']].apply(self.calculate_similarity, axis=1)
        X['fuzziness_similarity_score'] = [round(fuzz.partial_ratio(query1, query2)/100, 4) for query1, query2 in zip(X['jobseeker_b2c_queries'], X['job_title'])]
        return X[['w2v_similarity_score', 'fuzziness_similarity_score']].to_numpy()

    def get_feature_names_out(self, input_features=None):
        # Generate and return feature names based on input_features or any other logic
        return ['w2v_similarity_score', 'fuzziness_similarity_score']

    
class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    """
    Wraps `MultiLabelBinarizer` in a form that can work with `ColumnTransformer`
    """

    def __init__(self):
        self.mlbs = {}

    def fit(self, X, y=None):
        for col in X.columns:
            mlb = MultiLabelBinarizer()
            mlb.fit(X[col])
            self.mlbs[col] = mlb
        return self

    def transform(self, X):
        transformed_cols = [self.mlbs[col].transform(X[col]) for col in X.columns]
        return np.concatenate(transformed_cols, axis=1)

    def get_feature_names_out(self, input_features=None):
        feature_names_out = []
        for col in self.mlbs:
            feature_names_out.extend([f'{col}_{class_}' for class_ in self.mlbs[col].classes_])
        return feature_names_out


class DistanceCalculator(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        # compute the Euclidean distance
        X['distance'] = X.apply(lambda row: self.euclidean_distance(row['jobseeker_preference_lng'], 
                                                                    row['jobseeker_preference_lat'], 
                                                                    row['job_lng'], 
                                                                    row['job_lat']), axis=1)
        
        # if any column is 0, set distance to a large number (e.g., the maximum distance observed)
        X.loc[(X['jobseeker_preference_lat'] == 0) | (X['jobseeker_preference_lng'] == 0) | (X['job_lat'] == 0) | (X['job_lng'] == 0), 'distance'] = X['distance'].max()
        
        # fit the MinMaxScaler to the distances
        self.scaler.fit(X[['distance']])
        
        return self

    def euclidean_distance(self, lon1, lat1, lon2, lat2):
        """
        Calculate the Euclidean distance between two points 
        on the earth (specified in decimal degrees)
        """
        return ((lon2 - lon1)**2 + (lat2 - lat1)**2)**0.5
    
    def transform(self, X):
        # compute the Euclidean distance
        X['distance'] = X.apply(lambda row: self.euclidean_distance(row['jobseeker_preference_lng'], 
                                                                    row['jobseeker_preference_lat'], 
                                                                    row['job_lng'], 
                                                                    row['job_lat']), axis=1)
        max_distance = X['distance'].max()
        # if any column is 0, set distance to a large number (e.g., the maximum distance observed)
        X.loc[(X['jobseeker_preference_lat'] == 0) | (X['jobseeker_preference_lng'] == 0) | (X['job_lat'] == 0) | (X['job_lng'] == 0), 'distance'] = max_distance
        
        # normalize between 0 and 1 using the MinMaxScaler fitted during the fit method
        X['distance_normalized'] = self.scaler.transform(X[['distance']])
        
        return X[['distance_normalized']].to_numpy()
    
    def get_feature_names_out(self, input_features=None):
        return ["distance"]

    
class KldbTransformer(BaseEstimator, TransformerMixin):
    """Preprocesses and extracts the first N digits from a Kldb code"""
    def fit(self, X, y=None):
        # store the input feature names
        self.input_features_ = X.columns
        return self

    def transform(self, X):
        X_transformed = pd.DataFrame()
        for col in X.columns:
            X_1_digit = X[col].str.slice(0, 1)  # first digit
            X_2_digits = X[col].str.slice(0, 2)  # first two digits
            X_transformed = pd.concat([X_transformed, 
                                       pd.DataFrame({f'{col}_{n}_digit': X_n_digits for n, X_n_digits in zip(range(2, 0, -1), [X_2_digits, X_1_digit])})], axis=1)
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        # use the stored feature names if input_features is not provided
        return [f'{col}_{n}_digit' for col in self.input_features_ for n in range(2, 0, -1)]
