"""
pandas: an open source data analysis and manipulation tool, built on top of the Python programming language.
Github: https://github.com/pandas-dev/pandas
Main page: https://pandas.pydata.org/
"""
import pandas as pd
"""
numpy: The fundamental package for scientific computing with Python.
Github: https://github.com/numpy/numpy
Main page: https://numpy.org/
"""
import numpy as np
"""
urllib.parse: built in python module for parsing urls
docs: https://docs.python.org/3/library/urllib.parse.html#module-urllib.parse
"""
from urllib.parse import urlparse
"""
tldextract: Accurately separates a URL’s subdomain, domain, and public suffix, using the Public Suffix List (PSL).
Author: John Kurkowski
Github: https://github.com/john-kurkowski/tldextract
Main page: https://pypi.org/project/tldextract/
"""
import tldextract
"""
tensorflow: An Open Source Machine Learning Framework
Github: https://github.com/tensorflow/tensorflow
Main page: https://tensorflow.org
"""
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Conv1D, GlobalMaxPooling1D, Concatenate, BatchNormalization, Dropout, TextVectorization
"""
sklearn: machine learning package for python
Github: https://github.com/scikit-learn/scikit-learn
Main page: https://scikit-learn.org/stable/
"""
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
"""
matplotlib: library for creating static, animated, and interactive visualizations in Python
Github: https://github.com/matplotlib/matplotlib
Main page: https://matplotlib.org/
"""
import matplotlib.pyplot as plt
"""
seaborn: Statistical data visualization using matplotlib
Github: https://github.com/seaborn/seaborn
Main page: https://seaborn.pydata.org/
"""
import seaborn as sns


def split_url(url):
    """
    Split the url into its parts
    :param url: url to split
    :return: dictionary of url parts
    """
    # call urlparse to split url
    parsed = urlparse(url)
    # call tldextract to get top level domain
    extracted = tldextract.extract(url)

    # read output from parsers
    scheme = parsed.scheme if parsed.scheme != '' and 'http' in parsed.scheme else ''
    subdomain = extracted.subdomain if extracted.subdomain else ''
    second_level_domain = extracted.domain
    top_level_domain = extracted.suffix if extracted.suffix else ''
    port = str(parsed.port) if parsed.port is not None else ''
    path = parsed.path if parsed.path else ''
    query = parsed.query if parsed.query else ''
    params = parsed.params if parsed.params else ''
    fragment = parsed.fragment if parsed.fragment else ''

    # get subdirectory, path, and top level domain if the parser did not do it correctly
    subdirectory = ''
    if path != '':
        subdirectory = "/".join(path.strip("/").split("/")[:-1]) if "/" in path.strip("/") else ''
        if subdirectory != '' and top_level_domain != '':
            subdirectory = subdirectory.split(top_level_domain, 1)[-1]

        if subdirectory != '':
            path = path.split(subdirectory)[-1]
        elif top_level_domain != '':
            path = path.split(top_level_domain, 1)[-1]

    # get correct port number if present
    if top_level_domain != '':
        tld = top_level_domain + ":"
        if tld in url:
            port_number = url.split(tld)[1].split('/')[0]
            try:
                port_number = int(port_number)
                port = str(port_number)
                if subdirectory != '' and port in subdirectory:
                    subdirectory = subdirectory.split(port)[1]
            except ValueError:
                pass

    return {
        "scheme": scheme,
        "subdomain": subdomain,
        "second_level_domain": second_level_domain,
        "top_level_domain": top_level_domain,
        "subdirectory": subdirectory,
        "port": port,
        "path": path,
        "query": query,
        "parameters": params,
        "fragment": fragment
    }


def extract_features(text_series):
    """
    Calculates numerical features for urls
    :param text_series: urls to analyze
    :return: url data
    """
    # for each url in series calc numerical data
    return np.array([
        [len(text), sum(c.isdigit() for c in text), sum(not c.isalnum() for c in text)]
        for text in text_series
    ])


def create_text_vectorizer(max_tokens=500, output_seq_length=20):
    """
    Create a text vectorizer
    :param max_tokens: number of words in vocab
    :param output_seq_length: size of output space
    :return: text vectorizer
    """
    return TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=output_seq_length,
        split="character"
    )


def prep_inputs(df):
    """
    Create vectorizers for all text inputs and combine all numerical inputs
    :param df: url parts dataframe
    :return: prepared features
    """
    # One-hot encode scheme
    onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    scheme_encoded = onehot_encoder.fit_transform(df[["scheme"]])

    # Extract numerical features from text columns

    text_features = ["subdomain", "second_level_domain", "top_level_domain", "subdirectory", "path", "query", "parameters", "fragment"]
    text_numerical_features = np.hstack([extract_features(df[feature]) for feature in text_features])

    # Handle port - fill empty ports with 0
    port_feature = df["port"].replace("", 0).astype(float).values.reshape(-1, 1)

    # Combine text features and port feature
    text_numerical_features = np.hstack([text_numerical_features, port_feature])

    # Normalize numerical features
    scaler = StandardScaler()
    text_numerical_features = scaler.fit_transform(text_numerical_features)

    # Combine numerical features
    x_combined = np.hstack([scheme_encoded, text_numerical_features])

    # Character tokenization for embeddings

    # Create vectorizers
    text_vectorizers = {
        "subdomain": create_text_vectorizer(),
        "second_level_domain": create_text_vectorizer(),
        "top_level_domain": create_text_vectorizer(),
        "subdirectory": create_text_vectorizer(),
        "path": create_text_vectorizer(),
        "query": create_text_vectorizer(),
        "parameters": create_text_vectorizer(),
        "fragment": create_text_vectorizer(),
    }

    # Adapt vectorizers on data
    for key in text_vectorizers:
        text_vectorizers[key].adapt(df[key].values)

    # Convert text fields to sequences
    sequence_data = {key: np.array(text_vectorizers[key](df[key].values)) for key in text_vectorizers}

    # Encode target labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["type"])

    return x_combined, sequence_data, y


def create_model(x_combined, sequence_data):
    """
    create cnn model
    :param x_combined: numerical features
    :param sequence_data: vectorized text features
    :return: cnn model
    """
    inputs = {}
    inputs_list = []
    embeddings = []

    # Numerical input for scheme + port
    inputs['scheme_numerical_input'] = Input(shape=(x_combined.shape[1],), name='scheme_numerical_input')
    inputs_list.append(inputs['scheme_numerical_input'])
    x = Dense(32, activation='relu')(inputs['scheme_numerical_input'])

    embeddings.append(x)

    # Text sequence inputs
    for name in sequence_data:
        inputs[name] = Input(shape=(sequence_data[name].shape[1],), name=name)
        inputs_list.append(inputs[name])
        emb = Embedding(input_dim=500, output_dim=16)(inputs[name])
        emb = Conv1D(32, 3, activation="relu")(emb)
        emb = GlobalMaxPooling1D()(emb)
        embeddings.append(emb)

    # merge previous layers and put through dense layers
    merged = Concatenate()(embeddings)
    merged = Dense(128, activation="relu")(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.3)(merged)
    merged = Dense(64, activation="relu")(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.3)(merged)
    output = Dense(4, activation="softmax")(merged)

    model = Model(inputs=inputs_list, outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # model.summary()
    return model


def train_and_test(x_combined, sequence_data, y):
    """
    Perform k-fold cross validation on model and data
    :param x_combined: numerical features
    :param sequence_data: vectorized text features
    :param y: target labels
    :return: None
    """
    # Manual 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    all_reports = []

    # for each fold, get data and create new model, train and evaluate
    for train_idx, val_idx in kf.split(x_combined):
        print(f"\nFold {fold}:")

        # get data
        x_train = [val[train_idx] for key, val in sequence_data.items()]
        x_train.insert(0, x_combined[train_idx])
        y_train = y[train_idx]

        x_val = [val[val_idx] for key, val in sequence_data.items()]
        x_val.insert(0, x_combined[val_idx])
        y_val = y[val_idx]

        # create cnn and train
        model = create_model(x_combined, sequence_data)
        model.fit(x_train, y_train, epochs=10, batch_size=32, verbose='auto')

        # test model
        y_pred_probs = model.predict(x_val)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # get evaluation
        report = classification_report(y_val, y_pred, output_dict=True)
        all_reports.append(report)

        print(classification_report(y_val, y_pred))

        cm = confusion_matrix(y_val, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Fold {fold} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

        fold += 1

    # average scores across each fold
    avg_report = average_reports(all_reports)
    print("\nAverage Classification Report:")
    for label, metrics in avg_report.items():
        print(f"{label}: {metrics}")


def average_reports(reports):
    """
    Create an average report
    :param reports: scores
    :return: averaged scores
    """
    avg = {}
    keys = reports[0].keys()
    for key in keys:
        if isinstance(reports[0][key], dict):
            avg[key] = {}
            for subkey in reports[0][key]:
                avg[key][subkey] = np.mean([r[key][subkey] for r in reports])
        else:
            avg[key] = np.mean([r[key] for r in reports])
    return avg


def main():
    """
    Main driver for program
    :return: None
    """
    # get data
    text_df = pd.read_csv('malicious_phish.csv')
    split_urls = []
    # parse urls
    for index, row in text_df.iterrows():
        url_parts = split_url(row['url'])
        url_parts['type'] = row['type']
        split_urls.append(url_parts)

    url_df = pd.DataFrame(split_urls)
    x_combined, sequence_data, y = prep_inputs(url_df)
    # run model
    train_and_test(x_combined, sequence_data, y)


if __name__ == "__main__":
    main()
