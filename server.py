from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import ssl
from datetime import datetime, timedelta
import feedparser
import pandas as pd
from langchain_community.document_loaders import NewsURLLoader
import yfinance as yf
from langchain_openai import ChatOpenAI
import config


app = Flask(__name__)
CORS(app)


# A way to bypass SSL certificate verification errors
ssl._create_default_https_context = ssl._create_unverified_context


def fetch_and_process_data(name, period):
    # Remove any surrounding double quotes from the input strings
    symbol = name.strip('"')
    period = period.strip('"')

    # Create a Ticker object for the given stock symbol
    stock_name = yf.Ticker(symbol)

    # Define a list of columns to drop from the DataFrame
    to_drop = ["Dividends", "Stock Splits"]

    # Retrieve historical stock data for the specified period
    stock_dataframe = stock_name.history(period=period)

    # Check if 'Adj Close' column exists in the DataFrame
    if 'Adj Close' not in stock_dataframe.columns:
        # Drop columns listed in 'to_drop' list
        stock_dataframe.drop(to_drop, axis=1, inplace=True)

        # Reset index of the DataFrame and rename columns
        new_df = stock_dataframe.reset_index()
        new_df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

        # Convert data types of certain columns
        new_df['open'] = new_df['open'].astype(float)
        new_df['high'] = new_df['high'].astype(float)
        new_df['low'] = new_df['low'].astype(float)
        new_df['close'] = new_df['close'].astype(float)
        new_df['volume'] = new_df['volume'].astype(int)

        # Convert 'date' column to datetime and then to date
        new_df['date'] = pd.to_datetime(new_df['date'])
        new_df['date'] = new_df['date'].dt.date
        new_df["date"] = new_df["date"].astype(str)

        # Sort DataFrame by 'date' column in ascending order
        new_df = new_df.sort_values(by='date', ascending=True)

        # Create a list of dictionaries containing date and close price
        visualise_data = map(lambda v, c: dict(date=v, close=c),
                             new_df['date'], new_df["close"])

        # Convert the map object to a list and return it
        return list(visualise_data)
    else:
        # Return None if 'Adj Close' column exists in the DataFrame
        return None


def data_into_format(name, period):
    # Remove any leading or trailing double quotes from name and period
    name = name.strip('"')
    period = period.strip('"')

    # Create a Ticker object for the given stock name
    stock_name = yf.Ticker(name)

    # List of columns to drop from the dataframe if 'Adj Close' column is not present
    to_drop = ["Dividends", "Stock Splits"]

    # Retrieve historical data for the stock within the specified period
    stock_dataframe = stock_name.history(period=period)

    # Check if 'Adj Close' column is present in the dataframe
    if 'Adj Close' not in stock_dataframe.columns:
        # Drop unnecessary columns from the dataframe
        stock_dataframe.drop(to_drop, axis=1, inplace=True)

        # Reset index of the dataframe
        new_df = stock_dataframe.reset_index()

        # Rename columns for consistency
        new_df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

        # Convert data types of columns to appropriate types
        new_df['open'] = new_df['open'].astype(float)
        new_df['high'] = new_df['high'].astype(float)
        new_df['low'] = new_df['low'].astype(float)
        new_df['close'] = new_df['close'].astype(float)
        new_df['volume'] = new_df['volume'].astype(int)

        # Convert 'date' column to datetime and then to string for sorting
        new_df['date'] = pd.to_datetime(new_df['date'])
        new_df['date'] = new_df['date'].dt.date
        new_df["date"] = new_df["date"].astype(str)

        # Sort dataframe by date in ascending order
        new_df = new_df.sort_values(by='date', ascending=True)

        print(new_df)
        # Return the formatted dataframe
        return new_df
    else:
        # If 'Adj Close' column is present, return None
        return None


def predict_future_prices(data, model, predictors, days):

    # Create a copy of the data so that original data remains unchanged
    data_copy = data.copy()

    # Initialize a list to store predicted prices for each day
    predicted_prices = []

    # Loop over the number of days to predict
    for i in range(days):
        # Select all data except the most recent as the training data
        train = data_copy.iloc[:-1].copy()

        # Select the most recent data point as the test data
        test = data_copy.iloc[-1:].copy()

        # Predict using the provided model and predictors
        # Train the model to predict close price
        model.fit(train[predictors], train["close"])
        next_day_price = model.predict(test[predictors])

        # Append the predicted price for the next day to the list
        predicted_prices.append(next_day_price[0])

        # Add the predicted price to the dataset for the next iteration
        test["close"] = next_day_price[0]
        data_copy = pd.concat([data_copy, test])

    print(data_copy)

    return predicted_prices


def get_google_news_for_display(query):

    query = query.strip('"')
    query = "$" + query + "%20Stock"
    # Construct the RSS feed URL using the provided query
    rss_url = f'https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en'

    # Parse the RSS feed
    feed = feedparser.parse(rss_url)

    # Initialize an empty list to store URLs
    list_of_urls = []

    # Check if there are entries in the feed
    if feed.entries:
        # Iterate through the entries in the feed
        for entry in feed.entries:
            # Limit the number of URLs to 5
            if len(list_of_urls) >= 5:
                break
            else:
                # Extract the link from the entry and append it to the list
                link = entry.link
                list_of_urls.append(link)
    else:
        # If no entries are found in the feed, print a message
        print("Nothing Found!")

    # Return the list of URLs
    return list_of_urls


def get_google_news_for_sentiment(query):
    query = query.strip('"')
    query = "$" + query + "%20Stock"
    rss_url = f'https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en'
    feed = feedparser.parse(rss_url)
    list_of_urls = []
    if feed.entries:
        for entry in feed.entries:
            if len(list_of_urls) >= 10:
                break
            else:
                link = entry.link
                list_of_urls.append(link)
    else:
        print("Nothing Found!")

    return list_of_urls


def get_page_contents(list_of_urls):
    # Create an instance of the NewsURLLoader class with the provided list of URLs
    loader = NewsURLLoader(urls=list_of_urls)

    # Load data from the URLs using the NewsURLLoader instance
    data = loader.load()

    # Initialize an empty list to store page content of articles
    page_content_of_articles = []

    # Iterate through the loaded data
    for i in data:
        # Append the page content of each article to the list
        page_content_of_articles.append(i.page_content)

    # Return the list containing page content of articles
    return page_content_of_articles


# def preprocess_text(text):
#     # Initialize WordNetLemmatizer
#     wnl = WordNetLemmatizer()

#     # Tokenize the input text into individual words and convert them to lowercase
#     tokens = word_tokenize(text.lower())

#     # Get English stopwords
#     stop_words = set(stopwords.words('english'))

#     # Filter out stopwords from the tokens
#     filtered_tokens = [token for token in tokens if token not in stop_words]

#     # Lemmatize each token
#     lemmatized_tokens = [wnl.lemmatize(token) for token in filtered_tokens]

#     # Join the lemmatized tokens back into a single string
#     return ' '.join(lemmatized_tokens)


# def get_sentiment(text):
#     # BERT is natural language processing model
#     # Load the pre-trained FinBERT model for sentiment analysis
#     finbert = BertForSequenceClassification.from_pretrained(
#         "yiyanghkust/finbert-tone", num_labels=3)

#     # BERT has its own tokenizer corresponding to its pre-trained methods
#     # Load the tokenizer corresponding to the FinBERT model
#     tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")

#     # Create a pipeline for sentiment analysis using the loaded model and tokenizer
#     # Set max_length and truncation parameters for handling text longer than 512 tokens
#     nlp = pipeline("sentiment-analysis", model=finbert,
#                    tokenizer=tokenizer, max_length=512, truncation=True)

#     # Perform sentiment analysis on the input text using the pipeline
#     return nlp(text)


# def perform_sentiment_analysis(entries):
#     # Clean the text of each entry using the preprocess_text function
#     cleaned_text = [preprocess_text(entry) for entry in entries]

#     # Perform sentiment analysis on each cleaned text using the get_sentiment function
#     sentimental = [get_sentiment(text) for text in cleaned_text]

#     # Return the list of sentiment analysis results
#     return sentimental


# def prepare_data(sentimental, data):
#     # Initialize empty lists to store sentiment labels and scores
#     labels = []
#     scores = []

#     # Extract sentiment labels and scores from the sentimental list of dictionaries
#     for entry_sentiment in sentimental:
#         for sentiment in entry_sentiment:
#             labels.append(sentiment["label"])  # Extract sentiment label
#             scores.append(sentiment["score"])  # Extract sentiment score

#     # Create a DataFrame to store sentiment labels and scores
#     sentimental_df = pd.DataFrame({"sentiment": labels, "score": scores})

#     # Reverse the order of rows in the data DataFrame and reset index
#     data_index = data.index
#     data = data.iloc[::-1]
#     data.index = data_index

#     # Concatenate the data DataFrame and the sentimental DataFrame along columns
#     new_df = pd.concat([data, sentimental_df], axis=1)

#     # Remove rows with NaN values
#     new_df = new_df.dropna()

#     # Define categorical variables to convert to numerical columns
#     categorical_variables = ["sentiment"]

#     # Convert categorical columns to numerical columns
#     new_df = pd.get_dummies(new_df, columns=categorical_variables)

#     # Return the new DataFrame
#     return new_df


def llm_analysis(name):
    # Initialize the ChatOpenAI instance with the API key
    llm = ChatOpenAI(
        api_key=config.api_key)

    # Get news links related to the given stock name
    news_links = get_google_news_for_sentiment(name)

    # Load data from the news URLs
    loader = NewsURLLoader(urls=news_links)
    data = loader.load()

    # Extract page content of articles from the loaded data
    page_content_of_articles = []
    for i in data:
        page_content_of_articles.append(i.page_content)

    # Perform sentiment analysis on the page content of articles
    # sentimental = perform_sentiment_analysis(page_content_of_articles)

    # Perform sentiment analysis on the page content of articles
    response = llm.invoke(
        f"As a seasoned trader, your task involves leveraging real-world insights to predict future trends in financial markets,"
        f"specifically focusing on the movement of {name} stocks. Utilizing a comprehensive set of articles related to the stock market,"
        f"as detailed in {page_content_of_articles}, you are tasked with forecasting the direction of {name} stocks' movement."
        f"Your analysis should not only highlight key points from the news articles but also synthesize the overall"
        f"sentiment scores derived from these analyses. Aim to provide a concise yet comprehensive conclusion that"
        f"integrates both the content of the articles and the aggregated sentiment scores. Ensure your response is specific,"
        f"avoiding general statements, and structured in a single paragraph. Emphasize the importance of mentioning significant"
        f"details from the news articles and briefly summarizing the sentiment scores to support your prediction."
    )

    # return response
    return response.content


def get_wr(high, low, close, lookback_period):
    # Calculate the highest high within the lookback period
    highestHigh = high.rolling(14).max()

    # Calculate the lowest low within the lookback period
    lowestLow = low.rolling(lookback_period).min()

    # Calculate the Williams %R indicator
    wr = -100 * ((highestHigh - close) / (highestHigh - lowestLow))

    return wr


def get_signals(data):
    # Calculate the Williams %R indicator with a lookback period of 14 days
    data["wr_14"] = get_wr(data["high"], data["low"], data["close"], 14)

    # Identify buy signals where Williams %R is less than or equal to -80
    data["Buy"] = data[data["wr_14"] <= -80]["close"]

    # Identify sell signals where Williams %R is greater than or equal to -20
    data["Sell"] = data[data["wr_14"] >= -20]["close"]

    # new dataframe with only rows that have buy signals
    for_buy = data.dropna(subset=['Buy'])

    # new dataframe with only rows that have sell signals
    for_sell = data.dropna(subset=['Sell'])

    # Map buy signal dates and closing prices to dictionaries
    buy_signals_1 = map(lambda v, c: dict(date=v, close=c),
                        for_buy['date'], for_buy["Buy"])

    # Map sell signal dates and closing prices to dictionaries
    sell_signals_1 = map(lambda v, c: dict(date=v, close=c),
                         for_sell['date'], for_sell["Sell"])

    # Convert map objects to lists and return
    return list(buy_signals_1), list(sell_signals_1)


def generate_visualise_data(dataframe, first_prediction, days):
    # Extract the last 50 closing prices from the dataframe
    last_fifty = list(dataframe["close"].tail(50))

    # Combine the last 50 closing prices with the first prediction
    final_prediction = last_fifty + first_prediction

    # Get today's date without time
    dated = datetime.today().date()

    # Initialize an empty list to store dates
    list_of_dates = []

    # Generate dates for the specified number of days
    while len(list_of_dates) < days:
        dated += timedelta(days=1)
        # Check if the current date is a weekday (0 to 4 represent Monday to Friday)
        if dated.weekday() < 5:
            # Convert the date to a Timestamp object and append it to the list
            list_of_dates.append(pd.Timestamp(dated))

    # Extract the last 50 dates from the dataframe
    last_fifty_dates = list(pd.to_datetime(dataframe["date"].tail(50)))

    # Combine the last 50 dates with the newly generated dates
    new_list_dates = last_fifty_dates + list_of_dates

    # Format dates as strings in 'YYYY-MM-DD' format
    formatted_dates = [date.strftime('%Y-%m-%d') for date in new_list_dates]

    # Map dates and corresponding closing prices to dictionaries
    visualise_data = map(lambda v, c: dict(date=v, close=c),
                         formatted_dates, final_prediction)

    # Convert the map object to a list and return it
    return list(visualise_data)


# The same as function above but for the combined prediciton instead
# def generate_combined_visualise_data(dataframe, first_prediction, second_prediction, days):
#     combined_predictions = [(first + second) / 2 for first,
#                             second in zip(first_prediction, second_prediction)]
#     last_fifty = list(dataframe["close"].tail(50))

#     final_prediction = last_fifty + combined_predictions
#     dated = datetime.today().date()  # Get today's date without time
#     list_of_dates = []
#     while len(list_of_dates) < days:
#         dated += timedelta(days=1)
#         # Check if the current date is a weekday (0 to 4 represent Monday to Friday)
#         if dated.weekday() < 5:
#             # Convert to Timestamp object
#             list_of_dates.append(pd.Timestamp(dated))

#     last_fifty_dates = list(pd.to_datetime(dataframe["date"].tail(50)))
#     new_list_dates = last_fifty_dates + list_of_dates
#     formatted_dates = [date.strftime('%Y-%m-%d') for date in new_list_dates]
#     visualise_data = map(lambda v, c: dict(date=v, close=c),
#                          formatted_dates, final_prediction)

#     return list(visualise_data)


def run_simulation(name):
    # Making erros universal between both pages as some stocks cannot be run when trying to visualise_historical_date()
    catching_errors = fetch_and_process_data(name, "6mo")

    if catching_errors is None:
        return 400

    # Fetch historical data for the specified stock for the last 10 years
    dataframe = data_into_format(name, "10y")

    # If stock does not exist
    if dataframe is None:
        # Return status code 400
        return 400
    else:
        # Set the number of days for prediction
        days = 30

        # Initialize a RandomForestRegressor model with specified parameters
        model = RandomForestRegressor(
            n_estimators=200, min_samples_split=25, random_state=1)

        # Define predictors for the model
        predictors = ["close", "volume", "open", "high", "low"]

        # Predict future prices using the model and predictors
        first_prediction = predict_future_prices(
            dataframe, model, predictors, days)

        return first_prediction

        # # Get news feeds related to the specified stock for sentiment analysis
        # news_feeds = get_google_news_for_sentiment(name)

        # # Check if there are any news feeds available
        # if len(news_feeds) == 0:
        #     # If no news feeds are available, return prediction with purely historical data
        #     return generate_visualise_data(dataframe, first_prediction, days)
        # else:
        #     # Get page contents of the news feeds
        #     page_contents = get_page_contents(news_feeds)

        #     # Perform sentiment analysis on the page contents
        #     sentimental = perform_sentiment_analysis(page_contents)

        #     # Prepare data by combining sentiment analysis with historical data
        #     dataframe_with_sentiment = prepare_data(sentimental, dataframe)

        #     # Extract sentiment-related columns
        #     sentiment_columns = [
        #         i for i in dataframe_with_sentiment.columns if dataframe_with_sentiment[i].dtype == 'bool']

        #     # Define predictors including sentiment-related columns and sentiment scores
        #     other_predictors = predictors + sentiment_columns + ["score"]

        #     # Check if there is enough data for analysis
        #     if len(dataframe_with_sentiment) <= 3:
        #         # If there is not enough data, return prediction with purely historical data
        #         return generate_visualise_data(dataframe, first_prediction, days)
        #     else:
        #         # Predict future prices using the model and updated predictors
        #         second_prediction = predict_future_prices(
        #             dataframe_with_sentiment, model, other_predictors, days)

        #         # return combined prediciton
        #         return generate_combined_visualise_data(dataframe, first_prediction, second_prediction, days)


@app.route("/api/visualise", methods=["POST"])
def visualise_historical_data():
    try:
        # Retrieve JSON data from the request
        data = request.json
        name = data.get("name")
        print("Received JSON data:", name)
        # Fetch and process historical data for the last 6 months
        visualise_6month_data = fetch_and_process_data(name, "6mo")

        # Check if data for the last 6 months is sparse
        if visualise_6month_data != None:
            if len(visualise_6month_data) <= 45:
                # If data is sparse, return an error response
                return "Data sparse", 401

        if visualise_6month_data == None:
            return "Stock Not Found", 400

        # Fetch and process historical data for the last 1 year
        visualise_1year_data = fetch_and_process_data(name, "1y")

        # Fetch and process historical data for the last 5 years
        visualise_5year_data = fetch_and_process_data(name, "5y")

        # Fetch and process all available historical data
        visualise_max_data = fetch_and_process_data(name, "max")

        # Fetch and process historical data for the entire lifetime
        lifetime_data = fetch_and_process_data(name, "max")
        # Get buy and sell signals for the last 6 months
        buy_signals, sell_signals = get_signals(
            data_into_format(name, "6mo"))

        # Check the length of historical data for the last 5 years
        if len(visualise_5year_data) < 120:
            # If there are not enough data points to make a difference between the different views then return data returned for the visulisation is different
            return jsonify({
                "six_month_data": visualise_6month_data,
                "one_year_data": visualise_1year_data,
                "five_year_data": visualise_1year_data,
                "max_data": visualise_1year_data,
                "data_for_lifetime": lifetime_data,
                "buy_signals": buy_signals,
                "sell_signals": sell_signals
            })
        else:
            # Else return the data normally with the corresponding data
            return jsonify({
                "six_month_data": visualise_6month_data,
                "one_year_data": visualise_1year_data,
                "five_year_data": visualise_5year_data,
                "max_data": visualise_max_data,
                "data_for_lifetime": lifetime_data,
                "buy_signals": buy_signals,
                "sell_signals": sell_signals
            })
    except Exception as e:
        # Handle any other exceptions and return an error response
        print("Error processing JSON data:", str(e))
        return "Error: Please Try Again", 402


@app.route("/api/simulate", methods=["POST"])
def simulation():
    try:
        data = request.json
        name = data.get("name")
        print("Received JSON data:", name)
        visualise_6month_data = fetch_and_process_data(name, "6mo")
        # Return error if data is sparse
        if visualise_6month_data != None:
            if len(visualise_6month_data) <= 45:
                return "Data sparse", 401
        simulated_prices = run_simulation(name)
        # Return error if stock name does not exist
        if simulated_prices == 400:
            return "Stock not found", 400
        else:
            analysis_of_text = llm_analysis(name)
            news_articles = get_google_news_for_display(name)
            # Last data point which would be predicted price in the last 30 days
            last_price = simulated_prices[-1]["close"]
            price_30_days_ago = visualise_6month_data[-1]["close"]
            change_in_price = last_price - price_30_days_ago
            return jsonify({"simulated_prices": simulated_prices,
                            "stock_analysis": analysis_of_text,
                            "news_articles": news_articles,
                            "last_price": last_price,
                            "change_in_price": change_in_price})
    except Exception as e:
        print("Error processing JSON data:", str(e))
        return "Error: Please Try Again", 500


if __name__ == "__main__":
    app.run(debug=True, port=8080)
