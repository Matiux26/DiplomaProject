from datetime import datetime
import pyodbc
import sys
import time
from newsapi import NewsApiClient
from datetime import date

class Articles:

    def __init__(self):
        self.db = DatabaseConnection()
        #W tym miejscu trzeba podać klucz wygenerowany z News API
        self.apiKey = ''
        #Podłączenie się do News Api za pomocą klucza
        self.newsapi = NewsApiClient(api_key=self.apiKey)

    def run(self):
        self.getAllArticlesHeadlines()

    #Metoda pobierająca nagłówki wiadomości za dany dzień i zapisująca je w bazie
    def getAllArticlesHeadlines(self):

        currentDate = date.today()
        newsHeadlines = ''
        topArticles = self.newsapi.get_top_headlines(category='business',
                                          language='en',
                                          country='us')
        for article in topArticles['articles']:
            newsHeadlines = newsHeadlines + str(article['title'])

        newsHeadlines = newsHeadlines.replace("'", "")
        self.db.saveArticles(newsHeadlines, currentDate)

class DatabaseConnection:

    def __init__(self):
        self.conn = pyodbc.connect('Driver={SQL Server};'
                                   'Server=.\SQLEXPRESS;'
                                   'Database=Stonks;'
                                   'Trusted_connection=yes;')

    #Metoda do zapisu danych w bazie
    def saveArticles(self, newsHeadlines, currentDate):
        query = "INSERT INTO news_headlines(date, headlines) VALUES('" + str(currentDate) + "','" + str(newsHeadlines) + "')"
        cursor = self.conn.cursor()
        cursor.execute(query)
        cursor.commit()
        cursor.close()

    def closeConn(self):
        self.conn.close()

if __name__ == '__main__':
    Articles().run()