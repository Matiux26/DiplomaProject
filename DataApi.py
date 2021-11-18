import yfinance as yf
import pyodbc
import time
import numpy as np
import sys
import json

class DataApi:

    def __init__(self):
        #Definicje symboli spółek których dane mają zostać pobrane
        self.tickers = ["AAPL","EA","ATVI","NTES","TTWO","CZR","ZNGA","MOMO","CHDN","YY","IGT","GLUU","CYOU","CMCM","SOHU", "AMZN", "GOOGL", "FB", "GM", "NFLX", "^IXIC"]
        #Utworzenie połączenia z bazą danych
        self.conn = pyodbc.connect('Driver={SQL Server};'
                            'Server=.\SQLEXPRESS;'
                            'Database=Stonks;'
                            'Trusted_connection=yes;')
    
    #Metoda do początkowego zasilenia bazy danych
    def initialSearchThroughApi(self):
        #Iterowane są symbole spółki
        for ticker in self.tickers:
            #Biblioteka yfinance zwraca dane konretnej spółki
            #na podstawie symbolu spółki
            msft = yf.Ticker(ticker)

            response = msft.info
            self.cursor = self.conn.cursor()

            #Sprawdzenie czy dane spółki znajdują się już
            #w bazie jeżeli nie to dane są zapisywane
            record_id = self.companyExistInDb(ticker)
            if(record_id == None):
                #Zapis informacji o spółce
                record_id = self.saveCompanyInfoToDb(response)
                #Zapis danych giełdowych spółki za całościowy okres
                #jej istnienia na giełdzie
                actions = msft.history(period="max")
                self.saveCompanyMarketDataToDb(record_id, actions)
            #Ograniczenie prędkości pętli z powodu ograniczeń w API
            time.sleep(10)

    #Metoda do codziennego pobierania danych giełdowych
    def dailySearchThroughApi(self):
        for ticker in self.tickers:
            self.cursor = self.conn.cursor()

            msft = yf.Ticker(ticker)
            #Jeśli dana spółka nie znajduje się jeszcze w bazie
            #to tworzony jest wpis z informacjami tej spółki
            record_id = self.companyExistInDb(ticker)

            #Jeżeli dana spółka nie występuje w bazie to pobierane są
            #dane giełdowe za cały okres jej istnienia na giełdzie
            #w przeciwnym wypadku pobierane są dane za okres jednego dnia
            if(record_id == None):
                response = msft.info
                record_id = self.saveCompanyInfoToDb(response)
                actions = msft.history(period="max")
                self.saveCompanyMarketDataToDb(record_id, actions)
            else:
                actions = msft.history(period="1d")
                self.saveCompanyMarketDataToDb(record_id[0], actions)

            time.sleep(10)

    #Metoda sprawdzająca czy dana spółka znajduje się już w bazie danych
    def companyExistInDb(self, ticker):
        record_id = self.cursor.execute("SELECT id FROM company_info \
                        WHERE symbol = ? ", ticker).fetchone()

        return record_id

    #Metoda do zapisania informacji o podanej spółce w bazie danych
    def saveCompanyInfoToDb(self, response):
        print(response)
        self.cursor.execute("INSERT INTO company_info \
                        (sector,long_business_summary,full_time_employees,country,website,industry,short_name,symbol)  \
                        VALUES (?,?,?,?,?,?,?,?)", 
                        (response.get("sector"),response.get("longBusinessSummary"),response.get("fullTimeEmployees"),response.get("country"),
                        response.get("website"),response.get("industry"),response.get("shortName"),response.get("symbol")))
        record_id = self.cursor.execute('SELECT @@IDENTITY AS id;').fetchone()[0]
        self.cursor.commit()
        return record_id
    
    #Metoda do zapisu danych giełdowych spółki
    def saveCompanyMarketDataToDb(self, record_id, actions):
        for date, row in actions.iterrows():
            if(self.isAnyNaN(row) == False):
                self.cursor.execute("INSERT INTO market_data \
                            (date,[open],high,low,[close],volume,company_info_id)  \
                            VALUES (?,?,?,?,?,?,?)",
                            (date,row["Open"],row["High"],row["Low"],
                            row["Close"],row["Volume"],record_id))
                self.cursor.commit()

    #Metoda pomocnicza do sprawdzenia czy wartość jest typu NAN
    def isAnyNaN(self, row):
        for value in row:
            if np.isnan(value) == True:
                return True
        return False

    def closeDbConn(self):
        self.conn.close()

if __name__ == '__main__':
    param = sys.argv[1]
    if param == "help":
        print("List of parameters: \n \
        help - shows list of parameters \n \
        init - initial feed of db \n \
        daily - daily feed of db")
        exit()
    elif param == "init":
        DataApi().initialSearchThroughApi()
        DataApi().closeDbConn()
    elif param == "daily":
        DataApi().dailySearchThroughApi()
        DataApi().closeDbConn()
    
    