# Tensorflow
import tensorflow as tf
from tensorflow import keras

# Helper liblaries
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Binarizer
import pyodbc
from datetime import datetime as dt
import numpy as np
import pandas as pd
import os
import sys
import random
import math
import datetime
import pickle
from dateutil.parser import parse
import shap

class Machine_learning:

    #Propercje klasy Machine_learning.
    #NUMBER_OF_PAST_STEPS określa liczbę dni na podstawie
    #których model dokonuje prognozy.
    NUMBER_OF_PAST_STEPS = 30
    #NUMBER_OF_FUTURE_STEPS określa ilość dni dla których
    #wykonywana jest prognoza.
    NUMBER_OF_FUTURE_STEPS = 14
    #NUMBER_OF_FEATURES określa liczbę atrybutów wykorzystywanych
    #przez model podczas prognozowania.
    NUMBER_OF_FEATURES = 7
    #VOCAB_SIZE Określa ograniczenie ilości słów pochodzących
    #z nagłówków wiadomości. Jest to maksymalna wielkość słownika.
    VOCAB_SIZE = 12000

    #Konstruktor klasy Machine_learning.
    #Przekazywane są do niego dane giełdowe wraz
    #z nagłówkami wiadomości.
    #Opcjonalnie przekazywane są prawdziwe dane przyszłe
    #do wykonania porównania podczas wykonywania prognozy.
    def __init__(self, data, trueData=None):
        self.trueData = trueData
        #Proces tokenizacji i binarizacji nagłówków
        self.tokenizedText = self.tokenizeText(data['headlines'])
        #Załadowanie modelu jeżeli istnieje
        #w przeciwnym wypadku tworzony jest nowy model.
        self.lstmModel = self.loadModelIfExists()
        #Oddzielenie nagłówków wiadomości od ramki danych.
        data.pop('headlines')
        #Przeformatowanie ramki danych do tablicy
        #typu numpy
        self.dataset = data.values
    
    #Metoda zwraca model jeżeli istnieje.
    #W przeciwnym wypadku tworzona jest nowa instancja modelu
    def loadModelIfExists(self):
        if(os.path.isfile('MLMODEL.h5')):
            lstmModel = tf.keras.models.load_model('MLMODEL.h5')
        else:
            lstmModel = self.createModel([self.NUMBER_OF_PAST_STEPS, self.NUMBER_OF_FEATURES])

        return lstmModel

    #Metoda służąca do tokenizacji i binarizacji
    #nagłówków wiadomości.
    #Jako parametr przyjmuje tablicę z nagłówkami.
    def tokenizeText(self, text):
        tokenizerFileName = 'tokenizer.pickle'
        binarizerFileName = 'binarizer.pickle'

        #Obiekt klasy TfidfVectorizer jest
        #ładowany z pliku jeżeli istnieje
        if(os.path.isfile(tokenizerFileName)):
            with open(tokenizerFileName, 'rb') as handle:
                tokenizer = pickle.load(handle)
                #Załadowany tokenizer użyty jest do przekształcenia
                #podanego tekstu w tokeny.
                text = tokenizer.transform(text)
        #W przeciwnym wypadku obiekt klasy TfidfVectorizer
        #jest tworzony z ograniczeniem słownika o wielkości 
        #równej VOCAB_SIZE.
        else:
            tokenizer = TfidfVectorizer(max_features=self.VOCAB_SIZE)
            #Tokenizer jest uczony i jednocześnie użyty do 
            #przekształcenia nagłówków wiadomości w tokeny.
            text = tokenizer.fit_transform(text.values)
            #Tokenizer jest zapisywany do pliku.
            self.saveTokenizer(tokenizer)

        #Obiekt klasy Binarizer jest
        #ładowany z pliku jeżeli istnieje
        if(os.path.isfile(binarizerFileName)):
            with open(binarizerFileName, 'rb') as handle:
                binarizer = pickle.load(handle)
                #Załadowany binarizer użyty jest do przekształcenia
                #podanych tokenów w kod 1 z N.
                result = binarizer.transform(text)
        else:
            #Tworzony jest nowy obiekt klasy Binarizer.
            #Binarizer jest uczony i jednocześnie użyty do 
            #przekształcenia nagłówków wiadomości w tokeny.
            #Na końcu obiekt Binarizer jest zapisywany do pliku.
            binarizer = Binarizer()
            result = binarizer.fit_transform(text)
            self.saveBinarizer(binarizer)


        return result.toarray()

    #Metoda służąca do zapisania Tokenizera do pliku.
    def saveTokenizer(self, value):
        fileName = 'tokenizer.pickle'

        if(os.path.isfile(fileName)):
            return
        else:
            with open(fileName, 'wb') as handle:
                pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #Metoda służąca do zapisania Binarizera do pliku.
    def saveBinarizer(self, value):
        fileName = 'binarizer.pickle'

        if(os.path.isfile(fileName)):
            return
        else:
            with open(fileName, 'wb') as handle:
                pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #Metoda implementująca mechanizm walidacji krzyżowej.
    #W parametrze dataset przekazywane są dane giełdowe.
    def crossValidate(self, dataset):
        #Zmienna o nazwie NUMBER_OF_SEGMENTS
        #Określa liczbę segmentów na które podzielone
        #zostaną dane.
        NUMBER_OF_SEGMENTS = 5
        #Deklaracja pustych tablic do przechowywania
        #podzielonych danych
        segments = []
        news = []

        #Długość jednego segmentu liczona jest na podstawie liczby
        #wszystkich wierszy danych przez ilość segmentów.
        #Np. 4000 / 5 = 800. Czyli każdy segment ma długość 800.
        lengthOfSegment = math.floor(len(dataset) / NUMBER_OF_SEGMENTS)

        #Doklejanie kolejnych segmentów z danymi do
        #tablicy o nazwie segments.
        #Iteracja jest wykonywana od 0 do liczby segmentów.
        #W tym przypadku iteracja wykonywana jest 5 razy.
        for i in range(0, NUMBER_OF_SEGMENTS):
            #Indeks początkowy to długość segmentu razy numer iteracji.
            indexFrom = lengthOfSegment * i
            #Indeks końcowy to numer segmentu razy numer iteracji + 1
            indexTo = lengthOfSegment * (i + 1)
            #Do tablicy o nazwie segments dodawany jest wycinek z
            #tablicy dataset o indeksach od indexFrom do indexTo.
            #Są to dane giełdowe.
            segments.append(dataset[indexFrom:indexTo])
            #Do tablicy o nazwie news dodawany jest wycinek z
            #tablicy dataset o indeksach od indexFrom do indexTo.
            #Są to nagłówki wiadomości.
            news.append(self.tokenizedText[indexFrom:indexTo])
        
        #Wykonanie niezbędnych przekształceń dla każdego segmentu.
        #Iteracja jest wykonywana od 0 do liczby segmentów.
        #W tym przypadku iteracja wykonywana jest 5 razy.
        for i in range(0, NUMBER_OF_SEGMENTS):
            marketData = segments.copy()
            newsData = news.copy()
            #Podział danych giełdowych na dane treningowe i testowe.
            evalData, trainData = self.getTrainTestSegments(marketData, i)
            #Podział nagłówków na dane treningowe i testowe.
            evalNewsData, trainNewsData = self.getTrainTestSegments(newsData, i)

            #Wyodrębnienie kursu zamknięcia danej spółki (yTrainData) i 
            #zmiana kształtu danych treningowych.
            xTrainData, yTrainData = self.reshapeData(trainData, trainData[:, 0])
            #Wyodrębnienie kursu zamknięcia danej spółki (yEvalData) i 
            #zmiana kształtu danych testowych.
            xEvalData, yEvalData = self.reshapeData(evalData, evalData[:, 0])

            #Normalizacja danych
            xTrainData, yTrainData = self.normalize(xTrainData, yTrainData, True)
            xEvalData, yEvalData = self.normalize(xEvalData, yEvalData, False)
            #Zwrócenie poza pętle danego segmentu za pomocą klauzuli yield
            yield [xTrainData, yTrainData, xEvalData, yEvalData, trainNewsData, evalNewsData]
        #Gdy brak jest więcej segmentów zwracany jest False
        return False

    #Metoda która służy tylko do wykonania przekształcenia danych
    #oraz ich normalizacji. Używana tylko opodczas trenowania modelu.
    def getFitData(self):
        xTrainData, yTrainData = self.reshapeData(self.dataset, self.dataset[:, 0])
        xTrainData, yTrainData = self.normalize(xTrainData, yTrainData, True)

        return xTrainData, yTrainData

    #Metoda służy do podziału na dane treningowe i dane testowe
    def getTrainTestSegments(self, data, index):
        #w zależności od podanego indexu
        #inna część danych użyta jest jako
        #dane testowe. Metoda pop oddziela
        #dane o podanym indeksie od tablicy
        #o nazwie data i zwraca dane z tego
        #indeksu do tablicy o nazwie evalData.
        evalData = data.pop(index)

        #Reszta danych zostanie wykorzystana jako dane treningowe.
        #Tablica data jest przekształcana do tablicy typu numpy.
        data = np.array(data)
        
        #Zwracacane są z niej poszczególne
        #kształty które zostaną użyte do
        #przekształcenia kształtu tablicy.
        numOfSegmentsAfterPop = data.shape[0]
        numOfElements = data.shape[1]
        numOfFeatures = data.shape[2]

        #Kształt danych treningowych jest zmieniany na
        #Ilość elementów w segmencie x ilość segmentów po usunięciu indeksu testowego (np. 4) x ilość atrybutów
        trainData = np.reshape(data, (numOfElements * numOfSegmentsAfterPop, numOfFeatures))

        return evalData, trainData

    #Metoda służąca do tworzenia modelu.
    #Parametr shape określa rozmiar wejścia.
    def createModel(self, shape):
        #Zdefiniowanie punktu wejściowego o podanym rozmiarze
        #w tym przypadku 30x14
        input1 = tf.keras.layers.Input(shape=shape)

        #Tworzona jest nowa warstwa LSTM liczba 64 określa liczbę
        #jednostek warstwy ukrytej i wyjściowej.
        #Parametr return_sequences określa czy warstwa ma zwracać sekwencję
        #w przeciwnym wypadku zwracana byłaby pojedyńcza wartość.
        #Ostatni zapis (input1) określa podpięcie warstwy wejściowej input1 do
        #warstwy LSTM o nazwie lstm1
        lstm1 = tf.keras.layers.LSTM(64, return_sequences=True)(input1)
        #Stworzona jest kolejna warstwa LSTM połączona z poprzednią
        lstm2 = tf.keras.layers.LSTM(64)(lstm1)

        #Tworzony jest oddzielny pukty wejścia o rozmiarze 12000,
        input2 = tf.keras.layers.Input(shape=(self.VOCAB_SIZE,))
        #Tworzona jest warstwa o rozmiarze warstwy ukrytej i wyjściowej
        #równym 256. Warstwa ta jest spięta z punktem wejściowym o nazwie input2.
        dense1 = tf.keras.layers.Dense(256)(input2)
        #Tworzona jest kolejna warstwa o nazwie dense2 spięta z dense1.
        dense2 = tf.keras.layers.Dense(16)(dense1)

        #Warstwa concatLayer służy do połączenia wektorów z warstw
        #lstm2 i dense2. Tj. do tej warstwy zostają przekazane
        #dwa oddzielne wektory o rozmiarach [64,] i [16,]
        #zostają one połączone do wspólnego wektoru o rozmiarze [80,]
        concatLayer = keras.layers.concatenate([lstm2, dense2])
        #Na koniec warstwa konkatenacyjna łączona jest z
        #kolejną warstwą ukrytą o wymiarze 14. Jest
        #to już docelowa wartość wektora.
        output = tf.keras.layers.Dense(14)(concatLayer)
        #Tworzony jest model właściwy do którego przekazywane są 
        #punkty wejściowe i punkt wyjściowy
        model = tf.keras.Model([input1, input2], output)
        #Model jest kompilowany z użyciem optymyzatora Adam i funkcją straty MAE
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mae')
        #Wypisanie podsumowania modelu
        model.summary()

        return model

    #Metoda przekształcająca dane
    #parametr data przyjmuje dane giełdowe.
    #parametr target przyjmuje kurs zamknięcia
    #badanej spółki.
    def reshapeData(self, data, target):
        #Inicjalizacja pustych tablic do przechowywania
        #przekształconych danych historycznych i danych przyszłych.
        previousData = []
        futureData = []

        #początkowy indeks równy 30
        startIndex = 0 + self.NUMBER_OF_PAST_STEPS
        #końcowy indeks równy liczba elementów w tablicy data
        # + 14.
        endIndex = len(data) - self.NUMBER_OF_FUTURE_STEPS
        for i in range(startIndex, endIndex):
            #Początkowy indeks wynosi 0:30
            #w kolejnych iteracjach rośnie o jeden
            #czyli do tabeli previousData są wpisywane
            #dane z tabeli data z kolejnymi indeksami
            #0:30, 1:31, 2:32, 3:33 itd.
            indices = range(i - self.NUMBER_OF_PAST_STEPS, i, 1)
            previousData.append(data[indices])
            #Do tabeli futureData wpisywane są dane z tabeli target
            #z następującymi indeksami 30:44, 31:45, 32:46 itd.
            futureData.append(target[i:i + self.NUMBER_OF_FUTURE_STEPS])

        return np.array(previousData), np.array(futureData)

    #Metoda to testowania modelu przy
    #pomocy walidacji krzyżowej
    def testModel(self):
        #Dane zwracane są z generatora o nazwie
        #crossValidate.
        for data in self.crossValidate(self.dataset):
            if(data == False):
                break
            else:
                #Dane zwrócone z generatora crossValidate
                #przekazywane są do metody evalModel
                #która trenuje nowy model w oparciu o dane
                #z walidacji krzyżowej.
                self.evalModel(data[0], data[1], data[2], data[3], data[4], data[5])
                #W każdej iteracji walidacji krzyżowej
                #ładowany/tworzony jest nowy model
                self.lstmModel = self.loadModelIfExists()
        
    #Metoda służy do wywołania trenowania modelu.
    #Model jest następnie zapisywany.
    def trainModel(self):
        #Pobierane są dane potrzebne w procenie trenowania.
        xTrainData, yTrainData = self.getFitData()
        #Wywołany proces trenowania w oparciu o dane
        #xTrainData i yTrainData
        self.fitModel(xTrainData, yTrainData)
        #Model zostaje zapisany
        self.lstmModel.save('MLMODEL.h5') 

    #Metoda wykonuje prognozę w oparciu
    #o już wytrenowany model.
    def predictVal(self):
        #Zapis danych historycznych do zmiennej.
        trueVals = self.dataset[:, 0]
        #Załadowanie obiektów klasy MinMaxScaler
        #służących do normalizacji danych.
        xScaler = self.openScaler('xScaler')
        yScaler = self.openScaler('yScaler')
        #Przekształcenie danych historycznych.
        historcalData = np.reshape(self.dataset, (30, self.NUMBER_OF_FEATURES))
        #Jeżeli dane prawdziwe przyszłe
        #nie istnieją to zostają pominięte.
        if(len(self.trueData) == 14):
            yData = self.trueData.values
            yData = np.reshape(yData, (1, self.NUMBER_OF_FUTURE_STEPS))
            #Konkatenacja ostatniego indeksu z tablicy
            #wartości historycznych z pierwszym
            #indeksem wartości prawdziwych.
            yData = np.concatenate([[trueVals[29]],yData[0,:]])
            #Wyrysowanie na wykresie wartości prwadziwych.
            plt.plot(range(0,15), yData, 'b-', markersize=1, label='Prawdziwa przyszła')

        #Normalizacja danych historycznych.
        historcalData = xScaler.transform(historcalData)
        #Przekształcenie kształtu tabeli z nagłówkami wiadomości.
        text = np.reshape(self.tokenizedText[29], (1, self.VOCAB_SIZE))
        #Przekształcenie rozmiaru danych historycznych.
        historcalData = np.reshape(historcalData, (1, 30, self.NUMBER_OF_FEATURES))
        
        #Wykonanie prognozy właściwej, w oparciu o 
        #tabelę z danymi historycznymi i tabelę z nagłówkami wiadomości
        prediction = self.lstmModel.predict([historcalData, text])
        #Po wykonaniu prognozy wartości prognozy muszą
        #zostać zdenormalizowane za pomocą
        #metody inverse_transform z biblioteki MinMaxScaler
        prediction = yScaler.inverse_transform(prediction)
        prediction = np.reshape(prediction, (14,))
        prediction = np.concatenate([[trueVals[29]],prediction])
        #Wyświetlenie wykresów
        plt.plot(range(-29,1), self.dataset[:, 0] , 'g-', markersize=1, label='Poprzednie')
        plt.plot(range(0,15), prediction, 'r-', markersize=1, label='Przewidziana przyszła')
        ax = plt.gca()
        ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
        ax.tick_params(axis = 'both', which = 'minor', labelsize = 16)
        plt.legend(prop={'size': 16})
        plt.grid()
        plt.show()

    #Metoda odpowiadająca za wczytanie obiektu
    # #klasy MinMaxScaler z pliku.    
    def openScaler(self, name):
        fileName = name + '.pickle'

        if(os.path.isfile(fileName)):
            with open(fileName, 'rb') as handle:
                scaler = pickle.load(handle)

        return scaler

    #Metoda właściwa do testowania modelu
    #Dane treningowe - xTrainData, yTrainData, trainNewsData
    #Dane testowe - xEvalData, yEvalData, evalNewsData
    def evalModel(self, xTrainData, yTrainData, xEvalData, yEvalData, trainNewsData, evalNewsData):
        #Zdefiniowanie ścieżki w której zapisywane będą logi uczenia
        logdir="logs\\fit\\" + dt.now().strftime("%Y%m%d-%H%M%S")
        #Zdefiniowanie callbacku tensorboard który będzie wykonywał zapis logów
        tensorboardCallback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
        
        #Zdefiniowanie callbacku który zatrzyma proces uczenia w momencie wykrycia przetrenowania
        earlystopCallback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
            patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False
        )

        #Wywołanie procesu uczenia na modelu
        #Do metody fit przekazane są dane treningowe i testowe (validation_data)
        #Zdefiniowana jest liczba epok (epochs = 10)
        #Przekazane są zdefiniowane wcześniej callbacki (callbacks)
        history = self.lstmModel.fit([xTrainData, trainNewsData[:xTrainData.shape[0]]], yTrainData[:, 0], 
            validation_data = ([xEvalData, evalNewsData[:xEvalData.shape[0]]], yEvalData[:, 0]), 
            epochs=10, callbacks=[tensorboardCallback, earlystopCallback])

        #Wykonywane jest testowe wyrysowanie wyników na wykresie
        self.plotPred(xEvalData, yEvalData, evalNewsData)        
    
    #Metoda właściwa odpowiedzialna za proces uczenia modelu
    #W parametrach przekazywane są wyłącznie dane treningowe
    def fitModel(self, xTrainData, yTrainData):
        #Zdefiniowany callback zatrzymujący proces uczenia w momencie
        #wykrycia przetrenowania.
        earlystopCallback = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0,
            patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False
        )

        #Przekazanie danych treningowych do modelu i rozpoczęcie procesu uczenia.
        trainNews = self.tokenizedText[:xTrainData.shape[0],:]
        history = self.lstmModel.fit([xTrainData, trainNews], yTrainData[:, 0],
            epochs=2)

    #Metoda służąca do wyrysowania wykresu pokazującego
    #zestawienie danych prognozowanych, prawdziwych i historycznych.
    #Parametr xEvalData przyjmuje dane użyte do wykonania prognozy przez model.
    #Parametr yEvalData przyjmuje prawdziwe dane przyszłe.
    #Parametr evalNews przyjmuje nagłówki wiadomości potrzebne do 
    #wykonania prognozy.
    def plotPred(self, xEvalData, yEvalData, evalNews):
        #Losowana jest liczba wykorzystywana do określenia
        #który wycinek danych zostanie wyrysowany.
        n = random. randint(0,len(xEvalData))

        predictionVal = np.reshape(xEvalData[n], (1, self.NUMBER_OF_PAST_STEPS, self.NUMBER_OF_FEATURES))
        #zwrócenie prawdziwych cen akcji
        #do tabeli trueY
        trueY = np.array(yEvalData[n])
        #przekształcenie danych tekstowych do rozmiaru 1x12000
        #co wskazuje, że predykcja wykonywana jest tylko
        #na 14 dni
        textVal = np.reshape(evalNews[n], (1, self.VOCAB_SIZE))
        #Wykonanie prognozy przez model
        prediction = self.lstmModel.predict([predictionVal, textVal])
        prediction = np.reshape(prediction, (self.NUMBER_OF_FUTURE_STEPS, 1))

        #Denormalizacja wartości Y prognozy i prawdziwych
        trueY = np.reshape(trueY, (self.NUMBER_OF_FUTURE_STEPS, 1))
        trueFuture = self.denormalizeTarget(trueY)

        prediction = self.denormalizeTarget(prediction)

        #Wywrysowanie wykresów
        plt.plot(range(0,14), trueFuture, '-', markersize=1, label='Prawdziwa przyszła')
        plt.plot(range(0,14), prediction, 'r-', markersize=1, label='Przewidziana przyszła')
        xData = self.denormalizeTrain(xEvalData[n])
        plt.plot(range(-29,1),xData[:, 0] , 'g-', markersize=1, label='Poprzednie')
        plt.legend(loc='best')
        plt.grid()
        plt.show()

    #Metoda służąca do zdenormalizowania danych treningowych.
    #Wykorzystuje metodę klasy MinMaxScaler o nazwie
    #inverse_transform do odwrócenia normalizacji. 
    def denormalizeTrain(self, value):
        valToDenormalize = np.reshape(value, (self.NUMBER_OF_PAST_STEPS, self.NUMBER_OF_FEATURES))
        denormalizedVal = self.Xscaler.inverse_transform(valToDenormalize)

        return denormalizedVal

    #Metoda służąca do zdenormalizowania danych prognozowanych.
    #Wykorzystuje metodę klasy MinMaxScaler o nazwie
    #inverse_transform do odwrócenia normalizacji. 
    def denormalizeTarget(self, value):
        val = value.copy()
        valToDenormalize = np.reshape(val, (1, self.NUMBER_OF_FUTURE_STEPS))
        denormalizedVal = self.Yscaler.inverse_transform(valToDenormalize)
        denormalizedVal = np.reshape(denormalizedVal, (self.NUMBER_OF_FUTURE_STEPS,))

        return denormalizedVal

    #Metoda służąca do znormalizowania podanych danych.
    #W parametrze value przekazywane są dane giełdowe z 
    #wyłączeniem kursu zamknięcia badanej spółki.
    #W parametrze target przekazywany jest kurs zamknięcia
    #badanej spółki.
    #Parametr trainOnly określa czy normalizacja dokonywana jest dla
    #zbioru testowego czy treningowego.
    def normalize(self, value, target, trainOnly):
        valCopy = value.copy()
        targetCopy = target.copy()
        #Zanim dane zostaną przeskalowane, są wpierw przekształcane wa taki sposób
        #aby pasowały do metody fit_transform klasy MinMaxScaler
        v1 = np.reshape(valCopy, (value.shape[0] * value.shape[1], self.NUMBER_OF_FEATURES))
        v2 = np.reshape(targetCopy, (target.shape[0], self.NUMBER_OF_FUTURE_STEPS))

        if(trainOnly == True):
            #Stworzenie osobnych obiektów klasy MinMaxScaler do
            #zbioru testowego i zbioru treningowego.
            self.Xscaler = MinMaxScaler()
            self.Yscaler = MinMaxScaler()

            #Uczenie obiektów biblioteki MinMaxScaler z jednoczesnym
            #znormalizowaniem.
            #Wyuczone obiekty MinMaxScaler są następnie zapisywane.
            dataX = self.Xscaler.fit_transform(v1)
            self.saveScaler(self.Xscaler, 'Xscaler')
            dataY = self.Yscaler.fit_transform(v2)
            self.saveScaler(self.Yscaler, 'Yscaler')
        else:
            self.Xscaler = self.openScaler('Xscaler')
            self.Yscaler = self.openScaler('Yscaler')
            dataX = self.Xscaler.fit_transform(v1)
            dataY = self.Yscaler.fit_transform(v2)
        
        #Dane są przekształcane do początkowego kształtu i zwracane poza metodę.
        dataX = np.reshape(dataX, (value.shape[0], value.shape[1], self.NUMBER_OF_FEATURES))
        dataY = np.reshape(dataY, (value.shape[0], self.NUMBER_OF_FUTURE_STEPS))

        return dataX, dataY

    #Metoda służąca do zapisu w pliku obiektu klasy MinMaxScaler.
    def saveScaler(self, value, name):
        fileName = name + '.pickle'

        if(os.path.isfile(fileName)):
            return
        else:
            with open(fileName, 'wb') as handle:
                pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)

class DatabaseConnection:
 
    #Konstruktor w krórym wykonywane jest połączenie
    #z bazą danych.
    def __init__(self):
        self.conn = pyodbc.connect('Driver={SQL Server};'
                                   'Server=.\SQLEXPRESS;'
                                   'Database=Stonks;'
                                   'Trusted_connection=yes;')

    #Metoda ta zwraca całość danych dostępnych w bazie dla podanej spółki.
    def getCompanyData(self, ticker):
        sql = "select md.[close] as close_x, md.volume as volume_x, \
        md1.[open], md1.high, md1.low, md1.[close], md1.volume, \
        nh.headlines \
        from market_data md \
        left join market_data md1 on md1.date = md.date \
        left join company_info c1 on c1.id = md1.company_info_id \
        left join news_headlines nh on nh.date = md.date \
        left join company_info ci on ci.id = md.company_info_id \
        where c1.symbol = '^IXIC' and nh.headlines is not null and ci.symbol = ?"
        data = pd.read_sql(sql, con=self.conn, params=[ticker])
        return data
    
    #Metoda używana do dotrenowania modelu. Zwraca dane
    #o rok wcześniejsze względem daty wykonania skryptu.
    def getCompanyDataToRetrain(self, ticker):
        sql = "select md.[close] as close_x, md.volume as volume_x, \
        md1.[open], md1.high, md1.low, md1.[close], md1.volume, \
        nh.headlines \
        from market_data md \
        left join market_data md1 on md1.date = md.date \
        left join company_info c1 on c1.id = md1.company_info_id \
        left join news_headlines nh on nh.date = md.date \
        left join company_info ci on ci.id = md.company_info_id \
        where c1.symbol = '^IXIC' and md.date > DATEADD(year,-1,GETDATE()) \
        nh.headlines is not null and ci.symbol = ?"
        data = pd.read_sql(sql, con=self.conn, params=[ticker])
        return data

    #Metoda zwracająca dane potrzebne do wykonania prognozy na
    #kolejne 14 dni począwczszy od daty podanej w parametrze o nazwie date.
    #Dane zwrócone przez tą metodę są starsze o 30 dni względem parametru
    #o nazwie date.
    def getCompanyDataToPredict(self, ticker, date):
        sql = "select top 30 md.[close] as close_x, md.volume volume_x, \
        md1.[open], md1.high, md1.low, md1.[close], md1.volume, \
        nh.headlines \
        from market_data md \
        left join market_data md1 on md1.date = md.date \
        left join company_info c1 on c1.id = md1.company_info_id \
        left join news_headlines nh on nh.date = md.date \
        left join company_info ci on ci.id = md.company_info_id \
        where c1.symbol = '^IXIC' \
		and nh.headlines is not null \
		and ci.symbol = ? \
		and md1.date > DATEADD(DAY, -44, ?) \
		order by md.date asc"
        data = pd.read_sql(sql, con=self.conn, params=[ticker, date])

        return data

    #Metoda zwracająca prawdziwe dane giełdowe z bazy jeżeli istnieją.
    #Wykorzystywana jest przy prognozowaniu, żeby porównać wartość prognozowaną
    #z wartością prawdziwą.
    def getTrueDataPredictedIfExists(self, ticker, date):
        sql = "select top 14 md.[close] as close_x \
        from market_data md \
        left join market_data md1 on md1.date = md.date \
        left join company_info c1 on c1.id = md1.company_info_id \
        left join company_info ci on ci.id = md.company_info_id \
        where c1.symbol = '^IXIC' \
		and ci.symbol = ? \
		and md1.date >= ? \
		order by md.date asc"
        data = pd.read_sql(sql, con=self.conn, params=[ticker, date])

        return data

    #Metoda zwracająca wszystkie symbole spółek znajdujących się w bazie.
    def getAllSymbols(self):
        sql = "select distinct(symbol) from company_info"
        data = pd.read_sql(sql, con=self.conn)
        return data

    #Metoda pomocnicza do sprawdzenia czy podany parametr jest formatem typu data.
    def isDate(self, string):
        try: 
            parse(string)
            return True

        except ValueError:
            return False

if __name__ == '__main__':
    #Pobranie początkowego parametru z konsoli.
    param = sys.argv[1]
    #Utworzenie obiektu klasy DatabaseConnection.
    db = DatabaseConnection()
    #Pobranie wszystkich symboli spółek obecnych w bazie.
    tickers = db.getAllSymbols()
    if param == "help":
        print("List of parameters: \n \
        help - shows list of parameters \n \
        trainAll - trains model and saves it to local folder \n \
        testAll - uses saved model to test it")
        exit()
    elif param == "listTickers":
        tickers = db.getAllSymbols()
        print(tickers)
    elif param == "trainAll":
        for i, ticker in tickers.iterrows():
            if(ticker['symbol'] != '^IXIC'):
                data = db.getCompanyData(ticker['symbol'])
                Machine_learning(data).trainModel()
    elif param == "testAll":
        for i, ticker in tickers.iterrows():
            if(ticker['symbol'] != '^IXIC'):
                data = db.getCompanyData(ticker['symbol'])
                Machine_learning(data).testModel()
    elif param == "train":
        ticker = input("Podaj symbol firmy \n")
        data = db.getCompanyData(ticker)
        if(not data.empty):
            Machine_learning(data).trainModel()
    elif param == "test":
        ticker = input("Podaj symbol firmy \n")
        data = db.getCompanyData(ticker)
        if(not data.empty):
            Machine_learning(data).testModel()
    elif param == "retrain":
        for i, ticker in tickers.iterrows():
            if(ticker['symbol'] != '^IXIC'):
                data = db.getCompanyDataToRetrain(ticker['symbol'])
                Machine_learning(data).trainModel()
    elif param == "predict":
        ticker = input("Podaj symbol firmy \n")
        date = input("Podaj początkową datę prognozy (format YYYY-mm-dd) \n")
        if(db.isDate(date)):
            data = db.getCompanyDataToPredict(ticker, date)
            trueData = db.getTrueDataPredictedIfExists(ticker, date)
            print(data)
        else:
            print("Podaj date w formacie YYYY-mm-dd")
        if(not data.empty):
            Machine_learning(data, trueData=trueData).predictVal()