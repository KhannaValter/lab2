import csv  # read and write tabular data in CSV format
import sys
from tkinter import Tk, Label, Entry, Button

import controller as controller
import pandas as pd  # библиотека для анализа данных

from matplotlib import pyplot as plt  # для графиков
import re  # регулярные выражения
import nltk

from nltk.tokenize import word_tokenize  # для разбиения на предложения

nltk.download('punkt')  # загрузить все данные nltk
import numpy as np  # массивы

from tkinter.filedialog import askopenfilename

ham_list = list()
spam_list = list()

columns = ["v1", "v2"]  # надписи над ham/spam и соответствующими им сообщениями

Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file

fileReader = pd.read_csv(filename, encoding='ISO-8859-1', usecols=columns)  # считываем из .csv файла
size = len(fileReader)

stopWords = ['a', 'in', 'to', 'the']

for i in range(size):
    string = fileReader.loc[i, 'v2']
    string = string.lower()
    string = re.sub(r'[^A-Za-z\s]+', '', string)
    fileReader.loc[i, 'v2'] = string  # возвращаем нормализованное слово
    tokens = word_tokenize(string)
    no_stopwords_list = [k for k in tokens if not k in stopWords]
    if str(fileReader.loc[i, 'v1']) == 'ham':
        ham_list.extend(no_stopwords_list)
    elif str(fileReader.loc[i, 'v1']) == 'spam':
        spam_list.extend(no_stopwords_list)
    string = ' '.join(no_stopwords_list)
    fileReader.loc[i, 'v2'] = string

fileReader.to_csv("update-sms-spam-corpus.csv", index=False)  # нормализированный массив предложений

count_ham_dict = dict()
for i in ham_list:
    count_ham_dict[i] = count_ham_dict.get(i, 0) + 1

count_spam_dict = dict()
for i in spam_list:
    count_spam_dict[i] = count_spam_dict.get(i, 0) + 1

words_ham = pd.DataFrame.from_dict(count_ham_dict, orient="index")
words_spam = pd.DataFrame.from_dict(count_spam_dict, orient="index")

words_ham.to_csv("ham_frequency_words.csv")
words_spam.to_csv("spam_frequency_words.csv")

with open('update-sms-spam-corpus.csv', 'r', encoding='ISO-8859-1') as read_file:
    csvReader = csv.reader(read_file)
    stopWords = ['a', 'in', 'to', 'the']
    myPorterStemmer = nltk.stem.porter.PorterStemmer()  # стемминг, выбираем основу слова
    str_arrays = []
    ham_array = {}
    spam_array = {}
    for line in csvReader:
        if line[0] == 'ham':
            for word in line:
                if word != '':
                    if word not in stopWords:
                        ham_array[myPorterStemmer.stem(word)] = ham_array.setdefault(myPorterStemmer.stem(word), 0) + 1
        else:
            for word in line:
                if word != '':
                    if word not in stopWords:
                        spam_array[myPorterStemmer.stem(word)] = spam_array.setdefault(myPorterStemmer.stem(word),
                                                                                       0) + 1

    ham_array = dict(sorted(ham_array.items(), key=lambda item: len(item[0]), reverse=True))
    str_arrays.append(ham_array)
    spam_array = dict(sorted(spam_array.items(), key=lambda item: len(item[0]), reverse=True))
    str_arrays.append(spam_array)

    with open('sms_all_words.csv', 'w', newline='') as dictionary:
        writer = csv.writer(dictionary)
        writer.writerow(['type', 'word', 'length'])

        for key in str_arrays[0].keys():
            writer.writerow(['ham', key, len(key)])

        for key in str_arrays[1].keys():
            writer.writerow(['spam', key, len(key)])

wordsAmount = 0
countWordsHam = 0
countWordsSpam = 0

with open('ham_frequency_words.csv', 'r') as dictionary:
    reader = csv.reader(dictionary)
    for line in reader:
        if line[0] != 'word':
            wordsAmount += int(line[1].replace('\n', ''))
            countWordsHam += int(line[1].replace('\n', ''))

with open('spam_frequency_words.csv', 'r') as dictionary:
    reader = csv.reader(dictionary)
    for line in reader:
        if line[0] != 'word':
            wordsAmount += int(line[1].replace('\n', ''))
            countWordsSpam += int(line[1].replace('\n', ''))

pHam = countWordsHam / wordsAmount
pSpam = countWordsSpam / wordsAmount

all_words = pd.read_csv('sms_all_words.csv')
hams = all_words[all_words.type == 'ham']
spams = all_words[all_words.type == 'spam']

window = Tk()
window.title("Second lab")
window.configure(background="steelblue")
window.columnconfigure([0, 1, 2, 3, 6], minsize=500)
window.rowconfigure([0, 1, 2], minsize=40)
window.rowconfigure([3, 6], minsize=100)
window.geometry('500x300')
window.resizable(width=False, height=False)
lbl = Label(window, text="Введите слово:", font=("Ghotam Pro", 14))
lbl.grid(column=0, row=0)

txt = Entry(window, width=20)
txt.grid(column=0, row=1)

lbl2 = Label(window, font=("Ghotam Pro", 14))
lbl3 = Label(window, font=("Ghotam Pro", 14))
lbl4 = Label(window, font=("Ghotam Pro", 14))


def clicked():
    ham_dictionary = pd.read_csv('ham_frequency_words.csv', header=None, index_col=0, squeeze=True).to_dict()
    spam_dictionary = pd.read_csv('spam_frequency_words.csv', header=None, index_col=0, squeeze=True).to_dict()

    # получение слов и их обработка
    extracted_words = str(txt.get())
    processed_words = []
    for word in re.sub(r'[^A-Za-z\s]+', '', extracted_words).lower().split(" "):
        if word != '':
            if word not in stopWords:
                processed_words.append(word)

    p_text_ham = 1
    p_text_spam = 1
    counter_unknown_ham = 0
    counter_unknown_spam = 0
    all_hams = 0
    all_spams = 0

    for frequency in ham_dictionary.values():
        all_hams += int(frequency)

    for frequency in spam_dictionary.values():
        all_spams += int(frequency)

    # кількість повідомлень з категорії ham / загальна кількість повідомлень
    p_ham = all_hams / (all_spams + all_hams)
    # кількість повідомлень з категорії spam / загальна кількість повідомлень
    p_spam = all_spams / (all_spams + all_hams)

    # подсчет количества неизвестных HAM и SPAM
    for word in processed_words:
        if word not in ham_dictionary.keys():
            counter_unknown_ham += 1
        if word not in spam_dictionary.keys():
            counter_unknown_spam += 1

    # расчет возможностей
    for word in processed_words:
        number_of_words_repeating = 0

        # подсчет количества слов, повторяющихся в словаре HAM
        if word in ham_dictionary.keys():
            number_of_words_repeating = int(ham_dictionary[word])

        # (кількість word1 які належать категорії ham + 1) / (загальна кількість слів,
        # які належать категорії ham +кількість слів, яких немає в навчальній вибірці)
        p_text_ham *= (number_of_words_repeating + 1) / (len(ham_dictionary) + counter_unknown_ham)

        # подсчет количества повторений слов в словаре SPAM
        if word in spam_dictionary.keys():
            number_of_words_repeating = int(spam_dictionary[word])
        lbl4 = Label(window, font=("Ghotam Pro", 14))
        lbl4.configure(text=number_of_words_repeating)
        lbl4.grid(column=0, row=4)
        # (кількість word1 які належать категорії spam + 1) / (загальна кількість слів,
        # які належать категорії spam +кількість слів, яких немає в навчальній вибірці)
        p_text_spam *= (number_of_words_repeating + 1) / (len(spam_dictionary) + counter_unknown_spam)

    p_text_ham *= p_ham    # P(ham | text)   = P(ham) * P(text | ham) / P(text)
    p_text_spam *= p_spam  # P(spam | text) = P(spam) * P(text | spam) / P(text)

    # результат
    if p_text_ham > p_text_spam:
        lbl2.configure(text='Ham\n' + 'p(ham) = ' + str(p_text_ham) + '\np(spam) = ' + str(p_text_spam))
        lbl2.grid(column=0, row=3)
    else:
        lbl2.configure(text='Spam\n' + 'p(ham) = ' + str(p_text_ham) + '\np(spam) = ' + str(p_text_spam))
        lbl2.grid(column=0, row=3)


btn = Button(window, text="Поиск", command=clicked)
btn.grid(column=0, row=2)


def quit_program():
    window.destroy()
    sys.exit()


button3 = Button(window, text="Выход", command=quit_program)
button3.grid(column=0, row=6)

window.mainloop()
