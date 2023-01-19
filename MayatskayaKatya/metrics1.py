import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import datetime as dt
import tsad
from tsad.evaluating.evaluating import evaluating
from tqdm import tqdm



class Metrics:

    def __init__(self, y_test, y_pred, time_index, window):
        self.window = window
        self.y_test = y_test
        self.y_pred = y_pred
        self.time_index = time_index

        '''
        Обозначения:
        window - размер окна (10% от всего ряда, деленное на число реальных аномалий)
        time_index - столбец с временной меткой
        y_test - столбец с метками аномалий (0 -нет аномалии, 1- есть аномалия)
        y_pred - столбец с предсказанными метками аномалий 
        '''

        
    def maerror(self):
        '''
        Получаем оценку насколько в среднем предсказанная точка изменения (аномалия) близка к действительной
        Результат выражен в единицах времени
        '''
        today = dt.date.today()
        duration0 = dt.timedelta(minutes=0, seconds=0)

        for i in tqdm(range(len(self.y_pred))):
            time_true = 0
            time_pred = 0

            if self.y_test[i] == 1 and self.y_pred[i] != 1:
                time_true = dt.datetime.strptime(str(self.time_index[i]), '%Y-%m-%d %H:%M:%S').time()

                for k in range(i, len(self.y_pred)):
                    if self.y_pred[k] == 1:
                        time_pred = dt.datetime.strptime(str(self.time_index[k]), '%Y-%m-%d %H:%M:%S').time()
                        break

            if self.y_test[i] != 1 and self.y_pred[i] == 1:
                time_pred = dt.datetime.strptime(str(self.time_index[i]), '%Y-%m-%d %H:%M:%S').time()

                for k in range(i, len(self.y_test)):
                    if self.y_test[k] == 1:
                        time_true = dt.datetime.strptime(str(self.time_index[k]), '%Y-%m-%d %H:%M:%S').time()
                        break           

            if self.y_test[i] == 1 and self.y_pred[i] == 1:     
                time_true = dt.datetime.strptime(str(self.time_index[i]), '%Y-%m-%d %H:%M:%S').time() 
                time_pred = dt.datetime.strptime(str(self.time_index[i]), '%Y-%m-%d %H:%M:%S').time()

            if time_true != 0 and time_pred != 0:        
            #находим разницу между реальным и предсказанным значением
                if time_true > time_pred:
                    duration0 += abs(dt.datetime.combine(today, time_pred) - dt.datetime.combine(today, time_true))

                # рассчитываем MAE (нормируем по числу аномальных точек)
        mae = duration0/(sum(self.y_pred))
        print("MAE: {}". format(mae))

        return mae
        

    def classif_metrics(self):

        '''
        Рассчитываем метрики классификации
        '''
        df = pd.concat([self.y_test,self.y_pred],1, keys=["true", "pred"]).reset_index()

        # Рассчитаем TP, FP, FN, TN
        TP, FP, FN, TN = 0, 0, 0, 0

        eq = df[df['true'] == df['pred']]

        TP = eq[eq['true'] == 1].shape[0]
        TN = eq[eq['true'] == 0].shape[0]

        not_eq = df[df['true'] != df['pred']]

        FP = not_eq[not_eq['true'] == 0].shape[0]
        FN = not_eq[not_eq['true'] == 1].shape[0]

        print("Classification report")
        print('True Positives:%.0f' % (TP), end='\t')
        print('False Positives:%.0f' % (FP))
        print('False Negatives:%.0f' % (FN), end='\t')
        print('True Negatives:%.0f' % (TN))

        # Рассчитаем: Accuracy, Recall, Precission, F1 measure
        print('Accuracy:%.4f' % ((TP + TN) / (TP + TN + FP + FN)))
        print('Recall:%.4f' % (TP / (TP + FN)), end='\t')
        print('Precision:%.4f' % (TP / (TP + FP)))
        print('f1 measure:%.4f' % (TP / (TP + 0.5 * (FP + FN))))

        return TP, TN, FP, FN

    def nab_metric(self):

        '''
        Особенностями этого алгоритма являются вознаграждение за раннее обнаружение аномалии и штрафование
        за ложноположительные и ложноотрицательные результаты.
        '''

        df = pd.concat([self.y_test, self.y_pred, self.time_index], 1, keys=["true", "pred", "time"]).reset_index()
        df.set_index('time', inplace=True)

        results = evaluating(true=df["true"], prediction=df["pred"],
                             numenta_time=self.window,
                             anomaly_window_destenation='center',  # Расположение окна относительно аномалии
                             metric='nab', clear_anomalies_mode=True, plot_figure=True)
        # Результат:
        # Standart - назначает TPs, FPs и FNs с относительными весами (привязанными к размеру окна)
        # случайные обнаружения, сделанные в 10% случаев, получат в среднем нулевую итоговую оценку
        # LowFP и LowFN - начисляют большие штрафы за FPs и FNs, предназначены для иллюстрации поведения алгоритма.