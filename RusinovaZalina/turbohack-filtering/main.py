import pandas as pd
import numpy as np
from abc import abstractmethod, ABC
from sklearn.preprocessing import StandardScaler

# Путь до неразмеченных данных. Является константой, нельзя поменять маршрут монтирования данных.
# Тренировочные / тестовые данные для локальной работы следует размещать по этому маршруту
GT_RAW = "./data/public_test.csv"

# Путь, по которому должен генерироваться итоговый сабмит.
SUBM_PATH = "./data/submission.csv"


class AbstractAnomalyDetector(ABC):
    @abstractmethod
    def __init__(self, data: pd.Series, interval=None):
        """
        Detect anomalies and get label for each anomaly
        :param data: datetime indexed pd.Series
        :param interval: pair of dates to search for anomalies between
        """
        self.data = data
        if interval:
            self.start, self.end = interval
            if not self.start:
                self.start = 0
            if not self.end:
                self.end = self.data.shape[0]
        else:
            self.start, self.end = 0, self.data.shape[0]
        pass

    @abstractmethod
    def get_labels(self) -> pd.Series:
        pass


class OutlierDetector(AbstractAnomalyDetector):
    """
    Detects anomalies using distribution of data
    Detects:
            Outliers - such points where values are not in 3-sigma range of distribution
                (in other words values are too big or too low than the rest of the data);
    """

    def __init__(self, data: pd.Series, interval=None):
        super().__init__(data, interval)

    def get_labels(self):
        result = self._search_for_anomalies()
        result = result[~np.isnan(result)]
        result = pd.DataFrame(index=result.index, data={'label': result.values})['label']
        if self.start and self.end:
            result = result[self.start:self.end]
        return result

    def _search_for_anomalies(self):
        sigma_min, sigma_max = self.__get_stat()
        return self.data.apply(lambda r: 1 if r > sigma_max else -1 if r < sigma_min else np.nan)

    def __get_stat(self):
        mean_val = self.data.mean()
        std_val = self.data.std()
        sigma_min = mean_val - 3 * std_val
        sigma_max = mean_val + 3 * std_val
        return sigma_min, sigma_max


def clean_outliers(df):
    df_raw = df.copy()
    df_rez = df_raw.copy()
    for i in df_raw.columns:
        if (abs(df_raw[i].mean()/df_raw[i].median())>2) or (abs(df_raw[i].max()-df_raw[i].min())>10*abs(df_raw[i].std())):
            result = OutlierDetector(df_raw[i]).get_labels()
            df_raw[i] = pd.Series(result)
        else:
            df_raw[i] = pd.Series(np.zeros(len(df_raw[i])))
    df_raw["result"] = df_raw.sum(axis=1)

    del_indexes = df_raw[df_raw["result"] != 0].index

    df_rez.drop(index=del_indexes, inplace=True)

    return df_rez


def clean_outliers_in_corrs(df):
    df_raw = df.copy()
    df_rez = df_raw.copy()

    scaler = StandardScaler()
    np_scaled = scaler.fit_transform(df_rez)
    data = pd.DataFrame(np_scaled)
    data.columns = df_raw.columns
    data.index = df_raw.index

    pair_dict = {}
    list_except = []
    for i in data.columns:
        if i not in list_except:
            for j in data.columns:
                if (i != j):
                    del_series = np.abs((data[i]) - (data[j]))
                    if len(del_series[del_series < 0.03]) / len(data) > 0.7:
                        pair_dict[i] = j
                        list_except.append(j)

    list_means = []
    for col1, col2 in pair_dict.items():
        del_series = np.abs((data[col1]) - (data[col2]))
        list_means.append(del_series.mean())

    list_indxes = []
    for col1, col2 in pair_dict.items():
        del_series = np.abs((data[col1]) - (data[col2]))
        list_indxes += list((del_series[del_series > 20*np.mean(list_means)].index))

    rez = np.array(list(set(list_indxes)))
    df_rez.drop(index=rez, inplace=True)

    return df_rez


def baseline_subm(gt_raw):
    subm_df = gt_raw.copy()
    df_without_outliers = clean_outliers(subm_df)
    df_without_corr_outliers = clean_outliers_in_corrs(df_without_outliers)

    df_without_corr_outliers.to_csv(SUBM_PATH)


def main():
    gt_raw = pd.read_csv(GT_RAW).set_index("Параметр")
    subm_df = baseline_subm(gt_raw)


if __name__ == "__main__":
    main()
