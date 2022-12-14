### DeepXF: Прогнозирование с использованием современных глубоких нейронных сетей и динамической факторной модели

DeepXF - это библиотека с открытым исходным кодом на языке python для задач прогнозирования. DeepXF помогает в разработке сложных моделей прогнозирования и прогнозирования с встроенной утилитой для данных временных рядов. С помощью этого простого, удобного в использовании решения можно автоматически строить интерпретируемые модели глубокого прогнозирования. Оно позволяет пользователям быстро и эффективно проводить испытания Proof-Of-Concept (POC). Можно строить модели на основе глубоких нейронных сетей, таких как рекуррентная нейронная сеть (RNN), долговременная память (LSTM), рекуррентный блок с управлением (GRU), двунаправленная RNN/LSTM/GRU (BiRNN/BiLSTM/BiGRU), (SNN), (GNN), трансформеры, (GAN), (CNN) и другие. В программе также предусмотрена возможность построения прогнозной модели с использованием динамической факторной модели.

![download](https://user-images.githubusercontent.com/113238801/203783723-70823c19-0a66-4e2b-9bb2-f9ae3145cfb1.png)

DeepXF разработан Аджаем Аруначаламом - https://www.linkedin.com/in/ajay-arunachalam-4744581a/

Библиотека обеспечивает (не ограничиваясь этим):-

- Анализ данных с такими возможностями, как профилирование, фильтрация выбросов, построение унивариантных/многомерных графиков, интерактивные графики plotly, графики скользящего окна, обнаружение пиков и т.д.

- Предварительная обработка данных для временных рядов с такими возможностями, как поиск отсутствующих значений, извлечение даты-времени, создание единой временной метки, удаление нежелательных признаков и т.д.

- Описательная статистика для предоставленных данных временных рядов, оценка нормальности и т.д.

- Работа с признаками с такими возможностями, как генерация временных задержек, признаков даты-времени, одноточечное кодирование, циклические признаки даты-времени и т.д.

- Поиск сходства между однородными входными данными временных рядов с помощью сиамских нейронных сетей.

- Шумоподавление входных сигналов временных рядов.

- Построение модели глубокого прогнозирования с настройкой гиперпараметров и использованием доступных вычислительных ресурсов (CPU/GPU).

- Оценка эффективности модели прогнозирования с помощью нескольких ключевых метрик.

- Построение модели прогнозирования с использованием алгоритма максимизации ожиданий

- Nowcasting

### Требования

- Python 3.6.x
- torch[>=1.4.0]
- NumPy[>=1.9.0]
- SciPy[>=0.14.0]
- Scikit-learn[>=0.16]
- statsmodels[0.12.2]
- Pandas[>=0.23.0]
- Matplotlib
- Seaborn[0.9.0]
- tqdm
- shap
- keras[2.6.0]
- pandas_profiling[3.1.0]
- py-ecg-detectors

