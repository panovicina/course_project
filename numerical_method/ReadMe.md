# Численный метод
Ссылка на код численнного метода: https://bitbucket.org/Alolag/nonlinearsolver/src/worked/  
__Пример запуска для трехвидовой модели__: ./a -t <Title> [-species <3>] [-dim <1>] [-al <0.400000>] [-points <512>] [-a <2.000000>]  
                                           [-b <b [Species]>...] [-dvec <d [Species]>...] [-dmat <d' [Species * Species]>...] [-sw <sigma w [Species * Species]>...] [-sm <sigma m [Species]>...]  
__Параметры запуска__: -t - Название файлов для записи результатов; -species - Количество видов; -dim - Размерность пространства,  
                   на котором считается решение; -al - a - Из замыкания; -points - Количество точек для векторов, по которым 
                   считается решение; -a - Размер пространства, на котором считается решение; -b - Темп рождаемости; -dvec -  
                   Вероятность смертности от окружающей среды; -dmat - Сила конкуренции; -sw - Значение среднеквадратичного  
                   отклонения для ядра конкуренции; -sm - Значение среднеквадратичногоотклонения для ядра рождаемости.
