## CocktailMan

Проект будет представлять собой интернет-сайт, на котором работают обученные мной нейросетевые модели.

#### Этапы:

**1.** ***Обучение модели*** (multilabel classifier), которая по изображению коктейля может предложить набор ингридиентов, из которых можно этот коктейль приготовить.\
Как собираюсь реализовать: собрать датасет из фото коктейлей в google (выполнено, в очищенном датасете около 17000 фото 31 коктейля), для каждого коктейля будет определён "вектор рецепта", в котором 0 соответствует отсутствию ингридиента в коктейле, 1 - присутствию ингридиента, длина "вектора рецепта" равна числу возможных ингридиентов (кофе, молоко, текила и пр.). Далее планируется выполнить трансферное обучение EfficientNetV2-S, в качестве таргета - "вектор рецепта" для каждого изображения.

**2.** ***Создание простейшего интернет-сайта*** с функционалом загрузки изображения (непосредственно файлом или по ссылке), размещение на нём обученной модели. Функционал сайта на начальном этапе: предлагать перечень ингридиентов по фото.

**3.** ***Обучение детектора коктейля*** на фото. Детектор нужен для двух задач: улучшение качества классификатора (коктейль на фото может располагаться в разных местах), и получение датасета для обучения генеративных моделей (для генеративных моделей лучше, чтобы объект находился в центре кадра и занимал большую его часть).\
Как собираюсь реализовать: выполнить разметку примерно 1500 фото, обучить YoLoV5m, создать датасет с вырезанными коктейлями, обучить заново классификатор (так же, только на новом датасете), создать составную модель детектор->классификатор, разместить эту модель на сайте.

**4.** ***Обучение простой генеративной модели (CVAE)***, функционал модели - по заданным ингридиентам рисовать изображение коктейля разрешением 512x512.\
Как собираюсь реализовать:
1. Обучить Conditional Variational Autoencoder (CVAE): на входе энкодера - изображение коктейля, на выходе энкодера (после семплера) - латентный "вектор стиля" небольшой размерности (где-то 4-6), на входе декодера - латентный "вектор стиля" из энкодера и "вектор рецепта" с ингридиентами. Задача - научить генератор не просто генерировать случайные изображения, но генерировать по заданным векторам изображение определённого вида.\
Что ожидается от автоэнкодера: энкодер формирует "вектор стиля" при получении изображения, декодер по "вектору стиля" и "вектору рецепта" выдаёт размытое, но узнаваемое изображение коктейля. Распределение "векторов стиля" для каждого вида коктейля в латентном пространстве достаточно плотное, близкое к многомерному нормальному, семплирование случайных векторов из многомерного нормального распределения в этом пространстве позволяет формировать размытые, но относительно реалистичные изображения при помощи декодера.
2. Создать алгоритм генерации изображения, примерно такой: пользователь указывает, из каких ингридиентов он хочет "приготовить" коктейль, выбирает "случайные" или "ручные" настройки стиля: в случае случайных настроек, выполняется семплирование "вектора стиля" из многомерного нормального распределения, в случае выбора ручного режима, пользователь при помощи 4х-6ти "ползунков" TrackBar выбирает "вектор стиля". "Вектор стиля" вместе с "вектором рецепта" подаются в генератор, который формирует изображение коктейля, изображение показывается пользователю. Естественно, пользователь может выбрать набор ингридиентов, которого не было в обучающей выборке, для того чтобы генератор работал с такими случаями, представление векторов рецепта и стиля на входе генератора, я планирую сжать при помощи bottleneck линейного слоя с активацией.
3. Разместить генеративную модель на сайте.

**5.** ***Обучение продвинутой генеративной модели ([C-IDVAE](https://arxiv.org/pdf/1909.13062.pdf))***, функционал модели - по заданным ингридиентам рисовать изображение коктейля разрешением 256х256 (можно потом экстраполировать до 512х512).

*C-IDVAE - это гибрид CVAE и GAN, а именно, модель состоящая из двух основных блоков: энкодера=дискриминатора и декодера=генератора, где энкодер CVAE одновременно выполняет функции дискриминатора GAN, а декодер CVAE попеременно обучается как генератор GAN и как обычный декодер в автоэнкодере.*

Как собираюсь реализовать:
1. Обучить Conditional Implicit Discriminator in Variational Autoencoder (C-IDVAE): на входе энкодера=дискриминатора - изображение коктейля + "вектор рецепта" с ингридиентами, добавляется после первой свёртки, на выходе энкодера=дискриминатора после семплера - латентный "вектор стиля" небольшой размерности (где-то 4-6) + (до семплера) выход дискриминатора (для GAN Loss), на входе генератора=декодера - латентный "вектор стиля" из энкодера и "вектор рецепта" с ингридиентами. Для обучения C-IDVAE используются четыре функции потерь одновременно: BCE или MSE для дискриминатора=энкодера, reconstruction loss (MSE или MAE) для энкодера=дискриминатора и генератора=декодера, prior loss (KL divergence) для энкодера=дискриминатора и GAN loss (backprop MSE или BCE через дискриминатор) для генератора=декодера. \
Что ожидается от модели: генератор=декодер по "вектору стиля" и "вектору рецепта" выдаёт хоть и не фотореалистичное (скорее, будет в стиле не очень хорошей маслянной картины), но не размытое и хорошо узнаваемое изображение коктейля. Распределение "векторов стиля" для каждого вида коктейля в латентном пространстве достаточно плотное, близкое к многомерному нормальному, семплирование случайных векторов из многомерного нормального распределения в этом пространстве позволяет формировать относительно реалистичные изображения при помощи декодера.
2. В случае, если распределения "векторов стиля" в латентном пространстве будет сильно отличаться для различных коктейлей, необходимо обучить простую модель (скорее всего, линейную регрессию), которая по "вектору рецепта" будет предсказывать "вектор стиля" и дополнить ей алгоритм генерации изображения.
3. Разместить генеративную модель на сайте.

**6.** ***Добавление функции генерации изображения коктейля по текстовому описанию рецепта.***\
Как собираюсь реализовать:
1. Взять [датасет](https://www.kaggle.com/datasets/ai-first/cocktail-ingredients?select=all_drinks.csv) "Cocktail Ingredients" с Kaggle, в котором, помимо прочего, есть колонки "Наименование коктейля", "Текстовый рецепт", "Ингридиенты".
2. Обучить классификатор на основе FastText определять набор ингридиентов = "вектор рецепта" (можно даже ArgMax не делать, так и оставить вероятности).
3. Добавить модель на сайт.
4. Брать сжатое представление рецепта (например, размерности 8), которое формирует FastText, и использовать это представление в качестве "вектора рецепта" во всех моделях, описанных выше. Такое снижение размерности позволит улучшить представление данных для наборов ингридиентов, изображений которых нет в датасете.
5. Обновить модели на сайте.

**7.** ***Добавление переводчика с русского на английский***, чтобы русскоязычные описания коктейлей тоже понимал.



