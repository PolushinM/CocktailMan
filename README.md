## CocktailMan

[![Test](https://github.com/PolushinM/CocktailMan/actions/workflows/python-app.yml/badge.svg)](https://github.com/PolushinM/CocktailMan/actions/workflows/python-app.yml)
[![Pylint](https://github.com/PolushinM/CocktailMan/actions/workflows/pylint.yml/badge.svg)](https://github.com/PolushinM/CocktailMan/actions/workflows/pylint.yml)
[![Docker Image CI](https://github.com/PolushinM/CocktailMan/actions/workflows/docker-image.yml/badge.svg)](https://github.com/PolushinM/CocktailMan/actions/workflows/docker-image.yml)

[Интернет-сайт](http://185.117.72.245/), на котором работают три обученные мной нейросетевые модели:
1. Классификатор, который по фото коктейля подсказывает ингредиенты, из которых его можно приготовить. (Уже работает)
2. Детектор коктейля, который находит коктейль на фото и позволяет сделать красивое размытие вокруг коктейля на фото, а так же облегчает работу первой модели и готовит данные для третьей модели.
3. Генератор text2image, который принимает текст рецепта и рисует изображение коктейля. (В разработке)

### TODO:

**[Issue #1 (SOLVED)](https://github.com/PolushinM/CocktailMan/issues/1)** ***Сбор изображений и очистка датасета.***  Собрать 10000..30000 изображений, приблизительно, 30 напитков из Google-pictures.

**[Issue #2 (SOLVED)](https://github.com/PolushinM/CocktailMan/issues/2)** ***Обучение модели*** (multilabel classifier), которая по изображению коктейля может предложить набор ингридиентов, из которых можно этот коктейль приготовить.\
Как собираюсь реализовать: собрать датасет из фото коктейлей в google (выполнено, в очищенном датасете около 17000 фото 31 коктейля, 40 ингридиентов), для каждого коктейля будет определён "вектор рецепта", в котором 0 соответствует отсутствию ингридиента в коктейле, 1 - присутствию ингридиента, длина "вектора рецепта" равна числу возможных ингридиентов (кофе, молоко, текила и пр.). Далее планируется выполнить трансферное обучение EfficientNetV2-S (tiny), в качестве таргета - "вектор рецепта" для каждого изображения.

**[Issue #3 (SOLVED)](https://github.com/PolushinM/CocktailMan/issues/3)** ***Создание простейшего интернет-сайта***  с функционалом загрузки изображения (непосредственно файлом или по ссылке), размещение на нём обученной модели. Функционал сайта на начальном этапе: предлагать перечень ингридиентов по фото. Фото должно загружаться файлом или по ссылке.\
Планируется использовать фреймворк Flask.

**[Issue #4 (SOLVED)](https://github.com/PolushinM/CocktailMan/issues/4)** ***Разметка изображений для обучения детектора коктейля на фото.***  Разметить 1500..2000 изображений при помощи программы [LabelImg](https://vk.com/away.php?to=https%3A%2F%2Fgithub.com%2Ftzutalin%2FlabelImg&cc_key=).

**[Issue #5 (SOLVED)](https://github.com/PolushinM/CocktailMan/issues/5)** ***Обучение детектора коктейля на фото.*** Детектор нужен для двух задач: улучшение качества классификатора (коктейль на фото может располагаться в разных местах), и получение датасета для обучения генеративных моделей (для генеративных моделей лучше, чтобы объект находился в центре кадра и занимал большую его часть).\
Как собираюсь реализовать: выполнить разметку примерно 1500 фото, обучить YoLoV5m, создать датасет с вырезанными коктейлями, обучить заново классификатор (так же, только на новом датасете), создать составную модель детектор->классификатор, разместить эту модель на сайте.

**[Issue #6 (SOLVED)](https://github.com/PolushinM/CocktailMan/issues/6)** ***Обучение простой генеративной модели (CVAE)***, функционал модели - по заданным ингридиентам рисовать изображение коктейля разрешением 512x512.\
Как собираюсь реализовать:
1. Обучить Conditional Variational Autoencoder (CVAE): на входе энкодера - изображение коктейля, на выходе энкодера (после семплера) - латентный "вектор стиля" небольшой размерности (где-то 4-6), на входе декодера - латентный "вектор стиля" из энкодера и "вектор рецепта" с ингридиентами. Задача - научить генератор не просто генерировать случайные изображения, но генерировать по заданным векторам изображение определённого вида.\
Что ожидается от автоэнкодера: энкодер формирует "вектор стиля" при получении изображения, декодер по "вектору стиля" и "вектору рецепта" выдаёт размытое, но узнаваемое изображение коктейля. Распределение "векторов стиля" для каждого вида коктейля в латентном пространстве достаточно плотное, близкое к многомерному нормальному, семплирование случайных векторов из многомерного нормального распределения в этом пространстве позволяет формировать размытые, но относительно реалистичные изображения при помощи декодера.
2. Создать алгоритм генерации изображения, примерно такой: пользователь указывает, из каких ингридиентов он хочет "приготовить" коктейль, выбирает "случайные" или "ручные" настройки стиля: в случае случайных настроек, выполняется семплирование "вектора стиля" из многомерного нормального распределения, в случае выбора ручного режима, пользователь при помощи 4х-6ти "ползунков" TrackBar выбирает "вектор стиля". "Вектор стиля" вместе с "вектором рецепта" подаются в генератор, который формирует изображение коктейля, изображение показывается пользователю. Естественно, пользователь может выбрать набор ингридиентов, которого не было в обучающей выборке, для того чтобы генератор работал с такими случаями, представление векторов рецепта и стиля на входе генератора, я планирую сжать при помощи bottleneck линейного слоя с активацией.
3. Разместить генеративную модель на сайте.

**[Issue #7](https://github.com/PolushinM/CocktailMan/issues/7)** ***Добавление функции генерации изображения коктейля по текстовому описанию рецепта.***\
Как собираюсь реализовать:
1. Взять [датасет](https://www.kaggle.com/datasets/ai-first/cocktail-ingredients?select=all_drinks.csv) "Cocktail Ingredients" с Kaggle, в котором, помимо прочего, есть колонки "Наименование коктейля", "Текстовый рецепт", "Ингридиенты".
2. Обучить классификатор на основе FastText определять набор ингридиентов = "вектор рецепта" (можно даже ArgMax не делать, так и оставить вероятности). Скорее всего, FastText деплоить не стоит, нужно выгрузить из него словарь, и считать эмбеддинги в numpy.
3. Добавить модель на сайт.
4. Брать сжатое представление рецепта (например, размерности 8), которое формирует FastText, и использовать это представление в качестве "вектора рецепта" в C-IDVAE. Такое снижение размерности позволит улучшить представление данных для наборов ингридиентов, изображений которых нет в датасете.
5. Обновить модели на сайте.

**[Issue #8 (SOLVED)](https://github.com/PolushinM/CocktailMan/issues/8)** ***Обучение продвинутой генеративной модели ([C-IDVAE](https://arxiv.org/pdf/1909.13062.pdf))***, функционал модели - по заданным ингридиентам рисовать изображение коктейля разрешением 256х256 (можно потом экстраполировать до 512х512).

*C-IDVAE (Conditional Implicit Discriminator in Variational Autoencoder) - это гибрид CVAE и GAN, а именно, модель состоящая из двух основных блоков: энкодера=дискриминатора и декодера=генератора, где энкодер CVAE одновременно выполняет функции дискриминатора GAN, а декодер CVAE попеременно обучается как генератор GAN и как обычный декодер в автоэнкодере.*

Как собираюсь реализовать:
1. Обучить C-IDVAE: на входе энкодера=дискриминатора - изображение коктейля + "вектор рецепта" с ингридиентами, добавляется после первой свёртки, на выходе энкодера=дискриминатора после семплера - латентный "вектор стиля" небольшой размерности (где-то 4-6) + (до семплера) выход дискриминатора (для GAN Loss), на входе генератора=декодера - латентный "вектор стиля" из энкодера и "вектор рецепта" с ингридиентами. Для обучения C-IDVAE используются четыре функции потерь одновременно: BCE или MSE для дискриминатора=энкодера, reconstruction loss (MSE или MAE) для энкодера=дискриминатора и генератора=декодера, prior loss (KL divergence) для энкодера=дискриминатора и GAN loss (backprop MSE или BCE через дискриминатор) для генератора=декодера. \
Что ожидается от модели: генератор=декодер по "вектору стиля" и "вектору рецепта" выдаёт хоть и не фотореалистичное (скорее, будет в стиле не очень хорошей маслянной картины), но не размытое и хорошо узнаваемое изображение коктейля. Распределение "векторов стиля" для каждого вида коктейля в латентном пространстве достаточно плотное, близкое к многомерному нормальному, семплирование случайных векторов из многомерного нормального распределения в этом пространстве позволяет формировать относительно реалистичные изображения при помощи декодера.
2. В случае, если распределения "векторов стиля" в латентном пространстве будет сильно отличаться для различных коктейлей, необходимо обучить простую модель (скорее всего, линейную регрессию), которая по "вектору рецепта" будет предсказывать "вектор стиля" и дополнить ей алгоритм генерации изображения.
3. Разместить генеративную модель на сайте.

**[Issue #9](https://github.com/PolushinM/CocktailMan/issues/7)** ***Добавление русского языка***, чтобы русскоязычные описания коктейлей тоже понимал. Нужно перевести все рецепты датасета автоматическим переводчиком на русский язык, добавить их в обучающую выборку, обучить заново модель FastText.

**[Issue #10](https://github.com/PolushinM/CocktailMan/issues/10)** ***Обучение (finetune / adapter) VQGAN.*** План: Используя трансферное обучение и распространенный автоэнкодер, разработать и обучить вариационный автоэнкодер для генерации изображений коктейлей. Скорее всего, будет достаточно обучения небольшого автоэнкодера, работающего в латентном пространстве VQGAN, который используется в Stable Diffusion. Можно попробовать также использовать для этого CGAN.

**[Issue #11](https://github.com/PolushinM/CocktailMan/issues/11)** ***Дистилляция Stable Diffusion.*** План: Улучшить качество генеративной модели путем генерирования обучающих данных моделью Stable Diffusion. Можно генерировать изображения непосредственно, либо тензоры латентного пространства LDM.
