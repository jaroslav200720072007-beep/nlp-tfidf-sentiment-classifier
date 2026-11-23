# Класифікація відгуків: NLP + TF-IDF + Logistic Regression

Навчальний проєкт: бінарна класифікація тональності відгуків (позитивний / негативний) за допомогою Python і scikit-learn.

## Задача

За текстом відгуку передбачити його тональність:

- `1` — позитивний
- `0` — негативний

## Дані

Файл `synthetic_reviews.csv` — синтетичний датасет, згенерований скриптом із шаблонних фраз.

| Параметр | Значення |
|----------|----------|
| Кількість рядків | 600 |
| Колонки | `text`, `label` |
| Баланс класів | 300 позитивних / 300 негативних |
| Унікальність | усі 600 текстів унікальні |
| Складні випадки | 60 неоднозначних відгуків, наприклад «… was terrible, but overall highly recommend it» |

Частина текстів містить шум: нижній регістр, зайві знаки оклику.

## Підхід

1. Поділ даних: 70% навчання / 30% тест (`train_test_split`, `stratify=y`, `random_state=42`).
2. Векторизація: `TfidfVectorizer(lowercase=True, ngram_range=(1, 2))`, тобто уніграми та біграми. Векторизатор навчається лише на train.
3. Модель: `LogisticRegression(random_state=42)`.
4. Метрики: Accuracy та F1-score на тестовій вибірці.

## Результати

| Метрика | Значення |
|---------|----------|
| Accuracy | 0.9944 |
| F1-score | 0.9945 |

Тестова вибірка містить 180 текстів, тобто модель помилилась в одному з них. Жоден тестовий текст не повторюється в навчальній вибірці.

## Важливий урок: витік даних (data leakage)

Перша версія датасету містила 500 рядків, але лише близько 70 унікальних текстів, решта були копіями. Звичайний `train_test_split` ділить за рядками, тому однакові речення потрапили і в train, і в test. Результат: Accuracy = 1.0000 і F1 = 1.0000, що було наслідком запам'ятовування, а не якості моделі.

Що зроблено:

- датасет згенеровано заново так, щоб усі тексти були унікальними;
- перевірено, що частка тестових текстів, які є в train, дорівнює 0.0.

Якщо дані містять дублікати, потрібно дедуплікувати їх або ділити за групами (`GroupShuffleSplit`, `GroupKFold`).

## Обмеження

- Дані синтетичні й складаються з обмеженого набору фраз. Ключові слова («terrible», «excellent») майже напряму визначають клас, тому метрики близькі до ідеальних.
- Метрики на реальних відгуках, швидше за все, будуть нижчими.
- Тестова вибірка мала (180 текстів), тому одна помилка помітно змінює результат.

## Запуск

```bash
pip install pandas scikit-learn
```

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

data = pd.read_csv("synthetic_reviews.csv")
X, y = data["text"], data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(random_state=42)
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
```

## Можливі покращення

- Додати `classification_report` і `confusion_matrix`.
- Проаналізувати помилки моделі та найважливіші слова (коефіцієнти логістичної регресії).
- Використати крос-валідацію (`StratifiedKFold` або `GroupKFold`).
- Підібрати параметри через `GridSearchCV` (`ngram_range`, `C`) і порівняти з `LinearSVC`, `MultinomialNB`.
- Перевірити на реальному датасеті (IMDB, Amazon, Yelp reviews).
- Спробувати векторні представлення (embeddings), наприклад `sentence-transformers`.
