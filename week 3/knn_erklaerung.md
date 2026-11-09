# kNN einfach erklärt

Diese Erklärung gehört zu `knn_schritt_fuer_schritt.ipynb`. Lies sie am besten nebenbei: Die Abschnitte hier heißen genauso wie die Schritte im Notebook. In Klammern steht jeweils, auf welchen Folien von `03_DSAI` das Thema vorkommt.

---

## Die Idee in drei Sätzen

Wir wollen vorhersagen, welche Personen ihren Job wechseln wollen. kNN schaut sich dafür an, welche bekannten Personen einer neuen Person am ähnlichsten sind. Wollen die meisten dieser ähnlichen Personen wechseln, sagt das Modell auch für die neue Person „Wechsel“ voraus.

**k** ist einfach die Anzahl der Nachbarn, die gefragt werden. **kNN** heißt „k-nearest neighbors“, also „die k nächsten Nachbarn“.

**Vergleich aus dem Alltag:** Du willst wissen, ob dir ein Film gefällt. Du fragst die fünf Freunde, die den gleichen Geschmack haben wie du. Vier fanden ihn gut, also wird er dir wahrscheinlich gefallen. Das ist kNN mit k = 5.

---

## Wo wir im Projekt stehen

Das kNN-Notebook ist das Ende einer Kette. Jede Datei gibt ihr Ergebnis an die nächste weiter:

```text
customer_train.csv                        Rohdaten: 19.158 Personen, 14 Spalten
        │
        ▼
week 2/fix_data.ipynb                     richtige Datentypen, Datenprüfung
        │  speichert customer_train_typed.pkl
        ▼
week 2/aq_analysis.ipynb                  erste Zusammenhänge (Cramér's V)
week 2/missing_company.ipynb              Warum fehlen die Firmendaten?
week 3/missing_values_overview.ipynb      Warum fehlen die anderen Werte?
        │  Diese Erkenntnisse entscheiden, wie gefüllt wird
        ▼
week 3/missing_values_imputation.ipynb    Lücken füllen, Train und Test trennen
        │  speichert customer_train_final_train.pkl und ..._test.pkl
        ▼
week 3/knn_schritt_fuer_schritt.ipynb     das Modell (dieses Dokument)
```

`knn_methode.ipynb` war dein erster Versuch. Er füllt die Lücken noch innerhalb der Pipeline und codiert die Spalten etwas anders. Das neue Notebook nutzt stattdessen die fertigen Dateien aus dem Imputation-Notebook.

---

## Schritt 0: Setup

Hier werden die Werkzeuge geladen, zum Beispiel:

```python
from sklearn.neighbors import KNeighborsClassifier
RANDOM_STATE = 42
```

`RANDOM_STATE = 42` legt den Zufall fest. Daten werden öfter zufällig gemischt, und ohne festen Wert käme jedes Mal ein etwas anderes Ergebnis heraus. Die Zahl 42 hat keine Bedeutung, wichtig ist nur, dass sie gleich bleibt.

---

## Schritt 1: Was genau wollen wir vorhersagen? (Folien 6, 7, 78)

**Supervised Learning** heißt: Wir kennen bei den Trainingsdaten die richtige Antwort, und das Modell lernt daraus.

| Begriff | Bedeutung | Bei uns |
|---|---|---|
| Feature (X) | die Information, die das Modell nutzen darf | zum Beispiel Trainingsstunden |
| Target (y) | das, was vorhergesagt werden soll | `job_change` |
| Klassifikation | Das Ziel sind Kategorien | Wechsel ja oder nein |
| Regression | Das Ziel ist eine Zahl | nicht bei uns |

Drei Spalten nutzen wir bewusst nicht:

- `student_id`: Eine Nummer sagt nichts über eine Person aus.
- `city`: 123 Städte wären zu viele Spalten, und `city_development_index` beschreibt die Stadt schon.
- `gender`: Das Geschlecht soll aus Fairness-Gründen nicht in die Vorhersage einfließen. Das wurde schon in `missing_values_overview.ipynb` entschieden.

---

## Schritt 2: Daten laden

```python
train = pd.read_pickle("customer_train_final_train.pkl")
test  = pd.read_pickle("customer_train_final_test.pkl")
```

Wir laden zwei Dateien: **Train** mit 15.326 Personen zum Lernen und **Test** mit 3.832 Personen für die Abschlussprüfung.

**Warum zwei Dateien?** Wie bei einer Klausur: Wer die Fragen vorher kennt, schneidet gut ab, hat aber vielleicht nichts verstanden. Das Testset sind die Klausurfragen, die das Modell nie gesehen hat. Die Trennung wurde schon im Imputation-Notebook gemacht, darum teilen wir hier nicht noch einmal.

**Danach werden Features und Ziel getrennt:**

```python
X = df.drop(columns=["student_id", "city", "job_change"]).copy()
y = df["job_change"].astype(int)
```

- `X` ist die Tabelle mit allem, was das Modell sehen darf.
- `y` ist die Antwortspalte. `astype(int)` macht aus True und False die Zahlen 1 und 0.

Das Ziel muss aus `X` heraus, sonst könnte das Modell die Antwort einfach abschreiben.

---

## Schritt 3: Baseline, die Messlatte (Folien 40, 41)

Bevor wir ein Modell bewerten, fragen wir: Was erreicht man ganz ohne Modell?

75,1 % der Personen wollen nicht wechseln. Wer also immer „kein Wechsel“ sagt, liegt in 75,1 % der Fälle richtig. Trotzdem findet diese Regel keine einzige wechselwillige Person. Darum reicht die **Accuracy** (Anteil richtiger Antworten) allein nicht.

```python
baseline = DummyClassifier(strategy="most_frequent")
```

`DummyClassifier` ist so ein Modell ohne Wissen. `most_frequent` heißt: Sag immer die häufigste Antwort.

### Was ist der AUC? (Folien 68, 69)

Der AUC prüft: Bekommen wechselwillige Personen höhere Risikowerte als die anderen? Beispiel mit vier Personen:

| Person | Will wechseln? | Risikowert |
|---|---|---|
| A | ja | 70 % |
| B | ja | 30 % |
| C | nein | 40 % |
| D | nein | 10 % |

Wir vergleichen jede Person, die wechselt, mit jeder Person, die nicht wechselt:

| Paar | Richtig sortiert? |
|---|---|
| A gegen C: 70 > 40 | ja |
| A gegen D: 70 > 10 | ja |
| B gegen C: 30 < 40 | nein |
| B gegen D: 30 > 10 | ja |

3 von 4 Paaren stimmen, der AUC ist also **0,75**.

- **1,0** bedeutet perfekt sortiert.
- **0,5** bedeutet nicht besser als Raten. Die Baseline landet genau dort, weil sie allen denselben Wert gibt.

---

## Schritt 4: Daten für kNN vorbereiten (Folie 78: „with scaled features“)

kNN misst Abstände zwischen Personen. Daraus ergeben sich zwei Probleme.

### Problem 1: Unterschiedliche Größenordnungen

Zwei Personen unterscheiden sich beim Index um 0,3 (viel) und bei den Trainingsstunden um 10 (wenig). Die Distanz ist dann:

```text
√(0,3² + 10²) = √(0,09 + 100) ≈ 10,004
```

Der Index ist fast egal, nur weil seine Zahlen klein sind. Deshalb skalieren wir die Zahlenspalten und die Rangzahlen mit dem **StandardScaler**:

```text
neuer Wert = (Wert − Mittelwert) / Standardabweichung
```

Beispiel mit echten Werten aus dem Datensatz:

| | Wert | Mittelwert | Standardabweichung | skaliert |
|---|---|---|---|---|
| Trainingsstunden | 90 | 65,3 | 60,0 | (90 − 65,3) / 60,0 ≈ **0,41** |
| Index | 0,62 | 0,83 | 0,123 | (0,62 − 0,83) / 0,123 ≈ **−1,7** |

Jetzt sind beide Spalten vergleichbar: 0 ist der Durchschnitt, und −1,7 heißt „1,7 typische Abstände unter dem Durchschnitt“.

### Problem 2: Text kann man nicht subtrahieren

Text muss zu Zahlen werden. Wir nutzen dafür zwei Wege, je nachdem, ob eine Spalte eine natürliche Reihenfolge hat. Das nennen wir **gemischtes Encoding**.

**Mit Reihenfolge: ordinal.** Die Stufen werden durchnummeriert. Das gilt für `education_level`, `experience` und `last_new_job`.

| education_level | als Zahl |
|---|---|
| Primary School | 0 |
| High School | 1 |
| Graduate | 2 |
| Masters | 3 |
| Phd | 4 |

Graduate liegt dann wirklich näher an Masters als an Primary School.

**Ohne Reihenfolge: one-hot.** Jede Kategorie bekommt eine eigene Ja/Nein-Spalte. Das gilt für die übrigen Kategorien, zum Beispiel `company_type`.

| company_type | Spalte Pvt Ltd | Spalte NGO | Spalte Startup |
|---|---|---|---|
| Pvt Ltd | 1 | 0 | 0 |
| NGO | 0 | 1 | 0 |

Warum nicht auch hier einfach Pvt Ltd = 1, NGO = 2 und Startup = 3? Dann wäre Startup „weiter weg“ von Pvt Ltd als NGO. Diese Abstände wären frei erfunden.

Aus unseren 10 Features werden so 32 Spalten.

### Die Pipeline: alles an einem Stück

```python
def make_knn(...):
    return Pipeline([
        ("pre", make_preprocessor(...)),   # 1. Text zu Zahlen, skalieren
        ("knn", KNeighborsClassifier(...)) # 2. Modell
    ])
```

`Pipeline` ist ein Fließband: Die Daten laufen immer zuerst durch die Vorbereitung und dann ins Modell. `ColumnTransformer` verteilt dabei die Spalten auf die richtigen Stationen.

**Warum ist das wichtig?** Der Scaler lernt Mittelwert und Standardabweichung aus den Daten. Würde er auch die Prüfdaten sehen, flössen Informationen aus der Prüfung ins Lernen. Das heißt **Data Leakage**. Die Pipeline verhindert das, und auch im Imputation-Notebook wurde deshalb erst getrennt und dann gefüllt.

---

## Schritt 5: kNN von Hand (Folien 9 bis 11)

Das ist der wichtigste Abschnitt. kNN besteht aus vier Teilen:

1. Nimm eine neue Person.
2. Berechne den Abstand zu allen bekannten Personen.
3. Suche die k nächsten.
4. Lass sie abstimmen.

### So wird der Abstand berechnet

Vereinfacht mit drei Personen und bereits vorbereiteten Zahlen:

| Person | Index | Stunden | Pvt Ltd | NGO |
|---|---|---|---|---|
| A (neu) | −1,7 | 0,4 | 1 | 0 |
| B | −1,7 | 0,1 | 1 | 0 |
| C | 0,7 | 0,4 | 0 | 1 |

Die **euklidische Distanz** ist die Luftlinie: Differenz je Spalte bilden, quadrieren, alles addieren, Wurzel ziehen.

```text
A zu B: √(0² + 0,3² + 0² + 0²)       = √0,09 = 0,3
A zu C: √(2,4² + 0² + 1² + 1²)       = √7,76 ≈ 2,79
```

B ist A also viel ähnlicher als C.

### Ergebnis im Notebook

Die erste Person im Training spielt die neue Person. Ihre fünf Nachbarn haben fast das gleiche Profil, und von ihnen wollen vier wechseln:

| k | Anteil, der wechseln will | Vorhersage |
|---|---|---|
| 1 | 100 % | Wechsel |
| 3 | 100 % | Wechsel |
| 5 | 80 % | Wechsel |
| 15 | 60 % | Wechsel |
| 75 | 48 % | kein Wechsel |

**Dieselbe Person bekommt je nach k eine andere Antwort.** Darum müssen wir k sorgfältig wählen.

Der Code dazu:

```python
distances = euclidean_distances(pre.transform(probe), pre.transform(rest)).ravel()
nearest = np.argsort(distances, kind="stable")
share = y_rest[nearest[:k]].mean()
```

- `euclidean_distances` berechnet den Abstand der neuen Person zu allen anderen.
- `argsort` sortiert die Personen vom nächsten zum fernsten.
- `nearest[:k]` nimmt die ersten k.
- `.mean()` bildet den Mittelwert der Antworten. Bei Nullen und Einsen ist das genau der Anteil der Einsen.

### 5b: Die scikit-learn-Grammatik (Folie 14)

Jedes Modell funktioniert mit denselben drei Befehlen:

```python
knn = KNeighborsClassifier(n_neighbors=15)   # 1. erstellen
knn.fit(X, y)                                # 2. lernen
knn.predict(X_new)                           # 3. vorhersagen
```

Bei kNN speichert `fit` nur die Daten. Gerechnet wird erst bei `predict`.

Wichtig: `predict` gibt die Entscheidung aus (0 oder 1), `predict_proba` den Anteil der Nachbarn. Das Modell in 5b nutzt nur zwei Features und ist bloß ein Beispiel. Das endgültige Modell nutzt alle 10.

### 5c: Die Entscheidungsgrenze (Folie 13)

Mit zwei Features kann man kNN zeichnen. Die Hintergrundfarbe zeigt die Vorhersage, die Punktfarbe die Wahrheit. Ein roter Punkt auf blauer Fläche ist also eine übersehene Person.

---

## Schritt 6: Warum wir nicht auf den Trainingsdaten messen (Folien 16 bis 20)

Mit k = 1 erreicht das Modell auf den Trainingsdaten 99,8 %, auf neuen Daten aber nur 71,6 %. Das liegt sogar unter der Baseline.

**Warum?** Fragt man das Modell nach einer Person aus dem Training, findet es diese Person selbst als nächsten Nachbarn und schreibt ihre Antwort ab. Das ist **Auswendiglernen**.

### 6b: Holdout-Split

```python
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.3, random_state=21, stratify=y_train)
```

- `test_size=0.3`: 30 % werden zum Prüfen zurückgehalten.
- `stratify=y_train`: Der Anteil von 25 % Wechslern bleibt in beiden Teilen gleich.
- Geteilt werden nur die Trainingsdaten. Das echte Testset bleibt verschlossen.

Das Ergebnis bei k = 6 sind 76,6 % Accuracy.

---

## Schritt 7: Das beste k finden (Folien 21 bis 23, 31 bis 36)

`k` ist die wichtigste Einstellung von kNN. Um das beste k zu finden, brauchen wir für jedes k eine verlässliche Note. **Dafür nutzen wir die Cross-Validation.** Sie wählt k nicht selbst aus, sie liefert nur die Noten.

Warum messen wir nicht einfacher?

| Messen auf ... | Problem |
|---|---|
| Trainingsdaten | Das Modell kennt die Personen schon. k = 1 sähe mit einem AUC von 0,998 perfekt aus |
| Testset | Wir würden k passend zur Abschlussprüfung aussuchen, und sie wäre nicht mehr unabhängig |
| einem einzelnen Holdout-Split | Das Ergebnis hängt davon ab, wer zufällig im Prüfteil gelandet ist |

### Wie die Cross-Validation funktioniert

Ein einzelner Split ist wie eine einzige Klausur: Man kann Glück oder Pech mit den Fragen haben. Bei der **5-fachen Cross-Validation** gibt es deshalb fünf Prüfungen.

```text
Runde 1:  [PRÜFEN] [lernen] [lernen] [lernen] [lernen]
Runde 2:  [lernen] [PRÜFEN] [lernen] [lernen] [lernen]
Runde 3:  [lernen] [lernen] [PRÜFEN] [lernen] [lernen]
Runde 4:  [lernen] [lernen] [lernen] [PRÜFEN] [lernen]
Runde 5:  [lernen] [lernen] [lernen] [lernen] [PRÜFEN]
```

Unsere 15.326 Personen werden in fünf Teile mit je rund 3.065 Personen geschnitten. Jeder Teil wird einmal geprüft, die anderen rund 12.261 Personen dienen jeweils zum Lernen.

Wir berichten dann zwei Zahlen:

- **Mittelwert**: die typische Leistung.
- **Standardabweichung (±)**: wie stark das Ergebnis schwankt.

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
auc = cross_validate(model, X_train, y_train, cv=cv, scoring="roc_auc")
```

### Warum sind sehr kleine k schlecht?

kNN schätzt einen Anteil: „Wie viele Personen wie diese wechseln?“ Mit wenigen Nachbarn ist das wie eine Umfrage unter fünf Leuten, das Ergebnis schwankt stark. Die Schwankung lässt sich ausrechnen, bei einem wahren Anteil von 25 %:

```text
k = 5:    √(0,25 · 0,75 / 5)    ≈ 0,19    also ±19 Prozentpunkte
k = 101:  √(0,25 · 0,75 / 101)  ≈ 0,043   also ±4,3 Prozentpunkte
```

### Overfitting und Underfitting

| | Kleines k | Großes k |
|---|---|---|
| Modell | kompliziert | einfach |
| Problem | **Overfitting**: lernt auch Zufälle | **Underfitting**: wird zu grob |
| Bild | viele kleine Inseln | glatte Fläche |
| Alltag | Du glaubst der Meinung eines einzigen Freundes | Du fragst die ganze Stadt |

### Die Kurven lesen

Die blaue Linie zeigt die Leistung auf bekannten Daten, die rote auf unbekannten. Nur die rote zählt. Ein großer Abstand zwischen beiden bedeutet Auswendiglernen.

Die Validierungsspalte ist die Note aus der Cross-Validation:

| k | AUC Training | AUC Validierung | Abstand |
|---|---|---|---|
| 1 | 0,998 | 0,623 | 0,375 |
| 15 | 0,837 | 0,775 | 0,062 |
| 101 | 0,799 | 0,786 | 0,013 |

Ab etwa k = 75 ändert sich kaum noch etwas.

**7b** rechnet dieselbe Idee mit einer einfachen Schleife wie bei DataCamp nach. Dort liegt das Maximum bei k = 21, das übernehmen wir aber nicht: Es beruht auf nur einem Split und die Kurve ist dort praktisch flach.

---

## Schritt 8: Die beste Einstellung suchen (baut auf den Folien 31 bis 36 auf)

`GridSearchCV` probiert alle Kombinationen automatisch aus:

```python
param_grid = {
    "knn__n_neighbors": [15, 25, 41, 75, 101, 151, 201, 301, 401],
    "knn__weights": ["uniform", "distance"],
    "knn__p": [1, 2],
}
```

- `knn__n_neighbors` heißt: der Parameter `n_neighbors` im Pipeline-Schritt `knn`.
- `weights`: Bei `uniform` zählt jede Stimme gleich, bei `distance` zählen nähere Nachbarn mehr.
- `p`: Bei `2` wird die Luftlinie gemessen, bei `1` die Summe der Einzelabstände (Manhattan-Distanz). Von A zu C aus Schritt 5 wären das 2,4 + 0 + 1 + 1 = 4,4.

Das sind 9 · 2 · 2 = 36 Kombinationen, jede in 5 Folds, also 180 Trainingsläufe. Warum nur ungerade k? So kann es bei zwei Klassen kein Unentschieden geben.

**Ergebnis:** k = 101, `uniform`, `p = 1`, AUC 0,788 ± 0,009. Das ist auch unsere Validierung: Der Wert sagt, wie gut das Modell voraussichtlich bei neuen Personen ist und wie stark das Ergebnis schwankt.

Die besten Kombinationen liegen extrem nah beieinander. Die genaue Wahl ist also nicht kritisch.

### 8b: Hilft jedes Feature?

Wir lassen jedes Feature einmal weg und schauen, was mit dem AUC passiert. Sinkt er, hat das Feature geholfen.

| Weggelassen | AUC-Änderung |
|---|---|
| `city_development_index` | **−0,072** |
| alle anderen | zwischen −0,006 und +0,001 |

Ohne den Index fällt der AUC von 0,788 auf etwa 0,716, er ist also mit Abstand am wichtigsten. Kein Feature erfüllt unsere vorab festgelegte Regel zum Entfernen, darum bleiben alle 10 im Modell.

**Überleitung:** In `aq_analysis.ipynb` hast du mit Cramér's V gemessen, wie stark einzelne Merkmale mit `job_change` zusammenhängen. Dort wird jedes Merkmal für sich betrachtet, hier prüfen wir, was einem fertigen Modell fehlt, wenn ein Merkmal wegfällt.

---

## Schritt 9: Die beste Schwelle aus Kosten (Folien 58 bis 62, 70, 71)

Das Modell liefert keinen festen Wert „Wechsel“, sondern einen Anteil. Die **Schwelle** legt fest, ab wann jemand als „Wechsel“ gilt:

| Anteil wechselnder Nachbarn | Schwelle 0,50 | Schwelle 0,32 |
|---|---|---|
| 60 % | Wechsel | Wechsel |
| 40 % | kein Wechsel | **Wechsel** |
| 20 % | kein Wechsel | kein Wechsel |

Mit niedrigerer Schwelle markieren wir mehr Personen. Wir finden dadurch mehr Wechselwillige, haben aber auch mehr Fehlalarme.

### Wo die Kosten ins Spiel kommen

Wir treffen eine **Annahme**, die nicht aus den Daten stammt:

| Fehler | Was passiert | Kosten |
|---|---|---|
| False Positive | jemand wird angesprochen, will aber nicht wechseln | 100 € |
| False Negative | eine wechselwillige Person wird übersehen | 300 € |

### Rechnung für die Theorie

Eine Person hat die Wechselwahrscheinlichkeit p.

```text
Wenn wir sie ansprechen:           erwartete Kosten = (1 − p) · 100
Wenn wir sie nicht ansprechen:     erwartete Kosten = p · 300

Ansprechen lohnt sich, wenn:       p · 300 > (1 − p) · 100
Daraus folgt:                      p > 100 / (100 + 300) = 0,25
```

Auf den Daten ist 0,32 am günstigsten. Der Unterschied kommt daher, dass die Nachbarschaftsanteile von kNN keine exakten Wahrscheinlichkeiten sind.

### Wie der Code die Schwelle findet

```python
oof_proba = cross_val_predict(best_model, X_train, y_train, cv=cv,
                              method="predict_proba")[:, 1]
```

Jede Person bekommt ihren Wert von einem Modell, das sie nicht gesehen hat. `[:, 1]` nimmt die Spalte für die Klasse „Wechsel“.

Dann probiert der Code alle Schwellen von 0,02 bis 0,98 aus, berechnet jeweils die Kosten und nimmt die günstigste.

---

## Schritt 10: Die Abschlussprüfung (Folien 42 bis 52, 63 bis 69)

Jetzt ist alles festgelegt, und das Testset wird genau einmal benutzt.

### Confusion Matrix bei Schwelle 0,50

| | Vorhersage: kein Wechsel | Vorhersage: Wechsel |
|---|---|---|
| **Wahrheit: kein Wechsel** | 2.563 (True Negative) | 314 (False Positive) |
| **Wahrheit: Wechsel** | 524 (False Negative) | 431 (True Positive) |

Daraus berechnen sich alle Kennzahlen:

| Kennzahl | Frage | Rechnung | Ergebnis |
|---|---|---|---|
| Accuracy | Wie viel ist insgesamt richtig? | (2.563 + 431) / 3.832 | 78,1 % |
| Precision | Von den Markierten: Wie viele wollen wirklich wechseln? | 431 / (431 + 314) | 57,9 % |
| Recall | Von allen Wechselwilligen: Wie viele finden wir? | 431 / (431 + 524) | 45,1 % |
| F1 | Mittel aus Precision und Recall | 2 · 0,579 · 0,451 / (0,579 + 0,451) | 0,51 |

### Vergleich der drei Regeln

| Regel | Accuracy | Precision | Recall | F1 | Kosten |
|---|---|---|---|---|---|
| Baseline | 75,1 % | 0 | 0 | 0 | 286.500 € |
| kNN, Schwelle 0,50 | 78,1 % | 57,9 % | 45,1 % | 0,51 | 188.600 € |
| kNN, Schwelle 0,32 | 75,8 % | 50,9 % | 76,3 % | 0,61 | 138.100 € |

Die Kosten sind schnell nachgerechnet:

```text
Baseline:      955 übersehen · 300 €                     = 286.500 €
Schwelle 0,50: 314 · 100 € + 524 · 300 €                 = 188.600 €
Schwelle 0,32: 703 · 100 € + 226 · 300 €                 = 138.100 €
```

Die Accuracy sinkt bei 0,32 sogar leicht, trotzdem ist diese Regel für unser Ziel besser: Wir finden 729 statt 431 Wechselwillige.

### ROC-Kurve

Die ROC-Kurve zeigt alle Schwellen auf einmal. Nach oben ist gut (mehr gefunden), nach rechts ist schlecht (mehr Fehlalarme). Jede Schwelle ist ein Punkt auf der Kurve.

Der AUC von 0,799 bedeutet: In knapp 80 % der Paare aus einer wechselnden und einer nicht wechselnden Person bekommt die wechselnde den höheren Wert.

---

## Schritt 11: Was herausgekommen ist

Das endgültige Modell nutzt alle 10 Features, gemischtes Encoding, `StandardScaler`, kNN mit k = 101, `uniform`, Manhattan-Distanz und die Schwelle 0,32.

Die wichtigsten Punkte:

1. Das Modell ist besser als die Baseline.
2. Der `city_development_index` ist mit Abstand am wichtigsten.
3. Mit der niedrigeren Schwelle finden wir etwa drei Viertel der Wechselwilligen, aber rund die Hälfte der Markierten will gar nicht wechseln.
4. Die Kosten von 100 € und 300 € sind Annahmen. Für die Schwelle zählt nur ihr Verhältnis von 1 : 3.

---

## Bezug zur Präsentation am 12. November (Folie 78)

| Punkt der Folie | Wo im Notebook | Stand |
|---|---|---|
| 1. Prediction task | Schritt 1 | für kNN erledigt, „Moment“ noch genauer festlegen |
| 2. Two models | Schritte 4 bis 8 | **nur kNN, logistische Regression fehlt** |
| 3. Validation | 6b, 7 und 8 | für kNN erledigt |
| 4. Evaluation | Schritt 10 | für kNN erledigt |
| 5. Decision | Schritte 9 und 10 | für kNN erledigt |

---

## Kleines Lexikon

| Begriff | Einfach gesagt |
|---|---|
| Accuracy | Anteil aller richtigen Antworten |
| AUC | Wie gut sortiert das Modell Wechsler über Nichtwechsler? |
| Baseline | einfachste Regel als Messlatte |
| Confusion Matrix | Tabelle aus Wahrheit und Vorhersage |
| Cross-Validation | mehrere Prüfungen statt einer |
| Data Leakage | Prüfdaten beeinflussen unbemerkt das Lernen |
| Encoding | Kategorien in Zahlen umwandeln |
| F1 | ein Wert aus Precision und Recall |
| False Negative | Wechsler wird übersehen |
| False Positive | Fehlalarm |
| Feature | Eingabespalte |
| Fold | ein Teil bei der Cross-Validation |
| Holdout | zurückgehaltener Teil der Daten |
| Hyperparameter | Einstellung, die wir festlegen, zum Beispiel k |
| Overfitting | Modell lernt auswendig |
| Pipeline | Fließband aus Vorbereitung und Modell |
| Precision | Wie oft stimmt ein Alarm? |
| Recall | Wie viele Wechsler werden gefunden? |
| Skalierung | Spalten auf vergleichbare Größenordnung bringen |
| Target | Spalte, die vorhergesagt wird |
| Threshold / Schwelle | Ab welchem Anteil gilt jemand als „Wechsel“? |
| Underfitting | Modell ist zu grob |
