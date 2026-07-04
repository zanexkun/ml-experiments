# ML Experiments — What Didn't Work and Why

Projects that failed, what I found, and what I learned. The diagnostic process is usually more useful than the result.

---

## 1. App Store Games — Rating Prediction

**Goal:** Predict average user rating from game metadata (price, size, genre, age rating, in-app purchases)

**Model:** Random Forest Regressor

**Result:** R² = 0.04 — model learned almost nothing

**What I did:**
- Full preprocessing pipeline — handled missing values, encoded genres and age ratings, dropped irrelevant columns
- Trained Random Forest, got R² of 0.04
- Checked feature importances — no single feature had meaningful predictive power
- Confirmed with correlation matrix — max correlation of any feature with rating was 0.04

**Conclusion:** The dataset has no real signal. User ratings are influenced by things not in the data — game quality, marketing, timing, reviews — none of which are captured by metadata alone. The dataset itself is the problem, not the model.

**Lesson:** Always check correlations before building a model. A 0.04 correlation ceiling means no model will work, regardless of architecture or hyperparameters.

---

## 2. AI Jobs Impact Dataset (Kaggle)

**Goal:** Predict AI replacement risk from job features

**Model:** PyTorch fully connected network

**Result:** R² = -0.81 — worse than predicting the mean

**What I did:**
- Full preprocessing — ordinal encoding for education, one-hot for industry and job title
- Trained a 4-layer network for 30 epochs
- Got negative R², which means the model is actively worse than just guessing the average
- Ran correlation analysis — max correlation of any feature with AI_Replacement_Risk was 0.04
- Confirmed with scatter plots

**Conclusion:** The dataset is fake. The target variable has no real relationship with any feature. The gold badge reflects data formatting quality, not signal quality.

**Lesson:**  Correlation analysis catches fake data before you waste time training.

---

## 3. Diabetes Classification Dataset

**Goal:** Classify diabetes diagnosis from patient features

**Model:** PyTorch binary classifier, then Random Forest with cross-validation

**Result:** PyTorch — 100% accuracy. Random Forest cross-validation — 97.6%

**What I found:**
- 100% test accuracy on a medical dataset with only 128 rows is a red flag, not a success
- Cross-validation gave 97.6% with one fold dropping to 88%, showing the result isn't stable
- Real diabetes data is messy and class-imbalanced. This dataset had patterns too clean and separable to be real
- 128 rows is far too small for any reliable medical classifier

**Conclusion:** The model "succeeded" but the success is meaningless. The dataset is synthetic with artificially clean separations between classes. Suspiciously high accuracy is just as suspicious as suspiciously low accuracy — both signal a problem with the data.

**Lesson:** 100% accuracy on a small medical dataset is a warning sign, not a win. Real medical problems don't have perfectly clean decision boundaries. Always question results that seem too good.

---

## 4. Bone Fracture X-ray Classification

**Goal:** Classify bone fractures into 10 categories from X-ray images

**Model:** Custom CNN from scratch (2 conv blocks, grayscale input)

**Result:** 26.43% accuracy (random chance for 10 classes = 10%)

**What I did:**
- Converted images to grayscale — X-rays are single channel
- Built a custom CNN with 2 conv blocks
- Trained for 30 epochs with Adam
- Got 26.43% — better than random but far from useful

**Why it failed:**
- Only ~989 images for 10 classes — not enough for a CNN to learn from scratch
- Medical X-rays with annotations (arrows, numbers, text added by doctors) create visual noise the model picks up instead of actual fracture patterns
- 10 fine-grained medical categories requires much stronger feature extraction than a simple 2-block CNN can provide

**Conclusion:** This dataset needs Transfer Learning with a pretrained model, not a custom CNN from scratch. The same dataset with ResNet18 would likely perform significantly better. Small medical datasets with noisy annotations are one of the hardest cases for training from scratch.

**Lesson:** Dataset size matters more than model architecture. Transfer Learning exists precisely for cases like this — where you don't have enough data to train meaningful features from zero.

---

## Key Takeaways

- Check correlation before training — a 0.04 ceiling means no model will work
- Scatter plots diagnose fake data faster than any model metric
- Suspiciously high accuracy (100% on 128 rows) is as suspicious as low accuracy
- Small datasets + complex medical tasks = Transfer Learning, not training from scratch
- A negative R² means your model is worse than guessing the mean — always check this
