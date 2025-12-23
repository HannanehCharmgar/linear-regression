## اضافه کردن کتابخانه ها
NumPy / Pandas برای پردازش و مدیریت داده‌ها

Matplotlib / Seaborn برای ترسیم نمودارها و تحلیل بصری

Scikit-learn برای:

تقسیم داده‌ها

ساخت مدل‌های یادگیری ماشین

ارزیابی عملکرد مدل‌ها

تنظیمات ظاهری نمودارها و حذف هشدارهای غیرضروری

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_curve, auc, roc_auc_score)
```
