#!/usr/bin/env python
# coding: utf-8

# In[4]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# ✅ Match input to 2048 fingerprint bits
model = Sequential([
    Input(shape=(2048,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.save("C:/Users/Anshika/Documents/drug_dashboard_project/drug_model.keras")

print("✅ Updated model saved with input shape 2048")


# In[ ]:




