import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- sklearn ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- tensorflow / keras ---
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


def data_frame():
    df = pd.read_csv("heart.csv")

    # Show dataset info
    print("\n--- Data Info ---")
    print(df.info())

    print("\n--- Data Description ---")
    print(df.describe())

    print("\n--- Shape ---")
    print(df.shape)

    # Clean the data (assign results properly)
    df = df.sort_index(ascending=False)
    df = df.drop_duplicates()
    df = df.dropna()
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].astype(str).str.upper().str.strip()

    # Filter male and female datasets with valid ages
    df_male = df[(df['Age'] >= 0) & (df['Age'] < 120) & (df['Sex']== 'M')& (df['Cholesterol'] > 0)]
    df_female = df[(df['Age'] >= 0) & (df['Age'] < 120) &(df['Sex'] == 'F')& (df['Cholesterol'] > 0)]


    # Save cleaned subsets
    df_male.to_csv("heart_male.csv", index=False)
    df_female.to_csv("heart_female.csv", index=False)
    print("\nCleaned male and female datasets saved as 'heart_male.csv' and 'heart_female.csv'.")

    return df_male, df_female

def overlapping_compare(df_male, df_female):
    # Define age bins
    bins = [0, 20, 30, 40, 50, 60, 70, 100]
    labels = ['<20', '20–29', '30–39', '40–49', '50–59', '60–69', '70+']

    # Bin and count
    male_counts = pd.cut(df_male['Age'], bins=bins, labels=labels, right=False).value_counts().sort_index()
    female_counts = pd.cut(df_female['Age'], bins=bins, labels=labels, right=False).value_counts().sort_index()

    
   
    df_heat = pd.DataFrame(
        {
            'Male': male_counts,
            'Female': female_counts
        }
    )
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_heat.T, annot=True,fmt='d',cmap='YlOrRd',cbar_kws={'label': 'Number of People'})
    plt.xlabel('Age Groups')
    plt.ylabel('Sex')
    plt.title('Age Group Comparison: Male vs Female (Heart Dataset)')
    plt.tight_layout()
    plt.show()
    
def data_preprocessing(df):
    """
    Preprocesses numerical and categorical features.
    Returns X (features) and y (target column).
    """
    x = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']


    # Features
    numerical_features = x.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = x.select_dtypes(include=['object']).columns.tolist()
    # Pipelines for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown= 'ignore'))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    X_preprocessed = preprocessor.fit_transform(x)

    # convert to dataframe for readability
    cat_cols = []
    if categorical_features:
        cat_cols = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
    
    X_preprocessed = pd.DataFrame(X_preprocessed, columns=numerical_features + list(cat_cols))
    
    return X_preprocessed, y

def train_and_evaluate(df,dataset_name = 'Dataset'):
    x,y = data_preprocessing(df)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print(f"\n--- {dataset_name} Evaluation ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    return model

def model_design(input_dim):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_nn(df,dataset_name = 'Dataset',epochs=150,batch_size=32,save_model_path=None):
    x,y = data_preprocessing(df)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = model_design(x_train.shape[1])
    history = model.fit(x_train, y_train, epochs=epochs, 
                               batch_size=batch_size,
                        validation_data=(x_test, y_test),
                        verbose=1)
    
    loss , acc = model.evaluate(x_test,y_test,verbose=0)
    print(f"\n--- {dataset_name} Neural Network Evaluation ---")
    print(f"Test Accuracy: {acc:.4f}")
    
    # predictions for both training and test data
    y_train_pred = (model.predict(x_train) > 0.5).astype(int).flatten()
    y_test_pred = (model.predict(x_test) > 0.5).astype(int).flatten()
    
    #  classification report for both
    print(f"\n--- {dataset_name} Training Set Report ---")
    print(classification_report(y_train, y_train_pred))
    print(f"\n--- {dataset_name} Test Set Report ---")
    print(classification_report(y_test, y_test_pred))
    
    # Plot training and validation accuracy
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    if save_model_path:
        model.save(save_model_path)
        print(f"Model saved to {save_model_path}")
    
    return model , history , y_train_pred , y_test_pred
    
    
if __name__ == "__main__":
    # Run everything
    df_male, df_female = data_frame()
    overlapping_compare(df_male, df_female)

    # Logistic Regression
    model_male_lr = train_and_evaluate(df_male, 'Male Dataset')
    model_female_lr = train_and_evaluate(df_female, 'Female Dataset')

    # Neural Network
    model_male_nn, hist_male , y_train_pred_male, y_test_pred_male = train_nn(df_male,
                                    'Male Dataset', epochs=150, batch_size=32, 
                                    save_model_path='male_model.keras')
    model_female_nn, hist_female , y_train_pred_female, y_test_pred_female = train_nn(df_female,
                                    'Female Dataset', epochs=150, batch_size=32, 
                                    save_model_path='female_model.keras')
    # Load the saved models
    male_model = load_model('male_model.keras')
    female_model = load_model('female_model.keras')
    
    

