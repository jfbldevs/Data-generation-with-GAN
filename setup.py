import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import deep_tabular_augmentation as dta
import warnings; warnings.simplefilter('ignore')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_PATH = 'file.csv'

df = pd.read_csv(DATA_PATH, sep=',')

df.head()

cols = df.columns

def load_and_standardize_data(path):
    # read in from csv
    df = pd.read_csv(path, sep=',')
    # replace nan with -99
    df = df.fillna(-99)
    df = df.values.reshape(-1, df.shape[1]).astype('float32')
    # randomly split
    X_train, X_test = train_test_split(df, test_size=0.3, random_state=42)
    # standardize values
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)   
    return X_train, X_test, scaler


from torch.utils.data import Dataset, DataLoader
class DataBuilder(Dataset):
    def __init__(self, path, train=True):
        self.X_train, self.X_test, self.standardizer = load_and_standardize_data(DATA_PATH)
        if train:
            self.x = torch.from_numpy(self.X_train)
            self.len=self.x.shape[0]
        else:
            self.x = torch.from_numpy(self.X_test)
            self.len=self.x.shape[0]
        del self.X_train
        del self.X_test 
    def __getitem__(self,index):      
        return self.x[index]
    def __len__(self):
        return self.len

traindata_set=DataBuilder(DATA_PATH, train=True)
testdata_set=DataBuilder(DATA_PATH, train=False)

trainloader=DataLoader(dataset=traindata_set,batch_size=1024)
testloader=DataLoader(dataset=testdata_set,batch_size=1024)

trainloader.dataset.x.shape, testloader.dataset.x.shape

D_in = traindata_set.x.shape[1]
H = 50
H2 = 12

autoenc_model = dta.AutoencoderModel(trainloader, testloader, device, D_in, H, H2, latent_dim=3)

autoenc_model_fit = autoenc_model.fit(epochs=1000)

scaler = trainloader.dataset.standardizer
df_fake = autoenc_model_fit.predict_df(no_samples=1000, scaler=scaler, cols=cols)
df_fake['label'] = np.round(df_fake['label']).astype(int)
df_fake['label'] = np.where(df_fake['label']<1, 1, df_fake['label']) #Set min label
df_fake['label'] = np.where(df_fake['label']>3, 3, df_fake['label']) #Set max label
""" df_fake.head()
print(df_fake) """
#print(list(df_fake[Wine]==1))

""" total=list(df_fake['label']==1)

print(total.count(True))
print(df_fake)
print(df) """
#Save data
""" save_data=df_fake.to_csv('data.csv')
print(save_data) """
