# test_load.py
from kagglehub import dataset_load, KaggleDatasetAdapter

df = dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "koki25ando/scotch-whisky-dataset",
    "whisky.csv",  
)

print("Loaded OK")
print("Shape:", df.shape)
print(df.head(5))
