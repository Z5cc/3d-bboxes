from BB_Dataset import BB_Dataset
from Constants import TEST_PATH



def test_dataset():
    dataset = BB_Dataset(TEST_PATH, aug=True)
    # access first element
    x,y = dataset[0]
    n = len(dataset)
    x,y = dataset[n-1]
