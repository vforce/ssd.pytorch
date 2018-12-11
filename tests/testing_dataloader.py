from torch.utils.data import DataLoader

from data.IDD import IDDDataset

dataset = IDDDataset('/data/dataset/id-detection-data/training/', 'dataset', 'annotations')
dl = DataLoader(dataset, batch_size=10, shuffle=False)
x, y = next(iter(dl))
print("------ x -------\n", x)
print("------ y -------\n", y)
