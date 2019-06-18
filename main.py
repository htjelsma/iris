from data_loader import DataLoader

from metrics.accuracy import accuracy
from metrics.f1_score import f1_score
from metrics.log_loss import log_loss

from models.svc.model import Model

features, targets = DataLoader.load_data()

model = Model(log_C=0, kernel='linear', shrinking=1)

model.fit(X=features, y=targets)

print(model.predict(features))
print(log_loss(targets, model.predict(features)))